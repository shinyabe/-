from __future__ import annotations # 型ヒントでクラス自身を参照可能にする
from dataclasses import dataclass  # データクラスを使用
from typing import Dict, Tuple, List, Optional  # 型ヒントで使う汎用型をインポート
import math, random                 # 数学関数・乱数生成をインポート

random.seed(42)  # 乱数の再現性を確保（同じ入力で同じ結果になる）

# ===== モード設定（ここだけ切り替えればOK） =====
# 'L' = 根菜のみ, 'S' = 葉物のみ, 'both' = 両方を同時に扱う
MODE = 'S'  # ← 実験対象カテゴリを選択

# === 定数/集合 ===
T = 30                      # シミュレーション日数
G_LIST = ['L','S']          # 商品カテゴリ集合（L=根菜, S=葉物）
S_LIST = [1,2,3]            # 鮮度状態（1=良, 2=劣化, 3=廃棄）
N_LIST = ['W','R']          # 節点（W=卸, R=小売）

# === 価格・費用 ===
P_LIST   = {'L':180.0, 'S':180.0}   # 定価（小売価格）
DELTA    = {'L':0.70,  'S':0.70}    # 値引き率
P_W2R    = {'L':120.0, 'S':120.0}   # 卸→小売の卸売価格
C_BUY    = {'L':90.0,  'S':90.0}    # 卸の仕入原価/個
C_SHIP   = {'L':8.0,   'S':8.0}     # 出荷（配送）費/個
C_DISC   = {'L':60.0,  'S':60.0}    # 値引き運用の固定費/日（カテゴリ単位で発生）
##################################################################################ここまで読んだ########################################################################################
# === 保管コスト（鮮度×節点×カテゴリ）===
C_STOR = {
    ('W','L',1):1.0, ('W','L',2):1.0, ('W','L',3):1.0,    # 卸×根菜の保管費
    ('W','S',1):1.30,('W','S',2):1.30,('W','S',3):1.30,   # 卸×葉物の保管費
    ('R','L',1):1.7, ('R','L',2):1.7, ('R','L',3):1.7,    # 小売×根菜の保管費
    ('R','S',1):3.2, ('R','S',2):3.2, ('R','S',3):3.2,    # 小売×葉物の保管費
}

# === CO2係数（保存/出荷） ===
E_STOR0 = {'W':0.0020,'R':0.0036}   # 通常保存時：在庫1個×日あたりのCO2
E_STOR1 = {'W':0.0040,'R':0.0066}   # 高品質保存時：在庫1個×日あたりのCO2
E_SHIP  = {'L':0.010,'S':0.011}     # 出荷1個あたりのCO2
E_TRIP  = 2.5                       # 1便固定CO2（未使用）
CAP_TRUCK=200                       # トラック容量（未使用）

# === 需要上限（曜日・多周波で“谷”を作り廃棄を誘発しやすく） ===
def _weekday_factor(t: int) -> float:  # 曜日による需要係数
    d = (t-1) % 7                      # t=1 を月曜として 0..6 にマップ
    if d in (2, 6):   # 火・土は弱い
        return 0.78
    if d == 0:        # 月曜もやや弱い
        return 0.90
    return 1.0        # それ以外は標準

def seasonal_full(g: str, t: int) -> float:  # 定価需要の季節波形
    if g == 'L':
        base, amp1, amp2, phase1, phase2 = 40.0, 18.0, 6.0, 0.0, 0.5  # 根菜のパラメータ
    else:  # 'S'
        base, amp1, amp2, phase1, phase2 = 46.0, 22.0, 7.0, 2.0, 1.0  # 葉物のパラメータ
    val = base \
        + amp1 * math.sin(2.0*math.pi*(t+phase1)/7.0) \
        + amp2 * math.sin(2.0*math.pi*(t+phase2)/3.0)
    return max(0.0, _weekday_factor(t) * val)                          # 曜日係数を適用

def seasonal_disc(g: str, t: int) -> float:  # 値引き需要の季節波形
    # ※ 葉物Sの base を下げて定価で売れ残り気味な状況を作る
    if g == 'L':
        base, amp1, amp2, phase1, phase2 = 9.0, 4.0, 2.0, 1.0, 0.0     # 根菜のパラメータ
    else:  # 'S'
        base, amp1, amp2, phase1, phase2 = 11.0, 5.0, 2.0, 3.0, 1.0    # 葉物のパラメータ
    val = base \
        + amp1 * math.sin(2.0*math.pi*(t+phase1)/7.0) \
        + amp2 * math.sin(2.0*math.pi*(t+phase2)/3.0)
    return max(0.0, (0.9 + 0.1*_weekday_factor(t)) * val)             # 値引き需要にも曜日影響

# 需要テーブルを日別に生成
D_FULL = {('L',t): seasonal_full('L',t) for t in range(1, T+1)}        # 根菜の定価需要
D_FULL.update({('S',t): seasonal_full('S',t) for t in range(1, T+1)})  # 葉物の定価需要
D_DISC = {('L',t): seasonal_disc('L',t) for t in range(1, T+1)}        # 根菜の値引き需要
D_DISC.update({('S',t): seasonal_disc('S',t) for t in range(1, T+1)})  # 葉物の値引き需要

# === 初期在庫（序盤から在庫過多→廃棄の可能性を持たせる） ===
I_INIT = { (n,g,s):0.0 for n in N_LIST for g in G_LIST for s in S_LIST }  # 0で初期化
I_INIT[('R','S',1)] = 14.0  # 小売に葉物の鮮度1在庫
I_INIT[('R','L',1)] = 12.0  # 小売に根菜の鮮度1在庫
I_INIT[('W','L',1)] = 8.0   # 卸に根菜の鮮度1在庫

# === 鮮度遷移：小売Rの通常保存を悪化、高品質との差も拡大（Sで強化）
P0_base: Dict[Tuple[str,str,int,int], float] = {}  # 通常保存（theta=0）の遷移確率表
P1_base: Dict[Tuple[str,str,int,int], float] = {}  # 高品質保存（theta=1）の遷移確率表

def set_transitions_for_node(n: str):  # 節点ごとの遷移確率を設定
    # 根菜 L（卸/小売で異なる劣化速度）
    if n == 'W':
        P0_base[(n,'L',1,1)] = 0.87; P0_base[(n,'L',1,2)] = 0.13; P0_base[(n,'L',1,3)] = 0.00
        P0_base[(n,'L',2,1)] = 0.00; P0_base[(n,'L',2,2)] = 0.66; P0_base[(n,'L',2,3)] = 0.34
        P1_base[(n,'L',1,1)] = 0.94; P1_base[(n,'L',1,2)] = 0.06; P1_base[(n,'L',1,3)] = 0.00
        P1_base[(n,'L',2,1)] = 0.00; P1_base[(n,'L',2,2)] = 0.83; P1_base[(n,'L',2,3)] = 0.17
    else:  # R（小売）
        P0_base[(n,'L',1,1)] = 0.72; P0_base[(n,'L',1,2)] = 0.28; P0_base[(n,'L',1,3)] = 0.00
        P0_base[(n,'L',2,1)] = 0.00; P0_base[(n,'L',2,2)] = 0.48; P0_base[(n,'L',2,3)] = 0.52
        P1_base[(n,'L',1,1)] = 0.85; P1_base[(n,'L',1,2)] = 0.15; P1_base[(n,'L',1,3)] = 0.00
        P1_base[(n,'L',2,1)] = 0.00; P1_base[(n,'L',2,2)] = 0.70; P1_base[(n,'L',2,3)] = 0.30

    # 葉物 S（R側の劣化をより強く設定して差を大きく）
    if n == 'W':
        P0_base[(n,'S',1,1)] = 0.73; P0_base[(n,'S',1,2)] = 0.27; P0_base[(n,'S',1,3)] = 0.00
        P0_base[(n,'S',2,1)] = 0.00; P0_base[(n,'S',2,2)] = 0.47; P0_base[(n,'S',2,3)] = 0.53
        P1_base[(n,'S',1,1)] = 0.84; P1_base[(n,'S',1,2)] = 0.16; P1_base[(n,'S',1,3)] = 0.00
        P1_base[(n,'S',2,1)] = 0.00; P1_base[(n,'S',2,2)] = 0.63; P1_base[(n,'S',2,3)] = 0.37
    else:  # R（小売の方が劣化が速い設定）
        P0_base[(n,'S',1,1)] = 0.45; P0_base[(n,'S',1,2)] = 0.55; P0_base[(n,'S',1,3)] = 0.00
        P0_base[(n,'S',2,1)] = 0.00; P0_base[(n,'S',2,2)] = 0.18; P0_base[(n,'S',2,3)] = 0.82
        P1_base[(n,'S',1,1)] = 0.64; P1_base[(n,'S',1,2)] = 0.36; P1_base[(n,'S',1,3)] = 0.00
        P1_base[(n,'S',2,1)] = 0.00; P1_base[(n,'S',2,2)] = 0.40; P1_base[(n,'S',2,3)] = 0.60

for n in N_LIST:            # 卸と小売の2種に対して遷移確率を設定
    set_transitions_for_node(n)

# 鮮度3（廃棄）は3→3=1に固定
for n in N_LIST:            # 全節点で
    for g in G_LIST:        # 全カテゴリで
        P0_base[(n,g,3,1)] = 0.0; P1_base[(n,g,3,1)] = 0.0  # 3→1 なし
        P0_base[(n,g,3,2)] = 0.0; P1_base[(n,g,3,2)] = 0.0  # 3→2 なし
        P0_base[(n,g,3,3)] = 1.0; P1_base[(n,g,3,3)] = 1.0  # 3→3 で留まる

# ===== モードに応じてシミュレーション対象カテゴリを決定 =====
if   MODE == 'L':           # 根菜のみの場合
    ACTIVE_G = ['L']
elif MODE == 'S':           # 葉物のみの場合
    ACTIVE_G = ['S']
else:                       # 両方の場合
    ACTIVE_G = ['L','S']

# === 政策定義 ===
@dataclass(frozen=True)
class MekikiLevel:          # 卸の目利きレベル（劣化確率縮小＆コスト）
    level:str  # 'high' or 'low'
    def quality_boost(self)->float:
        return 0.8 if self.level=='high' else 1.0  # 劣化方向確率の倍率
    def cost_per_unit(self)->float:
        return 5.0 if self.level=='high' else 0.0  # 仕入1個あたり追加コスト

@dataclass(frozen=True)
class StorageMode:          # 保存モード（通常/高品質）
    mode:str # 'high' or 'low'
    def theta(self)->int:
        return 1 if self.mode=='high' else 0       # 1=高品質, 0=通常

@dataclass(frozen=True)
class ShipOrder:            # 出荷順序（棚出し/補充の優先）
    order:str # 'FIFO' or 'LIFO'

@dataclass(frozen=True)
class DiscountPolicy:       # 値引き運用の有無
    use_discount: bool

@dataclass
class DiscretePolicy:       # 一日の政策パラメータのセット
    buy_L: Optional[int]                # 卸の仕入（根菜）※MODEによりNoneあり
    buy_S: Optional[int]                # 卸の仕入（葉物）※MODEによりNoneあり
    safety_target_R_L: Optional[int]    # 小売安全在庫（根菜）
    safety_target_R_S: Optional[int]    # 小売安全在庫（葉物）
    mekiki:MekikiLevel                  # 目利き水準
    storage_W:StorageMode               # 卸の保存モード
    ship_order:ShipOrder                # 出荷順序
    discount:DiscountPolicy             # 値引き運用
    storage_R:StorageMode               # 小売の保存モード

# ==== 鮮度遷移の有効確率 ====
def peff(n,g,sf,st,theta,quality_boost):  # 保存モード・目利き効果を反映した遷移確率
    base = (1-theta)*P0_base[(n,g,sf,st)] + theta*P1_base[(n,g,sf,st)]  # 保存θで線形補間
    if st>sf:  # 劣化方向のみ目利きの縮小効果を適用
        return base * quality_boost
    return base

# ==== 在庫の翌日状態 ====
def inv_next(n,g,inv,arrival,theta,quality_boost):  # 当日在庫+当日入荷の合成を翌日に遷移
    out={1:0.0,2:0.0,3:0.0}                          # 翌日の鮮度別在庫（期待値）
    for sf in S_LIST:                                # 元の鮮度ごとに
        total_sf = inv.get(sf,0.0)+arrival.get(sf,0.0)  # 合算数量
        if total_sf<=0:
            continue
        raw = {st:peff(n,g,sf,st,theta,quality_boost) for st in S_LIST}  # 遷移確率
        ssum = sum(raw.values())                                         # 正規化用合計
        norm = {st:raw[st]/ssum for st in S_LIST}                        # 合計1に正規化
        for st in S_LIST:
            out[st]+=norm[st]*total_sf                                   # 期待値で按分
    return out

# === 結果集約 ===
@dataclass
class SimStats:             # 期間合計の主要KPI
    profit_W:float          # 卸利益
    profit_R:float          # 小売利益
    co2:float               # CO2排出量
    waste:float             # 廃棄数量

# === シミュレータ ===
class Simulator:
    def __init__(self, policy:DiscretePolicy):                          # 政策を受け取り初期化
        self.policy = policy
        self.I_W = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}            # 卸の在庫（鮮度別）
        self.I_R = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}            # 小売の在庫（鮮度別）
        for (n,g,s),v in I_INIT.items():                                # 初期在庫を反映
            if g not in self.I_W:
                continue
            if n=='W': self.I_W[g][s]=v
            else:      self.I_R[g][s]=v
        self.total_profit_W=0.0                                         # 卸利益累積
        self.total_profit_R=0.0                                         # 小売利益累積
        self.total_co2=0.0                                              # CO2累積
        self.total_waste=0.0                                            # 廃棄累積

    # ★変更点(1): 輸送中の劣化確率を導入（カテゴリ別・鮮度別）
    #    (p12[g], p23[g]) = (1→2 に落ちる確率, 2→3 に落ちる確率)
    SHIP_DECAY = {
        'L': (0.03, 0.00),  # 根菜は輸送でほぼ廃棄までは行かない
        'S': (0.06, 0.10),  # 葉物は輸送に弱い
    }

    def run(self)->SimStats:                    # T日分シミュレーションを走らせる
        for t in range(1,T+1):
            self._step_day(t)
        return SimStats(                        # 期間合計の統計を返す
            profit_W=self.total_profit_W,
            profit_R=self.total_profit_R,
            co2=self.total_co2,
            waste=self.total_waste
        )

    def _apply_ship_decay(self, g: str, take1: float, take2: float):  # 出荷時の輸送劣化を適用
        """
        ★変更点(2): 出荷品に輸送劣化を適用して、到着鮮度ミックスを返す
        - 1は p12 で2に落ちる
        - 2は p23 で3(=廃棄)に落ちる（到着時点で廃棄計上）
        """
        p12, p23 = self.SHIP_DECAY[g]           # カテゴリ別の劣化確率を取得
        drop12 = take1 * p12                    # 鮮度1→2に落ちる量
        arr1 = take1 - drop12                   # 到着時の鮮度1数量
        arr2_from1 = drop12                     # 鮮度1から2へ落ちた分
        drop23 = take2 * p23                    # 鮮度2→3(廃棄)に落ちる量
        arr2_from2 = take2 - drop23             # 到着時の鮮度2数量（元2の残り）
        waste_on_arrival = drop23               # 到着時点での廃棄数
        return (arr1, arr2_from1 + arr2_from2), waste_on_arrival  # 到着鮮度(1,2)と廃棄

    def _ship_from_W_to_R(self,g,need):         # 卸→小売補充の際の取り出し順序を反映
        """
        卸→小売の補充で ship_order を反映。
        FIFO: 古い(鮮度2)→新しい(鮮度1)
        LIFO: 新しい(鮮度1)→古い(鮮度2)
        """
        if self.policy.ship_order.order == 'FIFO':  # FIFOの場合
            take2 = min(self.I_W[g][2], need)       # 先に鮮度2を引き当て
            self.I_W[g][2]-=take2
            need-=take2
            take1 = min(self.I_W[g][1], need)       # 次に鮮度1
            self.I_W[g][1]-=take1
            need-=take1
        else:  # 'LIFO'                                 # LIFOの場合
            take1 = min(self.I_W[g][1], need)           # 先に鮮度1を引き当て
            self.I_W[g][1]-=take1
            need-=take1
            take2 = min(self.I_W[g][2], need)           # 次に鮮度2
            self.I_W[g][2]-=take2
            need-=take2

        (arr1, arr2), waste_arrival = self._apply_ship_decay(g, take1, take2)  # 輸送劣化を適用
        return {(g,1):arr1,(g,2):arr2}, (take1+take2), waste_arrival            # 到着鮮度・出荷総量・到着廃棄

    def _step_day(self,t:int):                 # t日目の処理
        pol = self.policy

        # 1) 卸仕入れ
        x_buy = {'L': pol.buy_L if (pol.buy_L is not None) else 0,   # 根菜の仕入数
                 'S': pol.buy_S if (pol.buy_S is not None) else 0}   # 葉物の仕入数
        for g in ACTIVE_G:
            self.I_W[g][1]+=x_buy[g]  # 仕入は鮮度1で卸在庫に積む

        # 2) 卸→小売 出荷（安全在庫目標まで補充）
        target_R: Dict[str,int] = {}  # カテゴリ別の安全在庫目標
        for g in ACTIVE_G:
            if g=='L':
                assert pol.safety_target_R_L is not None  # 根菜が対象なら必須
                target_R[g] = pol.safety_target_R_L
            else:
                assert pol.safety_target_R_S is not None  # 葉物が対象なら必須
                target_R[g] = pol.safety_target_R_S

        x_ship = {(g,s):0.0 for g in ACTIVE_G for s in S_LIST}  # 出荷量（卸→小売、鮮度別）
        waste_on_arrival_total = 0.0                             # 到着時廃棄の合計

        for g in ACTIVE_G:
            need = max(0.0, target_R[g]-(self.I_R[g][1]+self.I_R[g][2]))  # 必要補充量
            taken, total_taken, waste_arrival = self._ship_from_W_to_R(g,need)  # 出荷と輸送劣化
            x_ship[(g,1)] = taken[(g,1)]            # 到着した鮮度1
            x_ship[(g,2)] = taken[(g,2)]            # 到着した鮮度2
            waste_on_arrival_total += waste_arrival # 到着時廃棄を加算
            self.I_R[g][1]+=taken[(g,1)]            # 小売在庫へ反映（鮮度1）
            self.I_R[g][2]+=taken[(g,2)]            # 小売在庫へ反映（鮮度2）

        # 3) 小売販売：ship_order を定価販売順序に反映
        #    FIFO: 鮮度2→鮮度1（古いものから売る）
        #    LIFO: 鮮度1→鮮度2（新しいものから売る）
        z_disc = {g: 1 if (pol.discount.use_discount and self.I_R[g][2]>0) else 0
                  for g in ACTIVE_G}                             # 値引き運用フラグ
        x_sell_full={(g,s):0.0 for g in ACTIVE_G for s in [1,2]} # 定価販売数（鮮度別）
        x_sell_disc={(g,2):0.0 for g in ACTIVE_G}                # 値引き販売数（鮮度2のみ）

        for g in ACTIVE_G:
            demand_full = D_FULL[(g,t)]                          # 当日の定価需要
            if pol.ship_order.order == 'FIFO':                   # FIFO販売順序
                sell2_full = min(self.I_R[g][2], demand_full)    # 鮮度2を優先して定価販売
                x_sell_full[(g,2)] = sell2_full
                self.I_R[g][2]-=sell2_full
                demand_full-=sell2_full

                sell1 = min(self.I_R[g][1], max(0.0,demand_full))# 残需要に鮮度1を販売
                x_sell_full[(g,1)] = sell1
                self.I_R[g][1]-=sell1
            else:                                                # LIFO販売順序
                sell1 = min(self.I_R[g][1], demand_full)         # 鮮度1を優先して定価販売
                x_sell_full[(g,1)] = sell1
                self.I_R[g][1]-=sell1
                demand_full-=sell1

                sell2_full = min(self.I_R[g][2], max(0.0,demand_full))  # 残需要に鮮度2
                x_sell_full[(g,2)] = sell2_full
                self.I_R[g][2]-=sell2_full

            if z_disc[g]==1:                                     # 値引き販売を実施するなら
                demand_disc = D_DISC[(g,t)]                      # 値引き需要
                sell2_disc = min(self.I_R[g][2], demand_disc)    # 鮮度2のみ値引き販売
                x_sell_disc[(g,2)] = sell2_disc
                self.I_R[g][2]-=sell2_disc

        # 4) 劣化・廃棄（当日終了時点の在庫を翌日に遷移）
        theta_W = pol.storage_W.theta()                          # 卸の保存モード指標
        theta_R = pol.storage_R.theta()                          # 小売の保存モード指標
        I_W_next = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}     # 翌日の卸在庫
        I_R_next = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}     # 翌日の小売在庫
        waste_today = 0.0                                        # 当日の廃棄合計

        for g in ACTIVE_G:
            I_W_next[g] = inv_next('W',g,self.I_W[g],{1:0.0,2:0.0,3:0.0},theta_W,pol.mekiki.quality_boost())  # 卸の在庫劣化
            waste_today += I_W_next[g][3]; I_W_next[g][3]=0.0    # 鮮度3は即廃棄に計上し在庫から除去

            I_R_next[g] = inv_next('R',g,self.I_R[g],{1:0.0,2:0.0,3:0.0},theta_R,1.0)  # 小売の在庫劣化（目利き効果なし）
            waste_today += I_R_next[g][3]; I_R_next[g][3]=0.0    # 鮮度3を廃棄に加算

        self.I_W = I_W_next                                      # 卸在庫を翌日に更新
        self.I_R = I_R_next                                      # 小売在庫を翌日に更新

        # 5) 利益・CO2・廃棄集計（当日分を累積）
        ship_qty_g = {g:x_ship[(g,1)]+x_ship[(g,2)] for g in ACTIVE_G}  # 出荷合計（卸→小売）

        # 卸の損益
        rev_W  = sum(P_W2R[g]*ship_qty_g[g] for g in ACTIVE_G)          # 卸売上
        buy_W  = sum(C_BUY[g]*x_buy[g]      for g in ACTIVE_G)          # 卸仕入
        mekiki_cost = sum(pol.mekiki.cost_per_unit()*x_buy[g] for g in ACTIVE_G)  # 目利きコスト
        stor_W = sum(C_STOR[('W',g,s)]*self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)  # 卸保管費
        ship_c = sum(C_SHIP[g]*ship_qty_g[g]for g in ACTIVE_G)          # 配送費
        profit_W_day = rev_W - buy_W - mekiki_cost - ship_c - stor_W    # 卸当日利益

        # 小売の損益
        rev_R_full = sum(P_LIST[g]*x_sell_full[(g,1)] for g in ACTIVE_G) \
                   + sum(P_LIST[g]*x_sell_full[(g,2)] for g in ACTIVE_G) # 定価売上
        rev_R_disc = sum(DELTA[g]*P_LIST[g]*x_sell_disc[(g,2)] for g in ACTIVE_G) # 値引き売上
        cost_pur   = sum(P_W2R[g]*ship_qty_g[g] for g in ACTIVE_G)       # 仕入原価
        stor_R     = sum(C_STOR[('R',g,s)]*self.I_R[g][s] for g in ACTIVE_G for s in S_LIST) # 小売保管費
        disc_fix   = sum(C_DISC[g]*z_disc[g] for g in ACTIVE_G)          # 値引き固定費（起動カテゴリ分）
        profit_R_day = (rev_R_full+rev_R_disc) - cost_pur - stor_R - disc_fix  # 小売当日利益

        # CO2（日次）
        co2_ship = sum(E_SHIP[g]*ship_qty_g[g] for g in ACTIVE_G)        # 出荷由来CO2
        co2_stor = (((1-theta_W)*E_STOR0['W']+theta_W*E_STOR1['W']) *    # 卸保存由来CO2
                    sum(self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)
                  + ((1-theta_R)*E_STOR0['R']+theta_R*E_STOR1['R']) *    # 小売保存由来CO2
                    sum(self.I_R[g][s] for g in ACTIVE_G for s in S_LIST))
        co2_day = co2_ship + co2_stor                                    # 当日CO2合計

        # 到着時廃棄も合算して累積へ反映
        self.total_profit_W += profit_W_day
        self.total_profit_R += profit_R_day
        self.total_waste    += (waste_today + waste_on_arrival_total)
        self.total_co2      += co2_day

# ====== 仕入れ候補（モード依存） ======
def get_buy_candidates() -> Dict[str, List[int]]:  # 仕入れ数量の候補集合を返す
    base_L = [10,20,30,40,50,60,70,80,90,100]      # 根菜候補（等間隔）
    base_S = [12,24,36,48,60,72,84,96,108,120]     # 葉物候補（非等間隔寄り）
    if MODE == 'L':
        return {'L': base_L, 'S': []}              # 根菜のみ有効
    elif MODE == 'S':
        return {'L': [],   'S': base_S}            # 葉物のみ有効
    else:
        return {'L': base_L, 'S': base_S}          # 両方有効

# ====== 小売の安全在庫候補（モード依存） ======
def get_safety_candidates() -> Dict[str, List[int]]:  # 安全在庫目標の候補集合
    inv_L = [20,40,60,80,100,120,140,160,180,200]     # 根菜候補（等間隔）
    inv_S = [22,46,70,94,118,142,166,190]             # 葉物候補（非等間隔）
    if MODE == 'L':
        return {'L': inv_L, 'S': []}                  # 根菜のみ有効
    elif MODE == 'S':
        return {'L': [],           'S': inv_S}        # 葉物のみ有効
    else:
        return {'L': inv_L,  'S': inv_S}              # 両方有効

# ====== 評価関数 ======
def evaluate_metrics(pol: DiscretePolicy):  # 政策を30日走らせKPIを計算
    sim = Simulator(pol).run()              # シミュレータ実行
    return {
        "profit_total": sim.profit_W + sim.profit_R,  # 卸+小売の合計利益
        "profit_W": sim.profit_W,                     # 卸利益
        "profit_R": sim.profit_R,                     # 小売利益
        "co2": sim.co2,                               # CO2
        "waste": sim.waste,                           # 廃棄
        "policy": pol                                 # どの政策かを同梱
    }

# ====== ランダム政策 ======
def random_discrete_policy()->DiscretePolicy:  # 候補集合からランダムに政策を1つ生成
    buy_cands = get_buy_candidates()           # 仕入候補
    saf_cands = get_safety_candidates()        # 安全在庫候補
    buy_L = random.choice(buy_cands['L']) if buy_cands['L'] else None   # 根菜仕入（モード外ならNone）
    buy_S = random.choice(buy_cands['S']) if buy_cands['S'] else None   # 葉物仕入（モード外ならNone）
    saf_L = random.choice(saf_cands['L']) if saf_cands['L'] else None   # 根菜安全在庫
    saf_S = random.choice(saf_cands['S']) if saf_cands['S'] else None   # 葉物安全在庫
    pol = DiscretePolicy(
        buy_L=buy_L,
        buy_S=buy_S,
        safety_target_R_L=saf_L,
        safety_target_R_S=saf_S,
        mekiki=MekikiLevel(level=random.choice(['high','low'])),         # 目利き（高/低）
        storage_W=StorageMode(mode=random.choice(['high','low'])),       # 卸保存（高/低）
        ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),      # 出荷順序
        discount=DiscountPolicy(use_discount=random.choice([True,False])),# 値引き運用
        storage_R=StorageMode(mode=random.choice(['high','low'])),       # 小売保存（高/低）
    )
    return pol

# ====== 近傍生成 ======
def neighborhood(pol:DiscretePolicy)->DiscretePolicy:  # 現在の政策から±1段だけずらす
    def nudge(val, candidates):                         # 候補配列内で±1インデックスへ移動
        if not candidates:
            return None
        if val is None:
            return None
        idx = candidates.index(val)
        idx2 = max(0, min(len(candidates)-1, idx + random.choice([-1,1])))
        return candidates[idx2]

    buy_cands = get_buy_candidates()                    # 仕入候補集合を取得
    saf_cands = get_safety_candidates()                 # 安全在庫候補集合を取得

    next_buy_L = nudge(pol.buy_L, buy_cands['L'])       # 根菜仕入を±1段
    next_buy_S = nudge(pol.buy_S, buy_cands['S'])       # 葉物仕入を±1段
    next_saf_L = nudge(pol.safety_target_R_L, saf_cands['L'])  # 根菜安全在庫を±1段
    next_saf_S = nudge(pol.safety_target_R_S, saf_cands['S'])  # 葉物安全在庫を±1段

    return DiscretePolicy(
        buy_L=next_buy_L,
        buy_S=next_buy_S,
        safety_target_R_L=next_saf_L,
        safety_target_R_S=next_saf_S,
        mekiki=MekikiLevel(level=pol.mekiki.level if random.random()<0.8 else random.choice(['high','low'])),  # 20%で切替
        storage_W=StorageMode(mode=pol.storage_W.mode if random.random()<0.8 else random.choice(['high','low'])),
        ship_order=ShipOrder(order=pol.ship_order.order if random.random()<0.8 else random.choice(['FIFO','LIFO'])),
        discount=DiscountPolicy(use_discount=pol.discount.use_discount if random.random()<0.8 else (not pol.discount.use_discount)),
        storage_R=StorageMode(mode=pol.storage_R.mode if random.random()<0.8 else random.choice(['high','low'])),
    )

# ====== 政策キー ======
def policy_key(pol: DiscretePolicy) -> Tuple:  # 政策の同一性判定用キーを生成
    key: List[object] = [
        pol.mekiki.level, pol.storage_W.mode,        # 目利き/卸保存
        pol.ship_order.order, pol.discount.use_discount,  # 出荷順序/値引き運用
        pol.storage_R.mode                           # 小売保存
    ]
    if MODE in ('both','L'):
        key.insert(0, pol.buy_L)                    # 根菜仕入
        key.insert(1, pol.safety_target_R_L)        # 根菜安全在庫
    if MODE in ('both','S'):
        key.append(pol.buy_S)                       # 葉物仕入
        key.append(pol.safety_target_R_S)           # 葉物安全在庫
    return tuple(key)

# ====== 極値探索 ======
def search_best_extrema(num_init=50, num_neigh=20, tol: float = 1e-9):  # ランダム初期＋近傍で極値を探索
    best_profit_val = None                  # 利益最大の値
    best_profit_set: Dict[Tuple, dict] = {}# 利益最大の政策セット
    best_co2_val = None                     # CO2最小の値
    best_co2_set: Dict[Tuple, dict] = {}    # CO2最小の政策セット
    best_waste_val = None                   # 廃棄最小の値
    best_waste_set: Dict[Tuple, dict] = {}  # 廃棄最小の政策セット

    def update_sets(res):                   # 新しい結果で各極値集合を更新
        nonlocal best_profit_val, best_co2_val, best_waste_val
        key = policy_key(res["policy"])     # キー化して同一政策の重複登録を防止
        # 利益（最大化）
        if best_profit_val is None or (res["profit_total"] > best_profit_val + tol):
            best_profit_val = res["profit_total"]; best_profit_set.clear(); best_profit_set[key] = res
        elif abs(res["profit_total"] - best_profit_val) <= tol:
            pass                            # tol以内の同値は登録抑制（同値を減らす）
        # CO2（最小化）
        if best_co2_val is None or (res["co2"] < best_co2_val - tol):
            best_co2_val = res["co2"]; best_co2_set.clear(); best_co2_set[key] = res
        elif abs(res["co2"] - best_co2_val) <= tol:
            pass
        # 廃棄（最小化）
        if best_waste_val is None or (res["waste"] < best_waste_val - tol):
            best_waste_val = res["waste"]; best_waste_set.clear(); best_waste_set[key] = res
        elif abs(res["waste"] - best_waste_val) <= tol:
            pass

    for _ in range(num_init):               # 初期サンプルを評価
        pol = random_discrete_policy()
        res = evaluate_metrics(pol)
        update_sets(res)

    for _ in range(num_neigh):              # 近傍探索：代表から±1段で改善探索
        for rep in list(best_profit_set.values()):
            res = evaluate_metrics(neighborhood(rep["policy"])); update_sets(res)
        for rep in list(best_co2_set.values()):
            res = evaluate_metrics(neighborhood(rep["policy"])); update_sets(res)
        for rep in list(best_waste_set.values()):
            res = evaluate_metrics(neighborhood(rep["policy"])); update_sets(res)

    return {                                # 3目的それぞれの極値セットを返す
        "profit_max": list(best_profit_set.values()),
        "co2_min":    list(best_co2_set.values()),
        "waste_min":  list(best_waste_set.values())
    }

# ====== 出力ユーティリティ ======
def print_policy_block(idx:int, title:str, res:dict):  # 1政策ぶんを読みやすく表示
    pol = res["policy"]
    print("========================================")
    print(f"【{title} #{idx}】（MODE = {MODE}）")
    print("----------------------------------------")
    print(f"利益合計(卸+小売): {res['profit_total']:.2f} 円")
    print(f"  卸の利益: {res['profit_W']:.2f} 円")
    print(f"  小売の利益: {res['profit_R']:.2f} 円")
    print(f"CO2排出量合計: {res['co2']:.3f} kg")
    print(f"廃棄量合計: {res['waste']:.2f} 個")
    print("\n【政策パラメータ】")
    if 'L' in ACTIVE_G:
        print(f"- 卸の1日仕入（根菜L）: {pol.buy_L} 個/日")
        print(f"- 小売の安全在庫（L） : {pol.safety_target_R_L} 個")
    if 'S' in ACTIVE_G:
        print(f"- 卸の1日仕入（葉物S）: {pol.buy_S} 個/日")
        print(f"- 小売の安全在庫（S） : {pol.safety_target_R_S} 個")
    print(f"- 卸の目利きレベル     : {'高い' if pol.mekiki.level=='high' else '低い'}")
    print(f"- 卸の保存モード       : {'高品質' if pol.storage_W.mode=='high' else '通常'}")
    print(f"- 出荷順序             : {pol.ship_order.order}")
    print(f"- 小売の値引き運用     : {'あり' if pol.discount.use_discount else 'なし'}")
    print(f"- 小売の保存モード     : {'高品質' if pol.storage_R.mode=='high' else '通常'}")
    print("========================================\n")

def print_policies(title:str, results:List[dict]):     # 複数政策を順次表示
    if not results:
        print(f"【{title}】該当なし\n"); return
    for i, res in enumerate(results, 1):
        print_policy_block(i, title, res)

# ====== メイン ======
if __name__=="__main__":                                # スクリプトとして実行されたときのエントリ
    best_sets = search_best_extrema(num_init=50, num_neigh=20, tol=1e-9)  # 極値探索を実行
    print_policies("利益最大の戦略（同値を全て表示）", best_sets["profit_max"])   # 利益最大の結果を表示
    print_policies("CO2最小の戦略（同値を全て表示）", best_sets["co2_min"])    # CO2最小は必要時に表示
    print_policies("廃棄最小の戦略（同値を全て表示）", best_sets["waste_min"])  # 廃棄最小は必要時に表示
