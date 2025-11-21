from __future__ import annotations  # 型ヒントでクラス自身を参照可能にするための将来の機能をインポート
from dataclasses import dataclass   # データクラスを定義するためのモジュールをインポート
from typing import Dict, Tuple, List, Optional  # 型ヒント用の型をインポート
import math, random                 # 数学関数と乱数生成モジュールをインポート

import numpy as np                  # 数値計算ライブラリをインポート
import matplotlib.pyplot as plt     # グラフ描画ライブラリをインポート
import matplotlib as mpl            # matplotlibの基本設定をインポート

# ====== 日本語フォント設定 ======
mpl.rcParams['font.family'] = 'Hiragino Sans'     # 日本語フォント設定（Mac）
mpl.rcParams['axes.unicode_minus'] = False        # マイナス記号が tofu になるのを防ぐ

# ====== 共通設定 ======
random.seed(42)  # 乱数シードを固定（再現性のため）

# モード: 'L' 根菜のみ, 'S' 葉物のみ, 'both' 両方
MODE = 'L'  # シミュレーションモードを設定

T = 30  # シミュレーション日数
G_LIST = ['L', 'S']  # 商品カテゴリ集合：L=根菜、S=葉物
S_LIST = [1, 2, 3]   # 鮮度状態集合：1=良、2=劣化、3=廃棄対象
N_LIST = ['W', 'R']  # サプライチェーン節点：W=卸、R=小売

# 価格・費用
P_LIST   = {'L':180.0, 'S':180.0}  # 小売定価(円/個)
DELTA    = {'L':0.70,  'S':0.70}   # 値引き率：定価×DELTAが割引価格
P_W2R    = {'L':120.0, 'S':120.0}  # 卸→小売の卸売価格
C_BUY    = {'L':90.0,  'S':90.0}   # 卸の仕入原価
C_SHIP   = {'L':7.0,   'S':7.0}    # 卸→小売の配送費（1個あたり）
C_DISC   = {'L':300.0, 'S':300.0}  # 小売の値引き運用固定費（カテゴリ×日）

# 保管費（円/個/日）※「高品質保存モード」のときだけ有効な係数として解釈
C_STOR = {  # 節点・カテゴリ・鮮度ごとの在庫1個×日あたりの保管コスト
    ('W','L',1):1.0, ('W','L',2):1.0, ('W','L',3):1.0,    # 卸×根菜の保管費
    ('W','S',1):1.30,('W','S',2):1.30,('W','S',3):1.30,   # 卸×葉物の保管費
    ('R','L',1):1.7, ('R','L',2):1.7, ('R','L',3):1.7,    # 小売×根菜の保管費
    ('R','S',1):3.2, ('R','S',2):3.2, ('R','S',3):3.2,    # 小売×葉物の保管費
}

# CO2 係数（kg/個/日・kg/個）
E_STOR0 = {'W':0.0020,'R':0.0036}  # 通常保存（low）の場合のCO2排出係数（1個×日）
E_STOR1 = {'W':0.0040,'R':0.0066}  # 高品質保存（high）の場合のCO2排出係数（1個×日）
E_SHIP  = {'L':0.010,'S':0.011}    # 出荷1個あたりのCO2排出量（輸送由来）
E_TRIP  = 2.5                      # 1便あたりの固定CO2（ここでは未使用）
CAP_TRUCK = 200                    # トラック容量（ここでは未使用）

# ====== ペナルティ係数 ======
# 「利得 = 利益 − ペナルティ」の形でゲームを定義
PENALTY_CO2_W   = 100.0  # 卸CO2ペナルティ [円/kg-CO2]
PENALTY_CO2_R   = 100.0  # 小売CO2ペナルティ [円/kg-CO2]
PENALTY_WASTE_W = 20.0    # 卸廃棄ペナルティ [円/個]
PENALTY_WASTE_R = 20.0    # 小売廃棄ペナルティ [円/個]

# ★ 新しく追加：欠品（売り逃し）ペナルティ
PENALTY_LOST_W  = 0.0    # 卸側の欠品ペナルティ（今回は未使用）
PENALTY_LOST_R  = 50.0   # 小売側の売り逃しペナルティ [円/個]

ALPHA_SHIP_CO2   = 0.5   # 輸送CO2のうち何割を卸側負担とみなすか
BETA_SHIP_WASTE  = 0.5   # 輸送廃棄のうち何割を卸側負担とみなすか

# =========================================================
# === 需要生成（梶木裕斗 卒論のモデルに基づく形へ変更）===
# =========================================================
# 卒論の定数：
#   ・µC：一日当たりの小売店における客入り平均数 = 500
#   ・σC：一日当たりの小売店における客入り標準偏差 = √50
#   ・ci：i番目の消費者の購入数 = 1
#   → 1日の需要 D = Σ ci ≒ 来店客数 I に対応  

# カテゴリ別の客入り平均・標準偏差（2カテゴリ合計で元の N(500, 50) になるように設定）
MU_C = {'L': 250.0, 'S': 250.0}               # 各カテゴリの客入り平均数
SIGMA_C = {'L': math.sqrt(25.0), 'S': math.sqrt(25.0)}  # 各カテゴリの客入り標準偏差（分散25）

C_PER_CUSTOMER = 1.0   # ci：1人あたり購入数
DISC_SHARE = 0.2       # 総需要のうち「値引き棚」に向かう割合（素朴な近似）

# 需要テーブル（後で build_demand_tables で中身を埋める）
D_FULL: Dict[Tuple[str,int], float] = {}
D_DISC: Dict[Tuple[str,int], float] = {}

def build_demand_tables(T: int = 30):
    """梶木くんの需要モデルに基づき、30日分の定価需要・値引き需要をカテゴリごとに生成する"""
    global D_FULL, D_DISC
    D_FULL = {}
    D_DISC = {}

    # モジュールロード時の random.seed(42) の状態から順にサンプル
    for t in range(1, T+1):
        for g in G_LIST:
            # 正規分布 N(µC_g, σC_g^2) から来店客数をサンプルし、負値は0にクリップ
            I_g = random.gauss(MU_C[g], SIGMA_C[g])
            I_g = max(0.0, I_g)

            # 一人あたり購入数 ci=1 として需要量に変換
            D_total = I_g * C_PER_CUSTOMER

            # 定価需要と値引き需要に単純に分割
            full = (1.0 - DISC_SHARE) * D_total
            disc = DISC_SHARE * D_total

            D_FULL[(g, t)] = full
            D_DISC[(g, t)] = disc

# 需要テーブルを一度だけ構築
build_demand_tables(T)

# ====== 需要プロット用関数 ======
def plot_demands(T: int = 30):
    """30日分の定価需要・値引き需要をカテゴリごとにプロット"""
    days = range(1, T+1)

    # 根菜 L の定価・値引き需要
    L_full = [D_FULL[('L', t)] for t in days]
    L_disc = [D_DISC[('L', t)] for t in days]

    # 葉物 S の定価・値引き需要
    S_full = [D_FULL[('S', t)] for t in days]
    S_disc = [D_DISC[('S', t)] for t in days]

    # 根菜 L のグラフ
    plt.figure(figsize=(8, 4))
    plt.plot(days, L_full, label="根菜L: 定価需要")
    plt.plot(days, L_disc, label="根菜L: 値引き需要")
    plt.xlabel("日数 t")
    plt.ylabel("需要量（個）")
    plt.title("根菜 L の30日間の需要パターン（梶木モデルベース）")
    plt.legend()
    plt.tight_layout()

    # 葉物 S のグラフ
    plt.figure(figsize=(8, 4))
    plt.plot(days, S_full, label="葉物S: 定価需要")
    plt.plot(days, S_disc, label="葉物S: 値引き需要")
    plt.xlabel("日数 t")
    plt.ylabel("需要量（個）")
    plt.title("葉物 S の30日間の需要パターン（梶木モデルベース）")
    plt.legend()
    plt.tight_layout()

    plt.show()

# 初期在庫
I_INIT = {(n,g,s):0.0 for n in N_LIST for g in G_LIST for s in S_LIST}
I_INIT[('R','S',1)] = 14.0
I_INIT[('R','L',1)] = 12.0
I_INIT[('W','L',1)] = 8.0

# 鮮度遷移確率テーブル（P0:通常保存, P1:高品質保存）
P0_base: Dict[Tuple[str,str,int,int], float] = {}
P1_base: Dict[Tuple[str,str,int,int], float] = {}

def set_transitions_for_node(n: str):
    """卸(W)と小売(R)それぞれについて鮮度遷移確率を設定"""
    if n == 'W':  # 卸の場合（根菜）
        P0_base[(n,'L',1,1)] = 0.87; P0_base[(n,'L',1,2)] = 0.13; P0_base[(n,'L',1,3)] = 0.00
        P0_base[(n,'L',2,1)] = 0.00; P0_base[(n,'L',2,2)] = 0.66; P0_base[(n,'L',2,3)] = 0.34
        P1_base[(n,'L',1,1)] = 0.94; P1_base[(n,'L',1,2)] = 0.06; P1_base[(n,'L',1,3)] = 0.00
        P1_base[(n,'L',2,1)] = 0.00; P1_base[(n,'L',2,2)] = 0.83; P1_base[(n,'L',2,3)] = 0.17
    else:         # 小売の場合（根菜）
        P0_base[(n,'L',1,1)] = 0.72; P0_base[(n,'L',1,2)] = 0.28; P0_base[(n,'L',1,3)] = 0.00
        P0_base[(n,'L',2,1)] = 0.00; P0_base[(n,'L',2,2)] = 0.48; P0_base[(n,'L',2,3)] = 0.52
        P1_base[(n,'L',1,1)] = 0.85; P1_base[(n,'L',1,2)] = 0.15; P1_base[(n,'L',1,3)] = 0.00
        P1_base[(n,'L',2,1)] = 0.00; P1_base[(n,'L',2,2)] = 0.70; P1_base[(n,'L',2,3)] = 0.30
    if n == 'W':  # 卸の場合（葉物）
        P0_base[(n,'S',1,1)] = 0.73; P0_base[(n,'S',1,2)] = 0.27; P0_base[(n,'S',1,3)] = 0.00
        P0_base[(n,'S',2,1)] = 0.00; P0_base[(n,'S',2,2)] = 0.47; P0_base[(n,'S',2,3)] = 0.53
        P1_base[(n,'S',1,1)] = 0.84; P1_base[(n,'S',1,2)] = 0.16; P1_base[(n,'S',1,3)] = 0.00
        P1_base[(n,'S',2,1)] = 0.00; P1_base[(n,'S',2,2)] = 0.63; P1_base[(n,'S',2,3)] = 0.37
    else:         # 小売の場合（葉物）
        P0_base[(n,'S',1,1)] = 0.45; P0_base[(n,'S',1,2)] = 0.55; P0_base[(n,'S',1,3)] = 0.00
        P0_base[(n,'S',2,1)] = 0.00; P0_base[(n,'S',2,2)] = 0.18; P0_base[(n,'S',2,3)] = 0.82
        P1_base[(n,'S',1,1)] = 0.64; P1_base[(n,'S',1,2)] = 0.36; P1_base[(n,'S',1,3)] = 0.00
        P1_base[(n,'S',2,1)] = 0.00; P1_base[(n,'S',2,2)] = 0.40; P1_base[(n,'S',2,3)] = 0.60

for n in N_LIST:
    set_transitions_for_node(n)

for n in N_LIST:
    for g in G_LIST:
        P0_base[(n,g,3,1)] = 0.0; P1_base[(n,g,3,1)] = 0.0
        P0_base[(n,g,3,2)] = 0.0; P1_base[(n,g,3,2)] = 0.0
        P0_base[(n,g,3,3)] = 1.0; P1_base[(n,g,3,3)] = 1.0

# 対象カテゴリ（ACTIVE_G）をモードで制御
if   MODE == 'L': ACTIVE_G = ['L']
elif MODE == 'S': ACTIVE_G = ['S']
else:             ACTIVE_G = ['L','S']

# ====== 政策（戦略）定義 ======
@dataclass(frozen=True)
class MekikiLevel:
    level: str      # 'high' or 'low'
    def quality_boost(self)->float:
        return 0.8 if self.level=='high' else 1.0
    def cost_per_unit(self)->float:
        return 5.0 if self.level=='high' else 0.0

@dataclass(frozen=True)
class StorageMode:
    mode: str       # 'high' or 'low'
    def theta(self)->int:
        return 1 if self.mode=='high' else 0

@dataclass(frozen=True)
class ShipOrder:
    order: str      # 'FIFO' or 'LIFO'

@dataclass(frozen=True)
class DiscountPolicy:
    use_discount: bool

@dataclass
class DiscretePolicy:
    # 卸
    safety_target_W_L: Optional[int]
    safety_target_W_S: Optional[int]
    mekiki: MekikiLevel
    storage_W: StorageMode
    ship_order: ShipOrder
    # 小売
    safety_target_R_L: Optional[int]
    safety_target_R_S: Optional[int]
    discount: DiscountPolicy
    storage_R: StorageMode

# ====== シミュレーションの基本関数 ======
def peff(n,g,sf,st,theta,quality_boost):
    base = (1-theta)*P0_base[(n,g,sf,st)] + theta*P1_base[(n,g,sf,st)]
    if st>sf:  # 劣化方向
        return base * quality_boost
    return base

def inv_next(n,g,inv,arrival,theta,quality_boost):
    out={1:0.0,2:0.0,3:0.0}
    for sf in S_LIST:
        total_sf = inv.get(sf,0.0)+arrival.get(sf,0.0)
        if total_sf<=0:
            continue
        raw = {st:peff(n,g,sf,st,theta,quality_boost) for st in S_LIST}
        ssum = sum(raw.values())
        norm = {st:raw[st]/ssum for st in S_LIST}
        for st in S_LIST:
            out[st]+=norm[st]*total_sf
    return out

@dataclass
class SimStats:
    profit_W:float
    profit_R:float
    co2_total:float
    waste_total:float
    co2_W:float
    co2_R:float
    co2_ship:float
    waste_W:float
    waste_R:float
    waste_ship:float
    # ★ 新しく追加：小売の売り逃し個数
    lost_R:float

class Simulator:
    SHIP_DECAY = {'L': (0.03, 0.00), 'S': (0.06, 0.10)}  # 1→2, 2→3

    def __init__(self, policy:DiscretePolicy):
        self.policy = policy
        self.I_W = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}
        self.I_R = {g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}
        for (n,g,s),v in I_INIT.items():
            if g not in self.I_W:
                continue
            if n=='W': self.I_W[g][s]=v
            else:      self.I_R[g][s]=v

        self.total_profit_W=0.0
        self.total_profit_R=0.0
        self.total_co2=0.0
        self.total_waste=0.0

        self.co2_W=0.0
        self.co2_R=0.0
        self.co2_ship=0.0
        self.waste_W=0.0
        self.waste_R=0.0
        self.waste_ship=0.0

        # ★ 新しく追加：売り逃し（欠品）カウンタ
        self.lost_R=0.0

    def _apply_ship_decay(self, g: str, take1: float, take2: float):
        p12, p23 = self.SHIP_DECAY[g]
        drop12 = take1 * p12
        arr1 = take1 - drop12
        arr2_from1 = drop12
        drop23 = take2 * p23
        arr2_from2 = take2 - drop23
        waste_on_arrival = drop23
        return (arr1, arr2_from1 + arr2_from2), waste_on_arrival

    def _ship_from_W_to_R(self,g,need):
        if self.policy.ship_order.order == 'FIFO':
            take2 = min(self.I_W[g][2], need); self.I_W[g][2]-=take2; need-=take2
            take1 = min(self.I_W[g][1], need); self.I_W[g][1]-=take1; need-=take1
        else:
            take1 = min(self.I_W[g][1], need); self.I_W[g][1]-=take1; need-=take1
            take2 = min(self.I_W[g][2], need); self.I_W[g][2]-=take2; need-=take2
        (arr1, arr2), waste_arrival = self._apply_ship_decay(g, take1, take2)
        return {(g,1):arr1,(g,2):arr2}, (take1+take2), waste_arrival

    def run(self)->SimStats:
        pol = self.policy
        for t in range(1,T+1):

            # 1) 卸仕入
            def w_stock(g): return self.I_W[g][1]+self.I_W[g][2]
            target_W={}
            for g in ACTIVE_G:
                if g=='L': target_W[g]=pol.safety_target_W_L
                else:      target_W[g]=pol.safety_target_W_S
                assert target_W[g] is not None

            x_buy={'L':0,'S':0}
            for g in ACTIVE_G:
                gap = max(0.0, target_W[g]-w_stock(g))
                x_buy[g] = int(math.ceil(gap))
                self.I_W[g][1]+=x_buy[g]

            # 2) 卸->小売 出荷
            target_R={}
            for g in ACTIVE_G:
                if g=='L': target_R[g]=pol.safety_target_R_L
                else:      target_R[g]=pol.safety_target_R_S
                assert target_R[g] is not None

            x_ship = {(g,s):0.0 for g in ACTIVE_G for s in S_LIST}
            waste_on_arrival_total=0.0
            ship_qty_g={g:0.0 for g in ACTIVE_G}
            for g in ACTIVE_G:
                need = max(0.0, target_R[g]-(self.I_R[g][1]+self.I_R[g][2]))
                taken, total_taken, waste_arrival = self._ship_from_W_to_R(g,need)
                x_ship[(g,1)]=taken[(g,1)]; x_ship[(g,2)]=taken[(g,2)]
                ship_qty_g[g]=total_taken
                waste_on_arrival_total+=waste_arrival
                self.I_R[g][1]+=taken[(g,1)]
                self.I_R[g][2]+=taken[(g,2)]

            # 3) 小売販売
            z_disc = {g: 1 if (pol.discount.use_discount and self.I_R[g][2]>0) else 0 for g in ACTIVE_G}
            x_sell_full={(g,s):0.0 for g in ACTIVE_G for s in [1,2]}
            x_sell_disc={(g,2):0.0 for g in ACTIVE_G}

            # ★ この日の売り逃し個数を計測
            lost_R_today = 0.0

            for g in ACTIVE_G:
                # --- 定価販売 ---
                demand_full = D_FULL[(g,t)]
                if pol.ship_order.order == 'FIFO':
                    sell2_full=min(self.I_R[g][2], demand_full)
                    x_sell_full[(g,2)]=sell2_full
                    self.I_R[g][2]-=sell2_full
                    demand_full-=sell2_full
                    sell1=min(self.I_R[g][1], max(0.0,demand_full))
                    x_sell_full[(g,1)]=sell1
                    self.I_R[g][1]-=sell1
                else:
                    sell1=min(self.I_R[g][1], demand_full)
                    x_sell_full[(g,1)]=sell1
                    self.I_R[g][1]-=sell1
                    demand_full-=sell1
                    sell2_full=min(self.I_R[g][2], max(0.0,demand_full))
                    x_sell_full[(g,2)]=sell2_full
                    self.I_R[g][2]-=sell2_full

                # 定価需要の売り逃し（在庫が足りずに売れなかった分）
                lost_full = max(0.0, demand_full)

                # --- 値引き販売 ---
                lost_disc = 0.0
                if z_disc[g]==1:
                    demand_disc=D_DISC[(g,t)]
                    sell2_disc=min(self.I_R[g][2], demand_disc)
                    x_sell_disc[(g,2)]=sell2_disc
                    self.I_R[g][2]-=sell2_disc
                    # 値引き需要の売り逃し
                    lost_disc = max(0.0, demand_disc - sell2_disc)

                # このカテゴリ g の売り逃しを合算
                lost_R_today += (lost_full + lost_disc)

            # 4) 劣化・廃棄
            theta_W=pol.storage_W.theta()
            theta_R=pol.storage_R.theta()
            I_W_next={g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}
            I_R_next={g:{1:0.0,2:0.0,3:0.0} for g in ACTIVE_G}
            waste_today=0.0
            waste_W_today=0.0
            waste_R_today=0.0

            for g in ACTIVE_G:
                I_W_next[g]=inv_next('W',g,self.I_W[g],{1:0.0,2:0.0,3:0.0},theta_W,pol.mekiki.quality_boost())
                waste_today+=I_W_next[g][3]; waste_W_today+=I_W_next[g][3]; I_W_next[g][3]=0.0
                I_R_next[g]=inv_next('R',g,self.I_R[g],{1:0.0,2:0.0,3:0.0},theta_R,1.0)
                waste_today+=I_R_next[g][3]; waste_R_today+=I_R_next[g][3]; I_R_next[g][3]=0.0

            self.I_W=I_W_next
            self.I_R=I_R_next

            # 5) 収支・CO2計算
            rev_W  = sum(P_W2R[g]*ship_qty_g[g] for g in ACTIVE_G)
            buy_W  = sum(C_BUY[g]*x_buy[g]      for g in ACTIVE_G)
            mekiki_cost = sum(self.policy.mekiki.cost_per_unit()*x_buy[g] for g in ACTIVE_G)

            stor_W = theta_W * sum(C_STOR[('W',g,s)]*self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)
            ship_c = sum(C_SHIP[g]*ship_qty_g[g]for g in ACTIVE_G)
            profit_W_day = rev_W - buy_W - mekiki_cost - ship_c - stor_W

            rev_R_full = sum(P_LIST[g]*x_sell_full[(g,1)] for g in ACTIVE_G) \
                       + sum(P_LIST[g]*x_sell_full[(g,2)] for g in ACTIVE_G)
            rev_R_disc = sum(DELTA[g]*P_LIST[g]*x_sell_disc[(g,2)] for g in ACTIVE_G)
            cost_pur   = sum(P_W2R[g]*ship_qty_g[g] for g in ACTIVE_G)

            stor_R     = theta_R * sum(C_STOR[('R',g,s)]*self.I_R[g][s] for g in ACTIVE_G for s in S_LIST)
            disc_fix   = sum(C_DISC[g]*z_disc[g] for g in ACTIVE_G)
            profit_R_day = (rev_R_full+rev_R_disc) - cost_pur - stor_R - disc_fix

            co2_ship = sum(E_SHIP[g]*ship_qty_g[g] for g in ACTIVE_G)
            co2_W_day = ((1-theta_W)*E_STOR0['W']+theta_W*E_STOR1['W']) * \
                        sum(self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)
            co2_R_day = ((1-theta_R)*E_STOR0['R']+theta_R*E_STOR1['R']) * \
                        sum(self.I_R[g][s] for g in ACTIVE_G for s in S_LIST)
            co2_day = co2_ship + co2_W_day + co2_R_day

            self.total_profit_W += profit_W_day
            self.total_profit_R += profit_R_day
            self.total_waste    += (waste_today + waste_on_arrival_total)
            self.total_co2      += co2_day

            self.co2_W   += co2_W_day
            self.co2_R   += co2_R_day
            self.co2_ship+= co2_ship
            self.waste_W += waste_W_today
            self.waste_R += waste_R_today
            self.waste_ship += waste_on_arrival_total

            # ★ 売り逃しを累積
            self.lost_R += lost_R_today

        return SimStats(
            profit_W=self.total_profit_W,
            profit_R=self.total_profit_R,
            co2_total=self.total_co2,
            waste_total=self.total_waste,
            co2_W=self.co2_W, co2_R=self.co2_R, co2_ship=self.co2_ship,
            waste_W=self.waste_W, waste_R=self.waste_R, waste_ship=self.waste_ship,
            lost_R=self.lost_R
        )

# ====== 候補集合生成 ======
def get_safety_candidates_R()->Dict[str, List[int]]:
    inv_L = [10,20,30,40,50,60,70,80,90,100,110]
    inv_S = [5,10,15,20,25,30,35,40,45,50,55,60]
    if MODE=='L':   return {'L':inv_L,'S':[]}
    elif MODE=='S': return {'L':[],'S':inv_S}
    else:           return {'L':inv_L,'S':inv_S}

def get_safety_candidates_W()->Dict[str, List[int]]:
    inv_L = [30,60,90,120,150,180,210]
    inv_S = [30,55,80,105,130,155,180]
    if MODE=='L':   return {'L':inv_L,'S':[]}
    elif MODE=='S': return {'L':[],'S':inv_S}
    else:           return {'L':inv_L,'S':inv_S}

# ====== ポリシー生成ユーティリティ ======
def random_policy()->DiscretePolicy:
    safR = get_safety_candidates_R()
    safW = get_safety_candidates_W()
    pol = DiscretePolicy(
        safety_target_W_L=(random.choice(safW['L']) if safW['L'] else None),
        safety_target_W_S=(random.choice(safW['S']) if safW['S'] else None),
        mekiki=MekikiLevel(level=random.choice(['high','low'])),
        storage_W=StorageMode(mode=random.choice(['high','low'])),
        ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),
        safety_target_R_L=(random.choice(safR['L']) if safR['L'] else None),
        safety_target_R_S=(random.choice(safR['S']) if safR['S'] else None),
        discount=DiscountPolicy(use_discount=random.choice([True,False])),
        storage_R=StorageMode(mode=random.choice(['high','low'])),
    )
    return pol

def clone_with_W(base:DiscretePolicy, **kw)->DiscretePolicy:
    d = base.__dict__.copy()
    for k in ['safety_target_W_L','safety_target_W_S','mekiki','storage_W','ship_order']:
        if k in kw: d[k]=kw[k]
    return DiscretePolicy(**d)

def clone_with_R(base:DiscretePolicy, **kw)->DiscretePolicy:
    d = base.__dict__.copy()
    for k in ['safety_target_R_L','safety_target_R_S','discount','storage_R','ship_order']:
        if k in kw: d[k]=kw[k]
    return DiscretePolicy(**d)

# ====== メトリクス評価 ======
def evaluate_metrics(pol: DiscretePolicy):
    sim = Simulator(pol).run()
    return {
        "profit_W": sim.profit_W,
        "profit_R": sim.profit_R,
        "profit_total": sim.profit_W + sim.profit_R,
        "co2_total": sim.co2_total,
        "waste_total": sim.waste_total,
        "co2_W": sim.co2_W, "co2_R": sim.co2_R, "co2_ship": sim.co2_ship,
        "waste_W": sim.waste_W, "waste_R": sim.waste_R, "waste_ship": sim.waste_ship,
        "lost_R": sim.lost_R,
        "policy": pol
    }

# ====== 利得（利益 − ペナルティ） ======
def payoff_W(res: dict) -> float:
    eff_co2_W   = res["co2_W"] + ALPHA_SHIP_CO2 * res["co2_ship"]
    eff_waste_W = res["waste_W"] + BETA_SHIP_WASTE * res["waste_ship"]
    # 卸側は今回は売り逃しペナルティは見ない（PENALTY_LOST_W=0.0）
    return res["profit_W"] - PENALTY_CO2_W*eff_co2_W - PENALTY_WASTE_W*eff_waste_W

def payoff_R(res: dict) -> float:
    # 小売の利得 = 利益 − CO2ペナルティ − 廃棄ペナルティ − 売り逃しペナルティ
    return (res["profit_R"]
            - PENALTY_CO2_R*res["co2_R"]
            - PENALTY_WASTE_R*res["waste_R"]
            - PENALTY_LOST_R*res["lost_R"])

# ====== ベストレスポンス探索 ======
def best_response_W(base:DiscretePolicy, samples:int=40)->Tuple[DiscretePolicy, dict, float]:
    safW = get_safety_candidates_W()
    choices = []
    for _ in range(samples):
        cand = clone_with_W(
            base,
            safety_target_W_L=(random.choice(safW['L']) if safW['L'] else None),
            safety_target_W_S=(random.choice(safW['S']) if safW['S'] else None),
            mekiki=MekikiLevel(level=random.choice(['high','low'])),
            storage_W=StorageMode(mode=random.choice(['high','low'])),
            ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),
        )
        choices.append(cand)

    best_pol = base
    best_res = evaluate_metrics(base)
    best_val = payoff_W(best_res)

    for pol in choices:
        res = evaluate_metrics(pol)
        val = payoff_W(res)
        if val > best_val:
            best_val = val
            best_pol = pol
            best_res = res
    return best_pol, best_res, best_val

def best_response_R(base:DiscretePolicy, samples:int=40)->Tuple[DiscretePolicy, dict, float]:
    safR = get_safety_candidates_R()
    choices = []
    for _ in range(samples):
        cand = clone_with_R(
            base,
            safety_target_R_L=(random.choice(safR['L']) if safR['L'] else None),
            safety_target_R_S=(random.choice(safR['S']) if safR['S'] else None),
            discount=DiscountPolicy(use_discount=random.choice([True,False])),
            storage_R=StorageMode(mode=random.choice(['high','low'])),
            ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),
        )
        choices.append(cand)

    best_pol = base
    best_res = evaluate_metrics(base)
    best_val = payoff_R(best_res)

    for pol in choices:
        res = evaluate_metrics(pol)
        val = payoff_R(res)
        if val > best_val:
            best_val = val
            best_pol = pol
            best_res = res
    return best_pol, best_res, best_val

# ====== 交互ベストレスポンス ======
def alternating_best_response(iters:int=6, init:Optional[DiscretePolicy]=None,
                              samples_W:int=40, samples_R:int=40):
    pol = init or random_policy()
    history = []
    for k in range(iters):
        pol_R, res_R, _ = best_response_R(pol, samples=samples_R)
        pol = pol_R
        pol_W, res_W, _ = best_response_W(pol, samples=samples_W)
        pol = pol_W
        history.append(evaluate_metrics(pol))
    return pol, history

# ====== 結果表示 ======
def show_result(title:str, res:dict):
    pol:DiscretePolicy = res["policy"]
    print("="*60)
    print(f"[{title}] MODE={MODE}")
    print("-"*60)
    print(f" 卸利益: {res['profit_W']:.2f} 円 / 小売利益: {res['profit_R']:.2f} 円 / 合計: {res['profit_total']:.2f} 円")
    print(f" CO2: total {res['co2_total']:.3f} kg (W {res['co2_W']:.3f}, R {res['co2_R']:.3f}, ship {res['co2_ship']:.3f})")
    print(f" 廃棄: total {res['waste_total']:.2f} 個 (W {res['waste_W']:.2f}, R {res['waste_R']:.2f}, ship {res['waste_ship']:.2f})")
    print(f" 売り逃し: 小売 lost_R = {res['lost_R']:.2f} 個")
    print("-"*60)
    if 'L' in ACTIVE_G:
        print(f" 卸安全L: {pol.safety_target_W_L}, 小売安全L: {pol.safety_target_R_L}")
    if 'S' in ACTIVE_G:
        print(f" 卸安全S: {pol.safety_target_W_S}, 小売安全S: {pol.safety_target_R_S}")
    print(f" 目利き: {pol.mekiki.level}, 卸保存: {pol.storage_W.mode}, 小売保存: {pol.storage_R.mode}")
    print(f" 出荷/販売順序: {pol.ship_order.order}, 値引き: {'ON' if pol.discount.use_discount else 'OFF'}")

# ====== 戦略ラベル ======
def label_W(pol: DiscretePolicy) -> str:
    parts = []
    if 'L' in ACTIVE_G and pol.safety_target_W_L is not None:
        parts.append(f"L{pol.safety_target_W_L}")
    if 'S' in ACTIVE_G and pol.safety_target_W_S is not None:
        parts.append(f"S{pol.safety_target_W_S}")
    parts.append('M' + ('H' if pol.mekiki.level=='high' else 'L'))
    parts.append('St' + ('H' if pol.storage_W.mode=='high' else 'L'))
    parts.append('O' + ('F' if pol.ship_order.order=='FIFO' else 'L'))
    return "卸:" + "".join(parts)

def label_R(pol: DiscretePolicy) -> str:
    parts = []
    if 'L' in ACTIVE_G and pol.safety_target_R_L is not None:
        parts.append(f"L{pol.safety_target_R_L}")
    if 'S' in ACTIVE_G and pol.safety_target_R_S is not None:
        parts.append(f"S{pol.safety_target_R_S}")
    parts.append('D' + ('1' if pol.discount.use_discount else '0'))
    parts.append('St' + ('H' if pol.storage_R.mode=='high' else 'L'))
    parts.append('O' + ('F' if pol.ship_order.order=='FIFO' else 'L'))
    return "小売:" + "".join(parts)

# ====== 卸戦略・小売戦略のサンプル集合 ======
def sample_W_strategies(base: DiscretePolicy, n:int) -> List[DiscretePolicy]:
    safW = get_safety_candidates_W()
    lst = []
    for i in range(n):
        lst.append(clone_with_W(
            base,
            safety_target_W_L=(random.choice(safW['L']) if safW['L'] else None),
            safety_target_W_S=(random.choice(safW['S']) if safW['S'] else None),
            mekiki=MekikiLevel(level=random.choice(['high','low'])),
            storage_W=StorageMode(mode=random.choice(['high','low'])),
            ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),
        ))
    return lst

def sample_R_strategies(base: DiscretePolicy, n:int) -> List[DiscretePolicy]:
    safR = get_safety_candidates_R()
    lst = []
    for i in range(n):
        lst.append(clone_with_R(
            base,
            safety_target_R_L=(random.choice(safR['L']) if safR['L'] else None),
            safety_target_R_S=(random.choice(safR['S']) if safR['S'] else None),
            discount=DiscountPolicy(use_discount=random.choice([True,False])),
            storage_R=StorageMode(mode=random.choice(['high','low'])),
            ship_order=ShipOrder(order=random.choice(['FIFO','LIFO'])),
        ))
    return lst

def merge_WR(w_pol: DiscretePolicy, r_pol: DiscretePolicy) -> DiscretePolicy:
    """ 卸部分は w_pol、小売部分は r_pol から取って一つの政策にまとめる """
    return DiscretePolicy(
        safety_target_W_L=w_pol.safety_target_W_L,
        safety_target_W_S=w_pol.safety_target_W_S,
        mekiki=w_pol.mekiki,
        storage_W=w_pol.storage_W,
        ship_order=w_pol.ship_order,
        safety_target_R_L=r_pol.safety_target_R_L,
        safety_target_R_S=r_pol.safety_target_R_S,
        discount=r_pol.discount,
        storage_R=r_pol.storage_R,
    )

# ====== 戦略×戦略の利益ヒートマップ ======
def plot_strategy_profit_heatmaps(base_policy: DiscretePolicy,
                                  nW:int=8, nR:int=8,
                                  title_suffix:str="ペナルティ付き利益ゲーム"):
    """
    縦軸＝卸の戦略、横軸＝小売の戦略として
    卸利益・小売利益を色で表した2つのヒートマップを描画する。
    """
    W_list = sample_W_strategies(base_policy, nW)
    R_list = sample_R_strategies(base_policy, nR)

    profit_W_mat = np.zeros((nW, nR))
    profit_R_mat = np.zeros((nW, nR))

    for i, w_pol in enumerate(W_list):
        for j, r_pol in enumerate(R_list):
            pol_ij = merge_WR(w_pol, r_pol)
            res = evaluate_metrics(pol_ij)
            profit_W_mat[i, j] = res["profit_W"]
            profit_R_mat[i, j] = res["profit_R"]

    y_labels = [label_W(p) for p in W_list]
    x_labels = [label_R(p) for p in R_list]

    # 卸利益ヒートマップ
    plt.figure(figsize=(10, 8))
    plt.imshow(profit_W_mat, origin='lower', aspect='auto')
    plt.colorbar(label="卸の利益 [円]")
    plt.xticks(range(nR), x_labels, rotation=90)
    plt.yticks(range(nW), y_labels)
    plt.xlabel("小売の戦略")
    plt.ylabel("卸の戦略")
    plt.title(f"卸の利益ヒートマップ ({title_suffix})")
    plt.tight_layout()

    # 小売利益ヒートマップ
    plt.figure(figsize=(10, 8))
    plt.imshow(profit_R_mat, origin='lower', aspect='auto')
    plt.colorbar(label="小売の利益 [円]")
    plt.xticks(range(nR), x_labels, rotation=90)
    plt.yticks(range(nW), y_labels)
    plt.xlabel("小売の戦略")
    plt.ylabel("卸の戦略")
    plt.title(f"小売の利益ヒートマップ ({title_suffix})")
    plt.tight_layout()

    plt.show()

# ====== メイン実行 ======
if __name__ == "__main__":
    # 0) まず需要の形を確認
    plot_demands(T)

    # 1) ペナルティ付き利益ゲームの準NEを交互BRで探索
    pol0 = random_policy()
    eq_pol, history = alternating_best_response(iters=6, init=pol0,
                                                samples_W=30, samples_R=30)
    res_eq = evaluate_metrics(eq_pol)
    show_result("ペナルティ付き利益ゲーム（準NE）", res_eq)

    # 2) 準NE近傍の卸戦略×小売戦略における利益ヒートマップを表示
    plot_strategy_profit_heatmaps(eq_pol, nW=8, nR=8,
                                  title_suffix="準NE近傍の戦略プロファイル")
