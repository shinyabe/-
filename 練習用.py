# 需要モデルは梶木さんベース
from __future__ import annotations  # 型ヒントでクラス自身を参照可能にする
from dataclasses import dataclass  # データクラス定義用
from typing import Dict, Tuple, List, Optional  # 型ヒント用
import math, random               # 数学関数・乱数生成ライブラリ
import numpy as np                # 数値計算用の NumPy
import matplotlib.pyplot as plt  # グラフ描画用の matplotlib.pyplot
import matplotlib as mpl      # グラフ描画用の matplotlib
import pandas as pd                 # データフレーム操作・Excel 出力用の pandas

# ====== 日本語フォント設定 ======
mpl.rcParams["font.family"] = "Hiragino Sans" # macOS の場合の日本語フォント設定
mpl.rcParams["axes.unicode_minus"] = False # マイナス記号が文字化けするのを防止

# ====== 共通設定 ======
random.seed(42)  # 乱数シードを固定してシミュレーション結果の再現性を確保

# モード: 'L' 根菜のみ, 'S' 葉物のみ, 'both' 両方
MODE = "S"  # 実験対象カテゴリのモード設定

T = 30                      # シミュレーション日数
G_LIST = ["L", "S"]         # 商品カテゴリ集合: L=根菜, S=葉物
S_LIST = [1, 2, 3]          # 鮮度状態集合: 1=良,2=やや劣化,3=廃棄レベル
N_LIST = ["W", "R"]         # 節点集合: W=卸売, R=小売

# 価格・費用
P_LIST = {"L": 180.0, "S": 180.0}     # 小売定価
P_W2R  = {"L": 120.0, "S": 120.0}     # 卸→小売の仕入単価
C_BUY  = {"L": 90.0,  "S": 90.0}      # 卸の仕入単価
DELTA  = {"L": 0.70,  "S": 0.70}      # 値引き率（値引き価格 = DELTA * 定価）
C_SHIP = {"L": 7.0,   "S": 7.0}       # 卸→小売の輸送コスト（円/個）
C_DISC = {"L": 300.0, "S": 300.0}     # 値引き販売を行う際の固定費（円/日）
"""
C_DISC = {"L": 150.0, "S": 150.0}     # 値引き販売を行う際の固定費（円/日）(現実より？)
"""

# 保管費（円/個/日）※高品質保存モードのときに有効な係数
#必要に応じてどちらか選択
C_STOR = {
    ("W", "L", 1): 1.0,  ("W", "L", 2): 1.0,  ("W", "L", 3): 1.0,   # 卸・根菜・各鮮度の保管コスト
    ("W", "S", 1): 1.30, ("W", "S", 2): 1.30, ("W", "S", 3): 1.30, # 卸・葉物・各鮮度の保管コスト
    ("R", "L", 1): 1.7,  ("R", "L", 2): 1.7,  ("R", "L", 3): 1.7,  # 小売・根菜・各鮮度の保管コスト
    ("R", "S", 1): 3.2,  ("R", "S", 2): 3.2,  ("R", "S", 3): 3.2,  # 小売・葉物・各鮮度の保管コスト
}

# 例：全体を約1/2にしても現場感は保てる
"""
C_STOR = {
    ("W", "L", 1): 0.5,  ("W", "L", 2): 0.5,  ("W", "L", 3): 0.5, # 卸・根菜・各鮮度の保管コスト
    ("W", "S", 1): 0.8,  ("W", "S", 2): 0.8,  ("W", "S", 3): 0.8, # 卸・葉物・各鮮度の保管コスト
    ("R", "L", 1): 1.0,  ("R", "L", 2): 1.0,  ("R", "L", 3): 1.0,  # 小売・根菜・各鮮度の保管コスト
    ("R", "S", 1): 1.6,  ("R", "S", 2): 1.6,  ("R", "S", 3): 1.6,  # 小売・葉物・各鮮度の保管コスト
}
"""

# CO2係数
E_STOR0 = {"W": 0.0020, "R": 0.0036}  # 通常保存モード時の保管CO2係数（kg/個/日）
E_STOR1 = {"W": 0.0040, "R": 0.0066}  # 高品質保存モード時の保管CO2係数（kg/個/日）
E_SHIP  = {"L": 0.010,  "S": 0.010}   # 1個あたり輸送由来CO2排出量（kg）
"""
E_SHIP = {"L": 0.001, "S": 0.001}  # 1個あたり輸送由来CO2排出量（kg）(1g/個くらいに修正)
"""
E_TRIP  = 2.5                         # 1回の配送あたりの固定CO2排出量（未使用）
CAP_TRUCK = 200                       # トラックの容量（個）（未使用）

# ====== ペナルティ係数（利得 = 利益 − ペナルティ）======
PENALTY_CO2_W   = 100.0  # 卸側のCO2排出に対するペナルティ係数(円/kg)
PENALTY_CO2_R   = 100.0  # 小売側のCO2排出に対するペナルティ係数(円/kg)
"""
PENALTY_CO2_W = 0.3  # 卸側のCO2排出に対するペナルティ係数(円/kg) (炭素税水準で修正)
PENALTY_CO2_R = 0.3  # 小売側のCO2排出に対するペナルティ係数(円/kg) (炭素税水準で修正)
"""
"""
PENALTY_CO2_W = 2.0  # 卸側のCO2排出に対するペナルティ係数(円/kg) (将来の高炭素税水準を意識して修正)
PENALTY_CO2_R = 2.0  # 小売側のCO2排出に対するペナルティ係数(円/kg) (将来の高炭素税水準を意識して修正)
"""

PENALTY_WASTE_W = 20.0   # 卸の廃棄量に対するペナルティ係数(円/個)
PENALTY_WASTE_R = 20.0   # 小売の廃棄量（＋輸送廃棄）に対するペナルティ係数(円/個)
"""
(消費者に近いほど廃棄の社会的インパクトが大きいと考えられるため、小売の方が高く設定)
PENALTY_WASTE_W = 10.0   # 卸の廃棄量に対するペナルティ係数(円/個)
PENALTY_WASTE_R = 20.0   # 小売の廃棄量（＋輸送廃棄）に対するペナルティ係数(円/個)
"""
PENALTY_LOST_W  = 0.0    # 卸の売り逃しペナルティ（今回は0）
PENALTY_LOST_R  = 120.0  # 小売の売り逃しペナルティ（客一人分の利益イメージ）
ALPHA_SHIP_CO2  = 0.5    # 輸送CO2の重みパラメータ(未使用）(今回輸送由来のCO2は全て小売負担とする)
BETA_SHIP_WASTE = 0.5    # 輸送廃棄の重みパラメータ（未使用）

# === 需要（客入り）===
MU_C    = {"L": 250.0, "S": 250.0}   # μ^C : 一日あたりの小売店の平均客数
SIGMA_C = {"L": 70.0,  "S": 70.0}    # σ^C : 一日あたりの客数の標準偏差
C_PER_CUSTOMER = 1.0  # 一人あたり需要量の基準値（c_i を個別に決めるので未使用(一人当たりの購入数を固定する際に使用)）

I_CUSTOMERS: Dict[Tuple[str, int], int] = {}  # 来店客数 I_g,t を格納
D_DEMAND: Dict[Tuple[str, int], float] = {}  # 需要量 D_g,t を格納

def build_customer_tables(T: int = 30):
    #来店客数 I_g,t と 需要量 D_g,t を T 日分生成して保存する関数
    #カテゴリ g, 日 t ごとに I_g,t ~ N(μ^C_g, (σ^C_g)^2) を生成
    #各顧客 i の購入数 c_i を 0〜4 の整数から一様ランダムに決定
    #需要 D_g,t = Σ_i c_i を計算して保存する
    global I_CUSTOMERS, D_DEMAND       # グローバル変数を関数内で更新する宣言
    I_CUSTOMERS = {}                   # 来店客数テーブルを空で初期化
    D_DEMAND = {}                      # 需要テーブルも空で初期化
    for t in range(1, T + 1):          # 日 t = 1〜T についてループ
        for g in G_LIST:               # 各カテゴリ g（根菜L,葉物S）についてループ
            I = random.gauss(MU_C[g], SIGMA_C[g]) # 来店客数 I を正規分布 N(μ^C,σ^C) から1回サンプリング
            I = max(0.0, I)            # 負の値が出た場合は0に切り上げ
            I_int = int(round(I))      # 客数を四捨五入して整数値にする
            I_CUSTOMERS[(g, t)] = I_int  # (g,t) の来店客数として保存

            # 各顧客 i が購入する個数 c_i を 0〜4 の乱数で決めて合算する
            D = 0.0                    # その日の総需要量 D の初期化
            for _ in range(I_int):     # i = 1..I_int のループ
                ci = random.randint(0, 4)  # 0〜4個の中から一様ランダムに購入個数を決定
                D += ci                    # D に c_i を加算
            D_DEMAND[(g, t)] = D       # (g,t)の需要量 D_g,t として保存

# テーブル生成
build_customer_tables(T)  # T日分の来店客数・需要量テーブルを生成
#生成された I_g,t（客入り）と D_g,t（需要量）を折れ線グラフで可視化する関数
def plot_customers_and_demands(T: int = 30):
    days = range(1, T + 1)
    name_map = {"L": "根菜L", "S": "葉物S"} # ACTIVE_G に基づいて描画対象カテゴリを決定
    for g in ACTIVE_G:              # 各カテゴリごとにグラフを描く
        name = name_map[g]           # カテゴリ名を取得
        I_list = [I_CUSTOMERS[(g, t)] for t in days]  # 日ごとの来店客数リスト
        D_list = [D_DEMAND[(g, t)] for t in days]     # 日ごとの需要量リスト
        plt.figure(figsize=(8, 4))  # 新しい図を作成（横8,縦4インチ）
        plt.plot(days, I_list, label=f"{name}: 来店客数 I")      # 客数の折れ線グラフ
        plt.plot(days, D_list, label=f"{name}: 需要量 D (= I×c)")  # 需要量の折れ線グラフ
        plt.xlabel("日数 t")                        # x軸ラベル
        plt.ylabel("人数 / 個数")                  # y軸ラベル
        plt.title(f"{name}の客入り・需要（梶木モデル：正規分布＋D=Σc_i）")  # グラフタイトル
        plt.legend()                                # 凡例を表示
        plt.tight_layout()                          # レイアウトを自動調整

    plt.show()  # 全ての図を画面に表示

# 初期在庫
I_INIT = {(n, g, s): 0.0 for n in N_LIST for g in G_LIST for s in S_LIST}  # 全ての節点・カテゴリ・鮮度の初期在庫を0で初期化
I_INIT[("R", "S", 1)] = 80.0  # 小売・葉物・鮮度1の初期在庫を80個に設定
I_INIT[("R", "L", 1)] = 80.0  # 小売・根菜・鮮度1の初期在庫を80個に設定
I_INIT[("W", "S", 1)] = 80.0   # 卸・葉物・鮮度1の初期在庫を80個に設定
I_INIT[("W", "L", 1)] = 80.0   # 卸・根菜・鮮度1の初期在庫を80個に設定

# 鮮度遷移確率テーブル（P0:通常保存, P1:高品質保存）
P0_base: Dict[Tuple[str, str, int, int], float] = {}  # 通常保存モード時の遷移確率を格納
P1_base: Dict[Tuple[str, str, int, int], float] = {}  # 高品質保存モード時の遷移確率を格納

def set_transitions_for_node(n: str): #節点 n（W=卸, R=小売）ごとに、カテゴリ・鮮度状態間の遷移確率を設定する関数
    #P0_base: 通常保存、P1_base: 高品質保存
    if n == "W":  # 卸の遷移確率
        # 根菜L, 鮮度1→1,2,3 の確率（通常保存）
        P0_base[(n, "L", 1, 1)] = 0.87; P0_base[(n, "L", 1, 2)] = 0.13; P0_base[(n, "L", 1, 3)] = 0.00
        # 根菜L, 鮮度2→1,2,3 の確率（通常保存）
        P0_base[(n, "L", 2, 1)] = 0.00; P0_base[(n, "L", 2, 2)] = 0.66; P0_base[(n, "L", 2, 3)] = 0.34

        # 根菜L, 鮮度1→1,2,3 の確率（高品質保存）
        P1_base[(n, "L", 1, 1)] = 0.94; P1_base[(n, "L", 1, 2)] = 0.06; P1_base[(n, "L", 1, 3)] = 0.00
        # 根菜L, 鮮度2→1,2,3 の確率（高品質保存）
        P1_base[(n, "L", 2, 1)] = 0.00; P1_base[(n, "L", 2, 2)] = 0.83; P1_base[(n, "L", 2, 3)] = 0.17

        # 葉物S, 鮮度1→1,2,3 の確率（通常保存）
        P0_base[(n, "S", 1, 1)] = 0.73; P0_base[(n, "S", 1, 2)] = 0.27; P0_base[(n, "S", 1, 3)] = 0.00
        # 葉物S, 鮮度2→1,2,3 の確率（通常保存）
        P0_base[(n, "S", 2, 1)] = 0.00; P0_base[(n, "S", 2, 2)] = 0.47; P0_base[(n, "S", 2, 3)] = 0.53

        # 葉物S, 鮮度1→1,2,3 の確率（高品質保存）
        P1_base[(n, "S", 1, 1)] = 0.84; P1_base[(n, "S", 1, 2)] = 0.16; P1_base[(n, "S", 1, 3)] = 0.00
        # 葉物S, 鮮度2→1,2,3 の確率（高品質保存）
        P1_base[(n, "S", 2, 1)] = 0.00; P1_base[(n, "S", 2, 2)] = 0.63; P1_base[(n, "S", 2, 3)] = 0.37
    else:  # 小売の遷移確率
        # 根菜L, 鮮度1→1,2,3（通常保存）
        P0_base[(n, "L", 1, 1)] = 0.70; P0_base[(n, "L", 1, 2)] = 0.30; P0_base[(n, "L", 1, 3)] = 0.00
        # 根菜L, 鮮度2→1,2,3（通常保存）
        P0_base[(n, "L", 2, 1)] = 0.00; P0_base[(n, "L", 2, 2)] = 0.60; P0_base[(n, "L", 2, 3)] = 0.40

        # 根菜L, 鮮度1→1,2,3（高品質保存）
        P1_base[(n, "L", 1, 1)] = 0.95; P1_base[(n, "L", 1, 2)] = 0.05; P1_base[(n, "L", 1, 3)] = 0.00
        # 根菜L, 鮮度2→1,2,3（高品質保存）
        P1_base[(n, "L", 2, 1)] = 0.00; P1_base[(n, "L", 2, 2)] = 0.90; P1_base[(n, "L", 2, 3)] = 0.10

        # 葉物S, 鮮度1→1,2,3（通常保存）
        P0_base[(n, "S", 1, 1)] = 0.40; P0_base[(n, "S", 1, 2)] = 0.60; P0_base[(n, "S", 1, 3)] = 0.00
        # 葉物S, 鮮度2→1,2,3（通常保存）
        P0_base[(n, "S", 2, 1)] = 0.00; P0_base[(n, "S", 2, 2)] = 0.20; P0_base[(n, "S", 2, 3)] = 0.80

        # 葉物S, 鮮度1→1,2,3（高品質保存）
        P1_base[(n, "S", 1, 1)] = 0.85; P1_base[(n, "S", 1, 2)] = 0.15; P1_base[(n, "S", 1, 3)] = 0.00
        # 葉物S, 鮮度2→1,2,3（高品質保存）
        P1_base[(n, "S", 2, 1)] = 0.00; P1_base[(n, "S", 2, 2)] = 0.70; P1_base[(n, "S", 2, 3)] = 0.30

for n in N_LIST:           # 卸・小売それぞれについて
    set_transitions_for_node(n) # 遷移確率を設定する関数を呼び出し

# s=3 は吸収状態（廃棄レベル）なので、遷移確率を固定
for n in N_LIST:                    # 両節点(W,R)について
    for g in G_LIST:                # 両カテゴリ(S,L)について
        P0_base[(n, g, 3, 1)] = 0.0; P1_base[(n, g, 3, 1)] = 0.0  # 鮮度3から1への遷移確率0
        P0_base[(n, g, 3, 2)] = 0.0; P1_base[(n, g, 3, 2)] = 0.0  # 鮮度3から2への遷移確率0
        P0_base[(n, g, 3, 3)] = 1.0; P1_base[(n, g, 3, 3)] = 1.0  # 鮮度3は常に3にとどまる（吸収状態）

# 対象カテゴリ（MODE に応じてシミュレーション対象を絞る）
if MODE == "L":
    ACTIVE_G = ["L"]          # 根菜のみ
elif MODE == "S":
    ACTIVE_G = ["S"]          # 葉物のみ
else:
    ACTIVE_G = ["L", "S"]     # 両方

# ====== 政策（戦略）定義 ======
@dataclass(frozen=True) # 目利きレベル、高品質保存モード、出荷順序、値引き政策を定義
class MekikiLevel: #目利きレベル
    level: str  # 'high' or 'low'（目利きレベル）
    def quality_boost(self) -> float:
        return 0.8 if self.level == "high" else 1.0  # 高目利きだと劣化確率を軽減(20%劣化しにくくなる)
    """
    現実の目利きによる劣化速度改善は(10~30%)でその範囲で調整可能
    """
    def cost_per_unit(self) -> float:
        return 5.0 if self.level == "high" else 0.0  # 高目利きの際の1個あたり追加コスト

@dataclass(frozen=True)
class StorageMode:
    mode: str  # 'high' or 'low'（保存モード：高品質 or 通常）
    def theta(self) -> int:
        return 1 if self.mode == "high" else 0  # 高品質保存なら1, 通常なら0（遷移・CO2計算で使う）

@dataclass(frozen=True)
class ShipOrder:
    order: str  # 'FIFO' or 'LIFO'（出荷順序：先入先出 or 先入後出）

@dataclass(frozen=True)
class DiscountPolicy:
    use_discount: bool  # 値引き販売を行うかどうかのフラグ

@dataclass
class DiscretePolicy: # 卸・小売双方の戦略パラメータをまとめたクラス
    # 卸側の戦略パラメータ
    safety_target_W_L: Optional[int]  # 根菜の卸安全在庫水準
    safety_target_W_S: Optional[int]  # 葉物の卸安全在庫水準
    mekiki: MekikiLevel              # 目利きレベル
    storage_W: StorageMode           # 卸の保存モード
    ship_order_W: ShipOrder            # 卸の出荷順序
    # 小売側の戦略パラメータ
    safety_target_R_L: Optional[int]  # 根菜の小売安全在庫水準
    safety_target_R_S: Optional[int]  # 葉物の小売安全在庫水準
    discount: DiscountPolicy          # 値引き政策（ON/OFF）
    storage_R: StorageMode            # 小売の保存モード
    ship_order_R: ShipOrder           # 小売の出荷順序
    

# ====== シミュレーションの基本関数 ======
def peff(n, g, sf, st, theta, quality_boost):
    #節点 n, カテゴリ g, 現在鮮度 sf, 次期鮮度 st, 保存モード theta, 目利き補正 quality_boost に基づき、有効遷移確率を計算する関数
    base = (1 - theta) * P0_base[(n, g, sf, st)] + theta * P1_base[(n, g, sf, st)]  # 保存モードに応じた基本遷移確率
    if st > sf:  # 劣化方向への遷移のみ目利き補正を適用
        return base * quality_boost  # 目利き補正を適用
    return base   # 劣化しない遷移はそのまま

def inv_next(n, g, inv, arrival, theta, quality_boost):
    #節点n,カテゴリgについて、在庫 inv と追加到着量 arrival を基に、1日後の在庫分布を計算する関数
    out = {1: 0.0, 2: 0.0, 3: 0.0}   # 次期の鮮度1,2,3の在庫を0で初期化
    for sf in S_LIST:                # 現在鮮度 sf ごとに
        total_sf = inv.get(sf, 0.0) + arrival.get(sf, 0.0)  # 現在在庫＋到着分
        if total_sf <= 0:
            continue                 # その鮮度の在庫が無ければスキップ
        raw = {st: peff(n, g, sf, st, theta, quality_boost) for st in S_LIST}  # 各stへの遷移確率
        ssum = sum(raw.values())     # 正規化用の合計
        norm = {st: raw[st] / ssum for st in S_LIST}  # 合計1になるよう正規化
        for st in S_LIST:            # 次期鮮度stへ
            out[st] += norm[st] * total_sf  # total_sfを遷移確率で配分
    return out                       # 次期の在庫分布を返す

@dataclass
class SimStats: # シミュレーション統計情報を格納するクラス
    profit_W: float      # 卸の総利益
    profit_R: float      # 小売の総利益
    co2_total: float     # 全体CO2排出量
    waste_total: float   # 全体廃棄量
    co2_W: float         # 卸のCO2排出
    co2_R: float         # 小売のCO2排出
    co2_ship: float      # 輸送由来CO2排出
    waste_W: float       # 卸の廃棄量
    waste_R: float       # 小売の廃棄量
    waste_ship: float    # 輸送中廃棄量
    lost_R: float        # 小売側の売り逃し数

class Simulator: # シミュレータ本体クラス
    # 輸送中の鮮度劣化確率（カテゴリごと）: (1→2 の確率, 2→3 の確率)
    SHIP_DECAY = {"L": (0.01, 0.00), "S": (0.05, 0.05)} # 輸送中の鮮度劣化確率（カテゴリごと）: (1→2 の確率, 2→3 の確率)

    def __init__(self, policy: DiscretePolicy): # シミュレータの初期化
        self.policy = policy                                     # 使用する戦略を保持
        self.I_W = {g: {1: 0.0, 2: 0.0, 3: 0.0} for g in ACTIVE_G}  # 卸在庫（鮮度別）を0で初期化
        self.I_R = {g: {1: 0.0, 2: 0.0, 3: 0.0} for g in ACTIVE_G}  # 小売在庫（鮮度別）を0で初期化
        for (n, g, s), v in I_INIT.items():                      # 初期在庫設定ループ
            if g not in self.I_W:
                continue                                         # モード外カテゴリならスキップ
            if n == "W":
                self.I_W[g][s] = v                               # 卸の初期在庫をセット
            else:
                self.I_R[g][s] = v                               # 小売の初期在庫をセット

        # 利益・CO2・廃棄などの累積値を0で初期化
        self.total_profit_W = 0.0    # 卸の総利益
        self.total_profit_R = 0.0   # 小売の総利益
        self.total_co2 = 0.0       # 全体CO2排出量
        self.total_waste = 0.0    # 全体廃棄量
        self.co2_W = 0.0        # 卸のCO2排出
        self.co2_R = 0.0       # 小売のCO2排出
        self.co2_ship = 0.0    # 輸送由来CO2排出
        self.waste_W = 0.0      # 卸の廃棄量
        self.waste_R = 0.0     # 小売の廃棄量
        self.waste_ship = 0.0  # 輸送中廃棄量
        self.lost_R = 0.0  # 小売の売り逃し累計

    def _apply_ship_decay(self, g: str, take1: float, take2: float):
        #輸送中の鮮度劣化を計算する
        #take1: 鮮度1で出荷した数量, take2: 鮮度2で出荷した数量
        #戻り値: (到着時の(鮮度1,鮮度2)数量), 到着前に鮮度3に落ちて廃棄となる数量
        p12, p23 = self.SHIP_DECAY[g]   # カテゴリgの輸送劣化確率を取得
        drop12 = take1 * p12            # 1→2 に劣化する数量
        arr1 = take1 - drop12           # 輸送後も鮮度1として到着する数量
        arr2_from1 = drop12             # 1から2に落ちて到着する数量
        drop23 = take2 * p23            # 2→3 に劣化して到着時廃棄となる数量
        arr2_from2 = take2 - drop23     # 2のまま到着する数量
        waste_on_arrival = drop23       # 到着時点で廃棄（鮮度3）となる数量
        return (arr1, arr2_from1 + arr2_from2), waste_on_arrival  # (鮮度1,2の到着数量), 廃棄量

    def _ship_from_W_to_R(self, g, need):
        #卸から小売へカテゴリgを need 個だけ出荷する処理。
        #出荷順序（FIFO/LIFO）に従って鮮度1,2の在庫から払い出す。
        # 出荷順序（卸の払い出しルール）
        if self.policy.ship_order_W.order == "FIFO":  # 先に鮮度2から払い出す（先入先出）
            take2 = min(self.I_W[g][2], need); self.I_W[g][2] -= take2; need -= take2  # 鮮度2優先で出す
            take1 = min(self.I_W[g][1], need); self.I_W[g][1] -= take1; need -= take1  # 足りない分を鮮度1から
        else:  # LIFO：鮮度1優先
            take1 = min(self.I_W[g][1], need); self.I_W[g][1] -= take1; need -= take1  # 鮮度1優先で出す
            take2 = min(self.I_W[g][2], need); self.I_W[g][2] -= take2; need -= take2  # 残りを鮮度2から

        (arr1, arr2), waste_arrival = self._apply_ship_decay(g, take1, take2)  # 輸送劣化を適用
        return {(g, 1): arr1, (g, 2): arr2}, (take1 + take2), waste_arrival     # 到着在庫・出荷量・廃棄量を返す
#################################################ここまで調整済み#################################################

    def run(self) -> SimStats:
        #T日間のシミュレーションを実行し、統計情報を返す関数
        pol = self.policy  # 使う戦略をローカル変数にコピー
        for t in range(1, T + 1):  # 各日 t についてシミュレーション
            # 1) 卸仕入（安全在庫まで補充）
            def w_stock(g): return self.I_W[g][1] + self.I_W[g][2]  # 卸の販売可能在庫（鮮度1+2）の合計
            target_W = {} # 卸の安全在庫水準をカテゴリ別に格納
            for g in ACTIVE_G: # 各カテゴリ g について
                if g == "L": # 根菜の場合
                    target_W[g] = pol.safety_target_W_L  # 根菜の卸安全在庫水準
                else:
                    target_W[g] = pol.safety_target_W_S  # 葉物の卸安全在庫水準
                assert target_W[g] is not None           # None でないことをチェック（デバッグ用）

            x_buy = {"L": 0, "S": 0}  # 卸がその日に仕入れる数量をカテゴリ別に初期化
            for g in ACTIVE_G: # 各カテゴリ g について
                gap = max(0.0, target_W[g] - w_stock(g))  # 安全在庫水準と現有在庫の差分（不足量）
                x_buy[g] = int(math.ceil(gap))             # 不足分を切り上げて整数仕入量とする
                self.I_W[g][1] += x_buy[g]                 # 仕入れた分を鮮度1在庫として加算

            # 2) 卸→小売 出荷（小売の安全在庫まで補充）
            target_R = {} # 小売の安全在庫水準をカテゴリ別に格納
            for g in ACTIVE_G: # 各カテゴリ g について
                if g == "L": # 根菜の場合
                    target_R[g] = pol.safety_target_R_L  # 根菜の小売安全在庫水準
                else:
                    target_R[g] = pol.safety_target_R_S  # 葉物の小売安全在庫水準
                assert target_R[g] is not None           # None でないことをチェック

            waste_on_arrival_total = 0.0                 # 輸送中に鮮度3になり到着時廃棄となる量の合計
            ship_qty_g = {g: 0.0 for g in ACTIVE_G}      # カテゴリ別の出荷総量

            for g in ACTIVE_G: # 各カテゴリ g について
                need = max(0.0, target_R[g] - (self.I_R[g][1] + self.I_R[g][2]))  # 小売の不足量（鮮度1+2のみ）
                taken, total_taken, waste_arrival = self._ship_from_W_to_R(g, need)  # 卸から小売へ出荷
                ship_qty_g[g] = total_taken                       # 出荷総量を記録
                waste_on_arrival_total += waste_arrival           # 輸送中廃棄を累計

                self.I_R[g][1] += taken[(g, 1)]                   # 小売側の鮮度1在庫に到着分を加算
                self.I_R[g][2] += taken[(g, 2)]                   # 小売側の鮮度2在庫に到着分を加算

            # 3) 小売販売（梶木の「客数→需要D」から販売に反映）
            #    値引きあり：状態1(定価)と状態2(値引き)を購入対象とし、選択はランダム
            #    値引きなし：状態1(定価)のみ購入対象（状態2は売れない想定）
            z_disc = {g: 1 if (pol.discount.use_discount and self.I_R[g][2] > 0) else 0 for g in ACTIVE_G}
            # z_disc[g]=1 のとき、そのカテゴリgで値引き販売モードを実施

            x_sell_full = {(g, 1): 0.0 for g in ACTIVE_G}   # 鮮度1を定価で販売した数量
            x_sell_disc = {(g, 2): 0.0 for g in ACTIVE_G}   # 鮮度2を値引き価格で販売した数量

            lost_R_today = 0.0  # その日の小売の売り逃し数量

            for g in ACTIVE_G:
                demand = D_DEMAND[(g, t)]  # 需要量D（= Σ_i c_i）を取得

                inv1 = self.I_R[g][1]      # 小売の鮮度1在庫
                inv2 = self.I_R[g][2]      # 小売の鮮度2在庫

                if z_disc[g] == 0:
                    # 値引きなし：状態1のみ売れる（鮮度2は販売対象外）
                    sold1 = min(inv1, demand)  # 需要と在庫の小さい方だけ販売
                    sold2 = 0.0                # 値引き販売なしなので0
                    lost = max(0.0, demand - sold1)  # 需要を満たしきれなかった分が売り逃し

                else:
                    # 値引きあり：状態1(定価)と状態2(値引き)の中からランダムに購入
                    # 期待値として在庫比率で配分する（どちらが選ばれるかの確率が在庫比率に比例するイメージ）
                    avail = inv1 + inv2                           # 販売可能な在庫総量
                    sold_total = min(avail, demand) if avail > 0 else 0.0  # 販売総量（需要と在庫の小さい方）

                    if avail > 0:
                        if pol.ship_order_R.order == "FIFO":
                            # FIFO: 古い在庫（鮮度2）から優先して販売
                            sold2 = min(inv2, sold_total)
                            sold1 = max(0.0, sold_total - sold2)
                        else:
                            # LIFO: 新しい在庫（鮮度1）から優先して販売
                            sold1 = min(inv1, sold_total)
                            sold2 = max(0.0, sold_total - sold1)
                    else:
                        sold1 = 0.0
                        sold2 = 0.0

                    lost = max(0.0, demand - sold_total)         # 需要を満たせなかった分を売り逃しとする

                # 在庫更新：販売した分を差し引く
                self.I_R[g][1] -= sold1
                self.I_R[g][2] -= sold2

                # 記録
                x_sell_full[(g, 1)] = sold1
                x_sell_disc[(g, 2)] = sold2
                lost_R_today += lost      # その日の売り逃しを累計

            # 4) 劣化・廃棄（遷移→状態3は廃棄、さらに「値引きなし」は状態2も廃棄）
            theta_W = pol.storage_W.theta()   # 卸の保存モード（0:通常,1:高品質）
            theta_R = pol.storage_R.theta()   # 小売の保存モード

            I_W_next = {g: {1: 0.0, 2: 0.0, 3: 0.0} for g in ACTIVE_G}  # 翌日の卸在庫
            I_R_next = {g: {1: 0.0, 2: 0.0, 3: 0.0} for g in ACTIVE_G}  # 翌日の小売在庫

            waste_today = 0.0          # 1日分の全体廃棄量
            waste_W_today = 0.0        # 1日分の卸廃棄量
            waste_R_today = 0.0        # 1日分の小売廃棄量

            for g in ACTIVE_G:
                # 卸在庫の鮮度遷移
                I_W_next[g] = inv_next("W", g, self.I_W[g], {1: 0.0, 2: 0.0, 3: 0.0},
                                       theta_W, pol.mekiki.quality_boost())  # 目利き補正付き
                wW = I_W_next[g][3]    # 鮮度3となった量を廃棄扱いとする
                waste_today += wW; waste_W_today += wW  # 卸廃棄を累計
                I_W_next[g][3] = 0.0  # 廃棄したので在庫からは0にする

                # 小売在庫の鮮度遷移
                I_R_next[g] = inv_next("R", g, self.I_R[g], {1: 0.0, 2: 0.0, 3: 0.0},
                                       theta_R, 1.0)  # 小売側では目利き補正なし

                if pol.discount.use_discount:
                    # 値引きあり：状態3のみ廃棄（鮮度2は値引き販売対象なので残す）
                    wR = I_R_next[g][3]   # 鮮度3在庫を廃棄量とする
                    I_R_next[g][3] = 0.0  # 在庫から削除
                else:
                    # 値引きなし：状態2と状態3を廃棄（鮮度2は売れない前提）
                    wR = I_R_next[g][2] + I_R_next[g][3]  # 鮮度2+3を廃棄扱い
                    I_R_next[g][2] = 0.0                  # 在庫から削除
                    I_R_next[g][3] = 0.0

                waste_today += wR; waste_R_today += wR   # 小売廃棄を累計

            self.I_W = I_W_next  # 翌日の卸在庫に更新
            self.I_R = I_R_next  # 翌日の小売在庫に更新

            # 5) 収支・CO2計算
            rev_W  = sum(P_W2R[g] * ship_qty_g[g] for g in ACTIVE_G)   # 卸の売上（小売への販売）
            buy_W  = sum(C_BUY[g] * x_buy[g]      for g in ACTIVE_G)   # 卸の仕入コスト
            mekiki_cost = sum(pol.mekiki.cost_per_unit() * x_buy[g] for g in ACTIVE_G)  # 目利きコスト

            stor_W = theta_W * sum(C_STOR[("W", g, s)] * self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)  # 卸保管費
            ship_c = sum(C_SHIP[g] * ship_qty_g[g] for g in ACTIVE_G)  # 輸送費
            profit_W_day = rev_W - buy_W - mekiki_cost - ship_c - stor_W  # 卸の1日あたり利益

            # 小売売上：状態1は定価、状態2は値引き価格
            rev_R_full = sum(P_LIST[g] * x_sell_full[(g, 1)] for g in ACTIVE_G)           # 定価販売の売上
            rev_R_disc = sum(DELTA[g] * P_LIST[g] * x_sell_disc[(g, 2)] for g in ACTIVE_G)  # 値引き販売の売上
            cost_pur   = sum(P_W2R[g] * ship_qty_g[g] for g in ACTIVE_G)                  # 小売の仕入コスト

            stor_R = theta_R * sum(C_STOR[("R", g, s)] * self.I_R[g][s] for g in ACTIVE_G for s in S_LIST)  # 小売保管費

            disc_fix = sum(C_DISC[g] * (1 if (pol.discount.use_discount and z_disc[g] == 1) else 0) for g in ACTIVE_G)
            # 値引きモードを実際に使ったカテゴリに対する固定費

            profit_R_day = (rev_R_full + rev_R_disc) - cost_pur - stor_R - disc_fix  # 小売の1日あたり利益

            co2_ship = sum(E_SHIP[g] * ship_qty_g[g] for g in ACTIVE_G)  # 輸送由来CO2量
            co2_W_day = ((1 - theta_W) * E_STOR0["W"] + theta_W * E_STOR1["W"]) * \
                        sum(self.I_W[g][s] for g in ACTIVE_G for s in S_LIST)       # 卸保管由来CO2量
            co2_R_day = ((1 - theta_R) * E_STOR0["R"] + theta_R * E_STOR1["R"]) * \
                        sum(self.I_R[g][s] for g in ACTIVE_G for s in S_LIST)       # 小売保管由来CO2量
            co2_day = co2_ship + co2_W_day + co2_R_day                               # 1日トータルCO2量

            self.total_profit_W += profit_W_day                       # 卸利益を累積
            self.total_profit_R += profit_R_day                       # 小売利益を累積
            self.total_waste    += (waste_today + waste_on_arrival_total)  # 廃棄量（輸送中含む）を累積
            self.total_co2      += co2_day                            # CO2量を累積

            self.co2_W += co2_W_day               # 卸CO2累計
            self.co2_R += co2_R_day               # 小売CO2累計
            self.co2_ship += co2_ship             # 輸送CO2累計

            self.waste_W += waste_W_today         # 卸廃棄累計
            self.waste_R += waste_R_today         # 小売廃棄累計
            self.waste_ship += waste_on_arrival_total  # 輸送廃棄累計

            self.lost_R += lost_R_today           # 売り逃し累計

        # T日分のシミュレーションが終わったら指標をまとめて返す
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
def get_safety_candidates_R() -> Dict[str, List[int]]:
    """
    小売の安全在庫候補リストを返す。
    MODE に応じて L または S のみを有効にする。
    """
    inv_L = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    inv_S = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    if MODE == "L":
        return {"L": inv_L, "S": []}
    elif MODE == "S":
        return {"L": [], "S": inv_S}
    else:
        return {"L": inv_L, "S": inv_S}

def get_safety_candidates_W() -> Dict[str, List[int]]:
    """
    卸の安全在庫候補リストを返す。
    """
    base_L = [30, 60, 90, 120, 150, 180, 210, 240, 270]
    base_S = [30, 60, 90, 120, 150, 180, 210, 240, 270]

    if MODE == "L":
        return {"L": base_L, "S": []}
    elif MODE == "S":
        return {"L": [], "S": base_S}
    else:
        return {"L": base_L, "S": base_S}

# ====== ポリシー生成ユーティリティ ======
def random_policy() -> DiscretePolicy:
    """
    卸・小売の戦略をランダムに1つ生成する。
    ベストレスポンス法の初期戦略として利用。
    """
    safR = get_safety_candidates_R()  # 小売安全在庫候補
    safW = get_safety_candidates_W()  # 卸安全在庫候補
    return DiscretePolicy(
        safety_target_W_L=(random.choice(safW["L"]) if safW["L"] else None),
        safety_target_W_S=(random.choice(safW["S"]) if safW["S"] else None),
        mekiki=MekikiLevel(level=random.choice(["high", "low"])),
        storage_W=StorageMode(mode=random.choice(["high", "low"])),
        ship_order_W=ShipOrder(order=random.choice(["FIFO", "LIFO"])),
        safety_target_R_L=(random.choice(safR["L"]) if safR["L"] else None),
        safety_target_R_S=(random.choice(safR["S"]) if safR["S"] else None),
        discount=DiscountPolicy(use_discount=random.choice([True, False])),
        storage_R=StorageMode(mode=random.choice(["high", "low"])),
        ship_order_R=ShipOrder(order=random.choice(["FIFO", "LIFO"])),
    )

def clone_with_W(base: DiscretePolicy, **kw) -> DiscretePolicy:
    """
    既存戦略 base をコピーし、卸側に関する項目のみ kw で上書きした新戦略を作る。
    """
    d = base.__dict__.copy()                           # データクラスを辞書に展開
    for k in ["safety_target_W_L", "safety_target_W_S", "mekiki", "storage_W", "ship_order_W"]:
        if k in kw:
            d[k] = kw[k]
    return DiscretePolicy(**d)



def clone_with_R(base: DiscretePolicy, **kw) -> DiscretePolicy:
    """
    既存戦略 base をコピーし、小売側に関する項目のみ kw で上書きした新戦略を作る。
    """
    d = base.__dict__.copy()
    for k in ["safety_target_R_L", "safety_target_R_S", "discount", "storage_R", "ship_order_R"]:
        if k in kw:
            d[k] = kw[k]
    return DiscretePolicy(**d)

# ====== メトリクス評価 ======
def evaluate_metrics(pol: DiscretePolicy):
    """
    与えられた戦略 pol でシミュレーションを1回走らせ、
    さまざまな指標と戦略自体をまとめた辞書を返す。
    """
    sim = Simulator(pol).run()     # シミュレーションを実行
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
    """
    卸プレイヤーの利得（利益−CO2ペナルティ−廃棄ペナルティ）を計算。
    """
    return res["profit_W"] - PENALTY_CO2_W * res["co2_W"] - PENALTY_WASTE_W * res["waste_W"]

def payoff_R(res: dict) -> float:
    """
    小売プレイヤーの利得を計算。
    廃棄ペナルティには廃棄_小売＋廃棄_輸送が含まれ、売り逃しペナルティも加える。
    """
    eff_waste_R = res["waste_R"] + res["waste_ship"]
    return (res["profit_R"]
            - PENALTY_CO2_R * res["co2_R"]
            - PENALTY_WASTE_R * eff_waste_R
            - PENALTY_LOST_R * res["lost_R"])

# ====== ベストレスポンス探索 ======
def best_response_W(base: DiscretePolicy, samples: int = 40) -> Tuple[DiscretePolicy, dict, float]:
    """
    小売戦略を固定したもとで、卸側のベストレスポンス戦略をランダムサーチで求める。
    samples 回だけ候補戦略を生成して比較する。
    戻り値: (最良戦略, その評価結果, 利得値)
    """
    safW = get_safety_candidates_W()  # 卸安全在庫候補
    choices = []                      # 候補戦略のリスト
    for _ in range(samples):
        cand = clone_with_W(
            base,
            safety_target_W_L=(random.choice(safW["L"]) if safW["L"] else None),
            safety_target_W_S=(random.choice(safW["S"]) if safW["S"] else None),
            mekiki=MekikiLevel(level=random.choice(["high", "low"])),
            storage_W=StorageMode(mode=random.choice(["high", "low"])),
            ship_order_W=ShipOrder(order=random.choice(["FIFO", "LIFO"])),
        )
        choices.append(cand)

    best_pol = base                    # 現時点でのベスト戦略を base で初期化
    best_res = evaluate_metrics(base)  # base 戦略で評価
    best_val = payoff_W(best_res)      # 卸利得を計算

    for pol in choices:                # 各候補についてループ
        res = evaluate_metrics(pol)    # シミュレーションして指標取得
        val = payoff_W(res)            # 卸利得を計算
        if val > best_val:             # これまでより良ければ更新
            best_val = val
            best_pol = pol
            best_res = res

    return best_pol, best_res, best_val

def best_response_R(base: DiscretePolicy, samples: int = 40) -> Tuple[DiscretePolicy, dict, float]:
    """
    卸戦略を固定したもとで、小売側のベストレスポンスをランダムサーチで求める。
    """
    safR = get_safety_candidates_R()
    choices = []
    for _ in range(samples):
        cand = clone_with_R(
            base,
            safety_target_R_L=(random.choice(safR["L"]) if safR["L"] else None),
            safety_target_R_S=(random.choice(safR["S"]) if safR["S"] else None),
            discount=DiscountPolicy(use_discount=random.choice([True, False])),
            storage_R=StorageMode(mode=random.choice(["high", "low"])),
            ship_order_R=ShipOrder(order=random.choice(["FIFO", "LIFO"])),
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
def alternating_best_response(iters: int = 6, init: Optional[DiscretePolicy] = None,
                              samples_W: int = 40, samples_R: int = 40):
    """
    卸と小売が交互にベストレスポンスをとる近似ナッシュ均衡探索。
    iters 回だけ
      RのBR → WのBR を繰り返して戦略を更新する。
    """
    pol = init or random_policy()  # 初期戦略（指定がなければランダム）
    history = []                   # 各ステップの評価結果を記録
    for _ in range(iters):
        pol_R, _, _ = best_response_R(pol, samples=samples_R)  # 小売のBRを計算
        pol = pol_R                                            # 戦略を更新
        pol_W, _, _ = best_response_W(pol, samples=samples_W)  # 卸のBRを計算
        pol = pol_W
        history.append(evaluate_metrics(pol))                  # 現在戦略の評価値を履歴に保存
    return pol, history                                        # 最終戦略と履歴を返す

# ====== 結果表示 ======
def show_result(title: str, res: dict):
    """
    1つの戦略プロファイルに対する結果を整形してコンソールに表示する。
    """
    pol: DiscretePolicy = res["policy"]
    print("=" * 60)
    print(f"[{title}] MODE={MODE}")
    print("-" * 60)
    print(f" 卸利益: {res['profit_W']:.2f} 円 / 小売利益: {res['profit_R']:.2f} 円 / 合計: {res['profit_total']:.2f} 円")
    print(f" CO2: total {res['co2_total']:.3f} kg (W {res['co2_W']:.3f}, R {res['co2_R']:.3f}, ship {res['co2_ship']:.3f})")
    print(f" 廃棄: total {res['waste_total']:.2f} 個 (W {res['waste_W']:.2f}, R {res['waste_R']:.2f}, ship {res['waste_ship']:.2f})")
    print(f" 売り逃し: 小売 lost_R = {res['lost_R']:.2f} 個")
    print("-" * 60)
    if "L" in ACTIVE_G:
        print(f" 卸安全L: {pol.safety_target_W_L}, 小売安全L: {pol.safety_target_R_L}")
    if "S" in ACTIVE_G:
        print(f" 卸安全S: {pol.safety_target_W_S}, 小売安全S: {pol.safety_target_R_S}")
    print(f" 目利き: {pol.mekiki.level}, 卸保存: {pol.storage_W.mode}, 小売保存: {pol.storage_R.mode}")
    print(f" 出荷順序(卸): {pol.ship_order_W.order}, 出荷順序(小売): {pol.ship_order_R.order}, 値引き: {'ON' if pol.discount.use_discount else 'OFF'}")

# ====== 戦略ラベル ======
def label_W(pol: DiscretePolicy) -> str:
    """
    卸戦略を人間が理解しやすい文字列に変換する。
    安全在庫・目利き・保存モード・出荷順序を埋め込む。
    """
    parts = []
    if "L" in ACTIVE_G and pol.safety_target_W_L is not None:
        parts.append(f"L{pol.safety_target_W_L}")
    if "S" in ACTIVE_G and pol.safety_target_W_S is not None:
        parts.append(f"S{pol.safety_target_W_S}")
    parts.append("M" + ("H" if pol.mekiki.level == "high" else "L"))
    parts.append("St" + ("H" if pol.storage_W.mode == "high" else "L"))
    parts.append("O" + ("F" if pol.ship_order_W.order == "FIFO" else "L"))
    return "卸:" + "".join(parts)

def label_R(pol: DiscretePolicy) -> str:
    """
    小売戦略をラベル文字列に変換する。
    安全在庫・値引き・保存モード・出荷順序を含める。
    """
    parts = []
    if "L" in ACTIVE_G and pol.safety_target_R_L is not None:
        parts.append(f"L{pol.safety_target_R_L}")
    if "S" in ACTIVE_G and pol.safety_target_R_S is not None:
        parts.append(f"S{pol.safety_target_R_S}")
    parts.append("D" + ("1" if pol.discount.use_discount else "0"))
    parts.append("St" + ("H" if pol.storage_R.mode == "high" else "L"))
    parts.append("O" + ("F" if pol.ship_order_R.order == "FIFO" else "L"))
    return "小売:" + "".join(parts)

# ====== 全戦略列挙＆並び順定義 ======
def sort_key_W(pol: DiscretePolicy):
    """
    卸戦略の並び順を決めるためのキー関数。
    目利きの有無→保存モード→出荷順→安全在庫水準 の順でソート。
    """
    k_m = 0 if pol.mekiki.level == "high" else 1          # 高目利きを優先
    k_st = 0 if pol.storage_W.mode == "high" else 1       # 高品質保存を優先
    k_o = 0 if pol.ship_order_W.order == "LIFO" else 1      # LIFO を先に
    big = 10**9                                           # None 用の大きな値
    safeL = pol.safety_target_W_L if pol.safety_target_W_L is not None else big
    safeS = pol.safety_target_W_S if pol.safety_target_W_S is not None else big
    return (k_m, k_st, k_o, safeL, safeS)

def sort_key_R(pol: DiscretePolicy):
    """
    小売戦略の並び順を決めるキー関数。
    値引きON→高品質保存→LIFO→安全在庫水準 の順でソート。
    """
    k_d = 0 if pol.discount.use_discount else 1
    k_st = 0 if pol.storage_R.mode == "high" else 1
    k_o = 0 if pol.ship_order_R.order == "LIFO" else 1
    big = 10**9
    safeL = pol.safety_target_R_L if pol.safety_target_R_L is not None else big
    safeS = pol.safety_target_R_S if pol.safety_target_R_S is not None else big
    return (k_d, k_st, k_o, safeL, safeS)

def all_W_strategies(base: DiscretePolicy) -> List[DiscretePolicy]:
    """
    卸側のすべての戦略候補（安全在庫×目利き×保存×出荷順）の組み合わせを列挙し、
    sort_key_W の順でソートしたリストを返す。
    """
    safW = get_safety_candidates_W()
    cand_L = safW["L"] if safW["L"] else [None]
    cand_S = safW["S"] if safW["S"] else [None]

    mekiki_levels = ["high", "low"]
    storage_modes = ["high", "low"]
    ship_orders   = ["LIFO", "FIFO"]

    lst: List[DiscretePolicy] = []
    for ml in mekiki_levels:
        for sm in storage_modes:
            for so in ship_orders:
                for sL in cand_L:
                    for sS in cand_S:
                        pol = clone_with_W(
                            base,
                            safety_target_W_L=sL,
                            safety_target_W_S=sS,
                            mekiki=MekikiLevel(level=ml),
                            storage_W=StorageMode(mode=sm),
                            ship_order_W=ShipOrder(order=so),
                        )
                        lst.append(pol)
    lst.sort(key=sort_key_W)  # 並び順を整える
    return lst

def all_R_strategies(base: DiscretePolicy) -> List[DiscretePolicy]:
    """
    小売側のすべての戦略候補（安全在庫×値引き×保存×出荷順）を列挙してソートする。
    """
    safR = get_safety_candidates_R()
    cand_L = safR["L"] if safR["L"] else [None]
    cand_S = safR["S"] if safR["S"] else [None]

    discount_flags = [True, False]
    storage_modes  = ["high", "low"]
    ship_orders    = ["LIFO", "FIFO"]

    lst: List[DiscretePolicy] = []
    for dflag in discount_flags:
        for sm in storage_modes:
            for so in ship_orders:
                for sL in cand_L:
                    for sS in cand_S:
                        pol = clone_with_R(
                            base,
                            safety_target_R_L=sL,
                            safety_target_R_S=sS,
                            discount=DiscountPolicy(use_discount=dflag),
                            storage_R=StorageMode(mode=sm),
                            ship_order_R=ShipOrder(order=so),
                        )
                        lst.append(pol)
    lst.sort(key=sort_key_R)
    return lst

def merge_WR(w_pol: DiscretePolicy, r_pol: DiscretePolicy) -> DiscretePolicy:
    """
    卸戦略 w_pol と小売戦略 r_pol を1つの DiscretePolicy に合体させる。
    """
    return DiscretePolicy(
        safety_target_W_L=w_pol.safety_target_W_L,
        safety_target_W_S=w_pol.safety_target_W_S,
        mekiki=w_pol.mekiki,
        storage_W=w_pol.storage_W,
        ship_order_W=w_pol.ship_order_W,
        safety_target_R_L=r_pol.safety_target_R_L,
        safety_target_R_S=r_pol.safety_target_R_S,
        discount=r_pol.discount,
        storage_R=r_pol.storage_R,
        ship_order_R=r_pol.ship_order_R,
    )

# ====== 戦略×戦略の利益ヒートマップ ======
def plot_strategy_profit_heatmaps(base_policy: DiscretePolicy,
                                  title_suffix: str = "ペナルティ付き利益ゲーム",
                                  excel_filename: str = "sim_results_S1.xlsx"):
    """
    卸・小売の全戦略組合せについてシミュレーションを行い、
    卸利益・小売利益のヒートマップと詳細結果をExcelに出力する。
    """
    W_list = all_W_strategies(base_policy)  # 全卸戦略リスト
    R_list = all_R_strategies(base_policy)  # 全小売戦略リスト
    nW, nR = len(W_list), len(R_list)       # 戦略数

    profit_W_mat = np.zeros((nW, nR))       # 卸利益の行列（行:卸戦略,列:小売戦略）
    profit_R_mat = np.zeros((nW, nR))       # 小売利益の行列

    data_records = []                       # Excel出力用のレコード配列

    print("\n" + "=" * 80)
    print(f"  戦略プロファイルごとの結果一覧 ({title_suffix})")
    print("=" * 80)

    for i, w_pol in enumerate(W_list):          # 卸戦略インデックス i
        for j, r_pol in enumerate(R_list):      # 小売戦略インデックス j
            pol_ij = merge_WR(w_pol, r_pol)     # 卸・小売戦略を統合
            res = evaluate_metrics(pol_ij)      # シミュレーションして評価値取得

            profit_W_mat[i, j] = res["profit_W"]  # 卸利益を行列に格納
            profit_R_mat[i, j] = res["profit_R"]  # 小売利益を行列に格納

            w_label = label_W(w_pol)            # 卸戦略ラベル
            r_label = label_R(r_pol)            # 小売戦略ラベル

            print("-" * 80)
            print(f"[組み合わせ] 行i={i}, 列j={j}")
            print(f"  卸戦略: {w_label}")
            print(f"  小売戦略: {r_label}")
            print(f"  卸利益: {res['profit_W']:.2f} 円")
            print(f"  小売利益: {res['profit_R']:.2f} 円")
            print(f"  合計利益: {res['profit_total']:.2f} 円")

            record = {
                "行ID": i,
                "列ID": j,
                "卸戦略名": w_label,
                "小売戦略名": r_label,
                "卸利益": res["profit_W"],
                "小売利益": res["profit_R"],
                "合計利益": res["profit_total"],
                "CO2総量": res["co2_total"],
                "CO2_卸": res["co2_W"],
                "CO2_小売": res["co2_R"],
                "CO2_輸送": res["co2_ship"],
                "廃棄総量": res["waste_total"],
                "廃棄_卸": res["waste_W"],
                "廃棄_小売": res["waste_R"],
                "廃棄_輸送": res["waste_ship"],
                "売り逃し数": res["lost_R"],
                "卸安全在庫L": w_pol.safety_target_W_L,
                "卸安全在庫S": w_pol.safety_target_W_S,
                "小売安全在庫L": r_pol.safety_target_R_L,
                "小売安全在庫S": r_pol.safety_target_R_S,
                "小売値引き": r_pol.discount.use_discount,
                "卸保存モード": w_pol.storage_W.mode,
                "小売保存モード": r_pol.storage_R.mode,
                "卸出荷順序": w_pol.ship_order_W.order,
                "小売出荷順序": r_pol.ship_order_R.order,
            }
            data_records.append(record)  # レコードをリストに追加

    df = pd.DataFrame(data_records)  # pandas DataFrame に変換
    try:
        df.to_excel(excel_filename, index=False)  # 指定ファイル名でExcelに出力
        print(f"\n【成功】結果をExcelファイルに出力しました: {excel_filename}")
    except Exception as e:
        print(f"\n【エラー】Excel出力に失敗しました: {e}")  # 出力失敗時のエラーメッセージ

    y_labels = [label_W(p) for p in W_list]  # 行ラベル（卸戦略）
    x_labels = [label_R(p) for p in R_list]  # 列ラベル（小売戦略）

    plt.figure(figsize=(10, 8))
    plt.imshow(profit_W_mat, origin="lower", aspect="auto")  # 卸利益行列をヒートマップ表示
    plt.colorbar(label="卸の利益 [円]")                     # カラーバーにラベル
    plt.xticks(range(nR), x_labels, rotation=90)             # x軸に小売戦略名を縦書きで表示
    plt.yticks(range(nW), y_labels)                          # y軸に卸戦略名
    plt.xlabel("小売の戦略")                                 # x軸ラベル
    plt.ylabel("卸の戦略")                                   # y軸ラベル
    plt.title(f"卸の利益ヒートマップ ({title_suffix})")       # タイトル
    plt.tight_layout()                                       # レイアウト調整

    plt.figure(figsize=(10, 8))
    plt.imshow(profit_R_mat, origin="lower", aspect="auto")  # 小売利益ヒートマップ
    plt.colorbar(label="小売の利益 [円]")
    plt.xticks(range(nR), x_labels, rotation=90)
    plt.yticks(range(nW), y_labels)
    plt.xlabel("小売の戦略")
    plt.ylabel("卸の戦略")
    plt.title(f"小売の利益ヒートマップ ({title_suffix})")
    plt.tight_layout()

    plt.show()  # ヒートマップを表示

# ====== メイン実行 ======
if __name__ == "__main__":  # このファイルが直接実行されたときだけ実行するブロック
    # 0) 客入り（I）と需要（D）を確認
    plot_customers_and_demands(T)  # 客数と需要の推移をグラフ表示

    # 1) 全戦略の結果一覧とヒートマップを表示
    print("\n\n==============================")
    print(" 全戦略プロファイル一覧")
    print("==============================\n")
    pol_dummy = random_policy()  # 任意の基準戦略（安全在庫0以外の情報用）
    plot_strategy_profit_heatmaps(
        pol_dummy,
        title_suffix="全戦略プロファイル（客入り=梶木モデル, c_i∈{0..4}）",
        excel_filename="sim_results_all_S.xlsx"####################################################Excelファイル名################################################################################
    )

    # 2) ナッシュ均衡（交互ベストレスポンスの収束解）
    print("\n\n==============================")
    print(" ナッシュ均衡")
    print("==============================\n")

    pol0 = random_policy()  # 交互BRの初期戦略
    eq_pol, history = alternating_best_response(iters=6, init=pol0, samples_W=30, samples_R=30)
    res_eq = evaluate_metrics(eq_pol)  # 収束戦略の評価
    show_result("ペナルティ付き利益ゲーム（客入り=梶木モデル, c_i∈{0..4}）", res_eq)  # 結果表示
