from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Dict, Tuple, List

# 乱数を固定
random.seed(42)
#変更後のやつ

# ------------------------------
# 問題サイズ・集合の定義
# ------------------------------
T = 30                             # 計算する日数
G_LIST = ['L', 'S']                # 商品カテゴリ：L=根菜（長持ち）, S=葉物（短命）
S_LIST = [1, 2, 3]                 # 鮮度状態：1=良、2=やや劣化、3=販売不可(廃棄)
N_LIST = ['W', 'R']                # 節点：W=卸、R=小売

# ------------------------------
# 価格・費用・需要などのパラメータ
# ------------------------------
P_LIST = {'L': 180.0, 'S': 180.0}  # 小売定価
DELTA  = {'L': 0.8,   'S': 0.7}    # 割引率（割引価格＝DELTA[g]*P_LIST[g]）

P_W2R  = {'L': 120.0, 'S': 120.0}  # 卸→小売の取引価格
C_BUY  = {'L': 90.0,  'S': 90.0}   # 卸の仕入単価

# 保管コスト（節点n×カテゴリg×状態sごとに設定）
C_STOR = {
    ('W','L',1):1.0, ('W','L',2):1.0, ('W','L',3):1.0,
    ('W','S',1):1.2, ('W','S',2):1.2, ('W','S',3):1.2,
    ('R','L',1):1.5, ('R','L',2):1.5, ('R','L',3):1.5,
    ('R','S',1):2.0, ('R','S',2):2.0, ('R','S',3):2.0,
}

C_SHIP = {'L': 8.0, 'S': 8.0}      # 配送費（円/個）
C_DISC = {'L': 50.0, 'S': 50.0}    # 割引運用の固定費（ONの日だけ課す）

# 日別の需要上限（定価・割引）。実務では価格連動の需要関数に置き換え可
D_FULL = {('L',t): 50.0 for t in range(1,T+1)}
D_FULL.update({('S',t): 60.0 for t in range(1,T+1)})
D_DISC = {('L',t): 30.0 for t in range(1,T+1)}
D_DISC.update({('S',t): 50.0 for t in range(1,T+1)})

# 初期在庫：基本ゼロ。小売にだけS(葉物)の状態1を少し持たせる例
I_INIT = { (n,g,s): 0.0 for n in N_LIST for g in G_LIST for s in S_LIST }
I_INIT[('R','S',1)] = 10.0

# 保管容量（ここでは使っていないが、容量制約を入れる拡張が容易）
C_STOR_CAP = {'W': 1000.0, 'R': 400.0}

# CO2排出原単位（保存強度θ=0/1で異なる値）
E_STOR0 = {'W': 0.002, 'R': 0.003}  # 通常保存
E_STOR1 = {'W': 0.004, 'R': 0.006}  # 高品質保存（電力↑）
E_SHIP  = {'L': 0.01,  'S': 0.01}   # 配送（kgCO2/個）
# 配送1便あたりのCO2排出量（kgCO2/便）
E_TRIP = 2.5        # 例：2.5 kgCO2/便（トラック種別・距離で調整）

# 1便で運べる最大個数（車両容量）。配送回数=ceil(当日出荷総数 / CAP_TRUCK)
CAP_TRUCK = 200     # 例：200 個/便（適宜変更）


# ------------------------------
# 鮮度遷移確率（θ=0のP0, θ=1のP1）
# ・各(s_from)で(s_to)への和が1になるよう設定
# ・根菜は劣化が遅く、葉物は速い
# ------------------------------
P0 = {}
P1 = {}
for n in N_LIST:
    # 根菜 L
    P0[(n,'L',1,1)] = 0.85; P0[(n,'L',1,2)] = 0.15; P0[(n,'L',1,3)] = 0.0  #通常保存
    P0[(n,'L',2,1)] = 0.0;  P0[(n,'L',2,2)] = 0.70; P0[(n,'L',2,3)] = 0.30
    P0[(n,'L',3,1)] = 0.0;  P0[(n,'L',3,2)] = 0.0;  P0[(n,'L',3,3)] = 1.00
    P1[(n,'L',1,1)] = 0.92; P1[(n,'L',1,2)] = 0.08; P1[(n,'L',1,3)] = 0.0  #冷蔵保存
    P1[(n,'L',2,1)] = 0.0;  P1[(n,'L',2,2)] = 0.80; P1[(n,'L',2,3)] = 0.20
    P1[(n,'L',3,1)] = 0.0;  P1[(n,'L',3,2)] = 0.0;  P1[(n,'L',3,3)] = 1.00
    # 葉物 S
    P0[(n,'S',1,1)] = 0.70; P0[(n,'S',1,2)] = 0.30; P0[(n,'S',1,3)] = 0.0  #通常保存
    P0[(n,'S',2,1)] = 0.0;  P0[(n,'S',2,2)] = 0.55; P0[(n,'S',2,3)] = 0.45
    P0[(n,'S',3,1)] = 0.0;  P0[(n,'S',3,2)] = 0.0;  P0[(n,'S',3,3)] = 1.00
    P1[(n,'S',1,1)] = 0.80; P1[(n,'S',1,2)] = 0.20; P1[(n,'S',1,3)] = 0.0  #冷蔵保存
    P1[(n,'S',2,1)] = 0.0;  P1[(n,'S',2,2)] = 0.65; P1[(n,'S',2,3)] = 0.35
    P1[(n,'S',3,1)] = 0.0;  P1[(n,'S',3,2)] = 0.0;  P1[(n,'S',3,3)] = 1.00

# ------------------------------
# 目的関数の重み（利潤を最大化し、CO2・廃棄は罰則で最小化）
# ------------------------------
LAMBDA_W = 1.0
LAMBDA_R = 1.0
ALPHA_CO2 = 50.0   # 1 kgCO2 あたりのペナルティ（円相当）
BETA_WASTE = 5.0   # 1 個廃棄あたりのペナルティ

# ------------------------------
# 探索する「政策」の定義（意思決定ルール）
# ------------------------------
@dataclass
class Policy:
    buy_L: float       # 根菜の日次仕入量（定数方針）
    buy_S: float       # 葉物の日次仕入量（定数方針）
    th_theta_W: float  # 卸の高品質保存ONにする在庫総量の閾値
    th_theta_R: float  # 小売の高品質保存ONにする在庫総量の閾値
    target_R_L: float  # 小売の根菜目標在庫（状態1+2）
    target_R_S: float  # 小売の葉物目標在庫（状態1+2）
    th_disc_L: float   # 根菜：状態2在庫がこの閾値以上で割引ON
    th_disc_S: float   # 葉物：状態2在庫がこの閾値以上で割引ON

def random_policy() -> Policy:
    """ ランダムに初期政策を生成（ランダム探索の初期点） """
    return Policy(
        buy_L=random.uniform(20, 80),
        buy_S=random.uniform(20, 80),
        th_theta_W=random.uniform(80, 200),
        th_theta_R=random.uniform(40, 120),
        target_R_L=random.uniform(40, 120),
        target_R_S=random.uniform(40, 120),
        th_disc_L=random.uniform(10, 80),
        th_disc_S=random.uniform(10, 80),
    )

def neighbor_policy(p: Policy, scale: float=0.15) -> Policy:
    """ 既存政策の近傍（少しだけ値を揺らす）を作る：局所探索用 """
    q = Policy(**p.__dict__)
    def jiggle(x, lo, hi):
        span = (hi - lo) * scale
        return max(lo, min(hi, x + random.uniform(-span, span)))
    q.buy_L = jiggle(p.buy_L,  0, 150)
    q.buy_S = jiggle(p.buy_S,  0, 150)
    q.th_theta_W = jiggle(p.th_theta_W,  0, 300)
    q.th_theta_R = jiggle(p.th_theta_R,  0, 300)
    q.target_R_L = jiggle(p.target_R_L, 0, 200)
    q.target_R_S = jiggle(p.target_R_S, 0, 200)
    q.th_disc_L  = jiggle(p.th_disc_L,  0, 200)
    q.th_disc_S  = jiggle(p.th_disc_S,  0, 200)
    return q

# ------------------------------
# 劣化遷移の計算（保存強度θで線形補間）
# ------------------------------
def peff(n,g,sf,st,theta):
    return (1-theta)*P0[(n,g,sf,st)] + theta*P1[(n,g,sf,st)]

def inv_next(n,g,inv,arrival,theta):
    """
    在庫ベクトル inv(状態→個数) と 当日到着 arrival を合計し、
    1日分の鮮度遷移で s_from→s_to へ振り分けた在庫ベクトルを返す。
    """
    out = {1:0.0,2:0.0,3:0.0}
    for sf in S_LIST:
        total_sf = inv.get(sf,0.0) + arrival.get(sf,0.0)
        if total_sf<=0:
            continue
        for st in S_LIST:
            out[st] += peff(n,g,sf,st,theta)*total_sf
    return out

# ------------------------------
# シミュレーション（与えた政策でT日回す）
# ------------------------------
@dataclass
class SimResult:
    objective: float  # 目的値（利潤合計 − 罰則）
    profit_W: float   # 卸利潤合計
    profit_R: float   # 小売利潤合計
    co2: float        # CO2排出量累計
    waste: float      # 廃棄個数累計
    history: List[dict]  # 必要なら日別ログを入れる

def simulate(policy:Policy)->SimResult:
    # 在庫の初期化（辞書：カテゴリ→{状態→数量}）
    I_W = { g:{1:0.0,2:0.0,3:0.0} for g in G_LIST }
    I_R = { g:{1:0.0,2:0.0,3:0.0} for g in G_LIST }
    for (n,g,s),v in I_INIT.items():
        if n=='W': I_W[g][s]=v
        else:      I_R[g][s]=v

    profit_W=profit_R=co2=waste_total=0.0
    hist=[]

    # --- 日次ループ ---
    for t in range(1,T+1):
        # 1) 仕入れ量（定数方針）と、在庫総量による保存強度ON/OFF
        x_buy={'L':policy.buy_L,'S':policy.buy_S}
        theta_W=1 if sum(I_W[g][s] for g in G_LIST for s in S_LIST)>=policy.th_theta_W else 0
        theta_R=1 if sum(I_R[g][s] for g in G_LIST for s in S_LIST)>=policy.th_theta_R else 0

        # 2) 卸→小売の出荷（小売目標在庫まで補充：s=1優先、次にs=2）
        x_ship={(g,s):0.0 for g in G_LIST for s in S_LIST}
        target_R={'L':policy.target_R_L,'S':policy.target_R_S}
        for g in G_LIST:
            need=max(0.0,target_R[g]-(I_R[g][1]+I_R[g][2]))
            take1=min(I_W[g][1]+x_buy[g],need)              # s=1 在庫＋当日入荷から優先
            x_ship[(g,1)]=take1; need-=take1
            take2=min(I_W[g][2],max(0.0,need))              # 足りなければ s=2 から
            x_ship[(g,2)]=take2; need-=take2

        # 3) 卸在庫の更新（出荷控除→到着加算→劣化遷移）。状態3は即日廃棄
        I_W_next={ g:{1:0.0,2:0.0,3:0.0} for g in G_LIST }
        waste_W_today=0.0
        for g in G_LIST:
            arrival_W={1:x_buy[g],2:0.0,3:0.0}             # 仕入は状態1として入庫
            avail_after={                                   # 出荷後の残量
                1:max(0,I_W[g][1]-x_ship[(g,1)]),
                2:max(0,I_W[g][2]-x_ship[(g,2)]),
                3:I_W[g][3]
            }
            I_W_next[g]=inv_next('W',g,avail_after,arrival_W,theta_W)
            waste_W_today+=I_W_next[g][3]                  # 状態3は廃棄として数える
            I_W_next[g][3]=0.0
        I_W=I_W_next

        # 4) 小売：到着反映→販売→劣化遷移。割引は状態2在庫が閾値超でON
        I_R_after={ g:{
            1:I_R[g][1]+x_ship[(g,1)],
            2:I_R[g][2]+x_ship[(g,2)],
            3:I_R[g][3]
        } for g in G_LIST }

        z_disc={
            'L':1 if I_R_after['L'][2]>=policy.th_disc_L else 0,
            'S':1 if I_R_after['S'][2]>=policy.th_disc_S else 0
        }

        # 定価販売→不足分をs=2で補う→（割引ONなら）割引販売
        x_sell_full={(g,s):0.0 for g in G_LIST for s in [1,2]}
        x_sell_disc={(g,2):0.0 for g in G_LIST}
        for g in G_LIST:
            demand_full=D_FULL[(g,t)]
            sell1=min(I_R_after[g][1],demand_full)         # まずs=1を定価で販売
            x_sell_full[(g,1)]=sell1; demand_full-=sell1
            sell2_full=min(I_R_after[g][2],max(0,demand_full))  # それでも足りなければs=2を定価で
            x_sell_full[(g,2)]=sell2_full
            I_R_after[g][1]-=sell1; I_R_after[g][2]-=sell2_full

            if z_disc[g]==1:                                # 割引ONの時だけ割引需要を受ける
                demand_disc=D_DISC[(g,t)]
                sell2_disc=min(I_R_after[g][2],demand_disc)
                x_sell_disc[(g,2)]=sell2_disc
                I_R_after[g][2]-=sell2_disc

        # 販売後の在庫を劣化遷移。状態3は廃棄
        I_R_next={ g:{1:0.0,2:0.0,3:0.0} for g in G_LIST }
        waste_R_today=0.0
        for g in G_LIST:
            I_R_next[g]=inv_next('R',g,I_R_after[g],{1:0.0,2:0.0,3:0.0},theta_R)
            waste_R_today+=I_R_next[g][3]
            I_R_next[g][3]=0.0
        I_R=I_R_next

        # 5) 利潤（卸・小売）と CO2、廃棄の集計
        ship_qty_g={g:x_ship[(g,1)]+x_ship[(g,2)] for g in G_LIST}

        # 卸：売上 − 仕入 − 配送費 − 保管費
        rev_W   = sum(P_W2R[g]*ship_qty_g[g] for g in G_LIST)
        buy_W   = sum(C_BUY[g]*x_buy[g]      for g in G_LIST)
        ship_c  = sum(C_SHIP[g]*ship_qty_g[g]for g in G_LIST)
        stor_W  = sum(C_STOR[('W',g,s)]*I_W[g][s] for g in G_LIST for s in S_LIST)
        profit_W+=(rev_W-buy_W-ship_c-stor_W)

        # 小売：売上(定価＋割引) − 仕入 − 保管費 − 割引固定費
        rev_R_full = sum(P_LIST[g]*x_sell_full[(g,1)] for g in G_LIST) \
                   + sum(P_LIST[g]*x_sell_full[(g,2)] for g in G_LIST)
        rev_R_disc = sum(DELTA[g]*P_LIST[g]*x_sell_disc[(g,2)] for g in G_LIST)
        cost_pur   = sum(P_W2R[g]*ship_qty_g[g] for g in G_LIST)
        stor_R     = sum(C_STOR[('R',g,s)]*I_R[g][s] for g in G_LIST for s in S_LIST)
        disc_fix   = sum(C_DISC[g]*z_disc[g] for g in G_LIST)
        profit_R  += (rev_R_full+rev_R_disc) - cost_pur - stor_R - disc_fix

        # CO2（配送＋保管）。※保管はθに応じた原単位×在庫量の合計
        # 出荷総数から配送回数を計算（容量で割って切り上げ）
        total_ship = sum(ship_qty_g.values())
        trips = math.ceil(total_ship / CAP_TRUCK) if total_ship > 0 else 0

        # 配送CO2 = (配送1回あたり排出 × 配送回数) + (商品1個あたり排出 × 出荷数)
        co2_ship = E_TRIP * trips + sum(E_SHIP[g] * ship_qty_g[g] for g in G_LIST)

        co2_stor = (
            ((1-theta_W)*E_STOR0['W']+theta_W*E_STOR1['W']) * sum(I_W[g][s] for g in G_LIST for s in S_LIST)
          + ((1-theta_R)*E_STOR0['R']+theta_R*E_STOR1['R']) * sum(I_R[g][s] for g in G_LIST for s in S_LIST)
        )
        co2 += (co2_ship + co2_stor)

        # 廃棄（卸＋小売のその日の状態3）を加算
        waste_total += (waste_W_today + waste_R_today)

    # 6) 目的関数：利潤合計 − (CO2ペナルティ＋廃棄ペナルティ)
    obj = LAMBDA_W*profit_W + LAMBDA_R*profit_R - ALPHA_CO2*co2 - BETA_WASTE*waste_total
    return SimResult(obj,profit_W,profit_R,co2,waste_total,[])

# ------------------------------
# ランダム探索 + 近傍探索
# ------------------------------
def run_random_search(N_INIT:int=50,N_NEIGH:int=20)->Tuple[Policy,SimResult]:
    """
    1) ランダムに N_INIT 個の政策を作って一番良いものを選ぶ
    2) その政策の近傍を N_NEIGH 回試して、さらに良いものがあれば更新
    """
    best_p=None; best_r=None
    for _ in range(N_INIT):
        p=random_policy(); r=simulate(p)
        if (best_r is None) or (r.objective>best_r.objective):
            best_p,best_r=p,r
    for _ in range(N_NEIGH):
        q=neighbor_policy(best_p,0.2); rq=simulate(q)
        if rq.objective>best_r.objective:
            best_p,best_r=q,rq
    return best_p,best_r

# ------------------------------
# 実行部
# ------------------------------
if __name__=='__main__':
    best_policy,best_result=run_random_search(80,60)
    print('=== 最良政策 ==='); print(best_policy)
    print('\n=== 成績 ===')
    print(f'目的値:{best_result.objective:.2f}（円）')
    print(f'卸利潤:{best_result.profit_W:.2f}（円）')
    print(f'小売利潤:{best_result.profit_R:.2f}（円）')
    print(f'CO2:{best_result.co2:.3f}（kg）')
    print(f'廃棄:{best_result.waste:.2f}（個）')
