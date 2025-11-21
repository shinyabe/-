# 囚人のジレンマ（超コンパクト版）

actions = ["C", "D"]  # C: 協調, D: 裏切り

# 利得表 (Player0, Player1)
payoff = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# 利得表の表示
print("=== 利得表 ===")
for a0 in actions:
    for a1 in actions:
        u0, u1 = payoff[(a0, a1)]
        print(f"P0={a0}, P1={a1} -> ({u0}, {u1})")

# 純粋戦略ナッシュ均衡の探索
nes = []
for a0 in actions:
    for a1 in actions:
        u0, u1 = payoff[(a0, a1)]

        # P0の最適反応チェック
        br0 = all(payoff[(alt0, a1)][0] <= u0 for alt0 in actions)
        # P1の最適反応チェック
        br1 = all(payoff[(a0, alt1)][1] <= u1 for alt1 in actions)

        if br0 and br1:
            nes.append((a0, a1))

print("\n=== 純粋戦略ナッシュ均衡 ===")
for a0, a1 in nes:
    print(f"NE: P0={a0}, P1={a1}, Payoff={payoff[(a0, a1)]}")
