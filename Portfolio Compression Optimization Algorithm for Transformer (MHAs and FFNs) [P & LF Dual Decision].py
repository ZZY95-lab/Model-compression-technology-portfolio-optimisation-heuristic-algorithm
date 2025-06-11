import torch
import torch.nn.functional as F
import numpy as np

# --------- 基本设定参数(Basic parameter settings) ---------
Delta = 0.01
n_heads = 12 # 注意力头数(Number of attention heads)
N_ffn = 12 # 前馈神经数(Number of feedforward neural network)
dim_model = 512 # 模型维度(Model dimension)
rank_candidates = [8, 16, 32, 64] #秩选择集(Rank selection set)
T = 100 # 迭代轮数(Number of iterations)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 损失函数与目标函数权重(Loss & objective weights) ---------
rho1, rho2 = 0.7, 0.3
alpha1, alpha2 = 0.5, 0.5
gamma1, gamma2 = 0.6, 0.4
chi1, chi2 = 0.7, 0.3

# --------- 输入数据模拟(Input data) ---------
x_i = torch.randn((n_heads, dim_model), device=device)
H3_xj = torch.randn((N_ffn, dim_model), device=device)

wi_logits = torch.randn(n_heads, requires_grad=True, device=device)
wj_logits = torch.randn(N_ffn, requires_grad=True, device=device)

# --------- 低秩近似函数(Low-rank approximation function) ---------
def low_rank_approximation(x, r1, r2):
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    Ur = U[:, :r1]
    Sr = S[:r1]
    Vhr = Vh[:r1, :]
    approx = Ur @ torch.diag(Sr) @ Vhr
    U2, S2, Vh2 = torch.linalg.svd(approx, full_matrices=False)
    U2r = U2[:, :r2]
    S2r = S2[:r2]
    Vh2r = Vh2[:r2, :]
    return U2r @ torch.diag(S2r) @ Vh2r

# --------- Gumbel-Softmax 采样函数(Gumbel-Softmax sampling function) ---------
def gumbel_softmax_sample(logits, temperature=1.0):
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + noise
    return F.softmax(y / temperature, dim=-1)

def sample_weights(logits, temperature=1.0):

    return torch.sigmoid(logits / temperature)

# --------- 主循环优化(Main loop optimisation) ---------
optimizer = torch.optim.Adam([wi_logits, wj_logits], lr=0.05)
best_L = -float('inf')
best_wi, best_wj = None, None
best_r, best_rp = None, None

for t in range(T):
    temperature = max(0.5, 1.0 - t*0.01)
    w_i = sample_weights(wi_logits, temperature)  # shape (n_heads,), 值在(0,1)之间
    w_j = sample_weights(wj_logits, temperature)  # shape (N_ffn,)

    for r in rank_candidates:
        for r_prime in rank_candidates:
            # Step 1: Applying pruning
            head_star = w_i.unsqueeze(1) * x_i            # shape (n_heads, dim_model)
            ffn_star = w_j.unsqueeze(1) * H3_xj           # shape (N_ffn, dim_model)

            # Step 2: Construct compressed outputs
            concat_star = torch.cat([head_star[i] for i in range(n_heads)], dim=0)
            concat_orig = torch.cat([x_i[i] for i in range(n_heads)], dim=0)

            H3_hat = low_rank_approximation(H3_xj, r, r_prime)

            # Step 3: Calculating perceived loss
            LP_star = alpha1 * (torch.abs(head_star - x_i).sum()) + alpha2 * (torch.abs(ffn_star - H3_xj).sum())
            LLF_star = gamma1 * torch.norm(concat_star - concat_orig, p=2) + gamma2 * torch.norm(H3_hat - H3_xj, p=2)
            Delta_A2 = LP_star + LLF_star

            # Step 4: Calculating gains from compressed resources
            Delta_params = torch.sum(1 - w_i) + torch.sum(1 - w_j)  # 这里用sum代替，体现节省比例
            Delta_time = 1.0 / r + 1.0 / r_prime
            Delta_E2 = rho1 * Delta_params + rho2 * Delta_time

            # Step 5: Construct optimization objective
            L = chi1 * Delta_E2 - chi2 * Delta_A2

            optimizer.zero_grad()
            (-L).backward()
            optimizer.step()

            if L.item() > best_L:
                best_L = L.item()
                best_wi = w_i.detach().clone()
                best_wj = w_j.detach().clone()
                best_r = r
                best_rp = r_prime

    if t % 10 == 0 or t == T-1:
        print(f"Iter {t}, Best L: {best_L:.4f}, Best ranks: r={best_r}, r'={best_rp}")

# --------- 输出最终结果(Output the final result) ---------
print("\n=== Final Best Results ===")
print("Best ranks: r = {}, r' = {}".format(best_r, best_rp))
print("Best objective value L = {:.4f}".format(best_L))
print("Best pruning weights for attention heads (w_i):", best_wi)
print("Best pruning weights for FFN (w_j):", best_wj)