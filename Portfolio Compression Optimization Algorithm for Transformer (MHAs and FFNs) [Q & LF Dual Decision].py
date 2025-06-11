import torch
import torch.nn.functional as F
import numpy as np

# --------- 基本设定参数(Basic parameter settings) ---------
Delta = 0.01
B_i = 4
B_j = 6
n_heads = 12  # 注意力头数(Number of attention heads)
N_ffn = 12  # 前馈神经数(Number of feedforward neural network)
dim_model = 512 # 模型维度(Model dimension)
rank_candidates = [8, 16, 32, 64] # 秩选择集(Rank selection set)
T = 100  # 迭代轮数(Number of iterations)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 损失函数与目标函数权重(Loss function and objective function weights) ---------
rho1, rho2, rho3 = 0.3, 0.4, 0.3
beta1, beta2 = 0.5, 0.5
gamma1, gamma2 = 0.6, 0.4
chi1, chi2 = 0.7, 0.3 #损失函数与目标函数权重可自行根据实际情况进行动态调整(The loss function and objective function weights can be dynamically adjusted according to actual conditions.)

# --------- 输入数据模拟(Input data simulation) ---------
x_i = torch.randn((n_heads, dim_model), device=device)
x_j = torch.randn((N_ffn, dim_model), device=device)
H3_xj = torch.randn_like(x_j)

Qo_xi = x_i.clone().detach()
Qo_xj = x_j.clone().detach()

# --------- 核心函数定义(Core function definition) ---------
def quantize(tensor, delta=Delta):
    return delta * torch.round(tensor / delta)

def attention_quantized(x):
    q = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x))
    k = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x))
    v = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x))
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dim_model)
    probs_q = quantize(F.softmax(scores, dim=-1))
    return quantize(torch.matmul(probs_q, v))

def ffn_quantized(x):
    w1 = torch.nn.Linear(dim_model, N_ffn).to(device)
    w2 = torch.nn.Linear(N_ffn, dim_model).to(device)
    h1 = quantize(w1(x))
    act = F.relu(h1)
    return quantize(w2(act))

def low_rank_approximation(x, r1, r2):
    U, S, V = torch.svd_lowrank(x, q=r1)
    approx = torch.matmul(U, torch.matmul(torch.diag(S), V.T))
    U2, S2, V2 = torch.svd_lowrank(approx, q=r2)
    return torch.matmul(U2, torch.matmul(torch.diag(S2), V2.T))

# --------- 主循环优化(Main loop optimisation) ---------
L_best = -float('inf')
Q_xi_star, Q_xj_star, r_best, r_prime_best = None, None, None, None

for t in range(T):
    for r in rank_candidates:
        for r_prime in rank_candidates:
            Q_xi = attention_quantized(x_i)
            Q_xj = ffn_quantized(x_j)

            H3_hat = low_rank_approximation(H3_xj, r, r_prime)

            # 感知损失(perceptual loss)
            LQ = beta1 * F.mse_loss(Q_xi, Qo_xi) + beta2 * F.mse_loss(Q_xj, Qo_xj)
            concat_q = torch.cat([Q_xi[i] for i in range(n_heads)], dim=0)
            concat_orig = torch.cat([x_i[i] for i in range(n_heads)], dim=0)
            Wo = torch.nn.Linear(concat_q.shape[-1], dim_model).to(device)
            LLF = gamma1 * torch.norm(Wo(concat_q - concat_orig)) + gamma2 * torch.norm(H3_hat - H3_xj)
            Delta_A = LQ + LLF

            # 简化压缩收益估计(Simplified compression gain estimation)
            delta_params = (n_heads + N_ffn) * dim_model - (B_i + B_j) * dim_model
            delta_flops = 1.0 / r + 1.0 / r_prime
            delta_time = 1.0 / (r + r_prime)
            Delta_E = rho1 * delta_params + rho2 * delta_flops + rho3 * delta_time

            # 优化目标函数(Optimise the objective function)
            L = chi1 * Delta_E - chi2 * Delta_A

            if L > L_best:
                L_best = L
                Q_xi_star = Q_xi.clone()
                Q_xj_star = Q_xj.clone()
                r_best, r_prime_best = r, r_prime

# --------- 输出最终结果(Output the final result) ---------
print("Best ranks:", r_best, r_prime_best)
print("Best objective value:", L_best)