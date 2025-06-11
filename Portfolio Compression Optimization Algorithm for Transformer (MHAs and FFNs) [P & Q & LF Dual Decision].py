import torch
import torch.nn.functional as F
import numpy as np

# --------- 基本参数((Parameter settings)) ---------
B_i = 4
B_j = 6
n_heads = 12 # 注意力头数(Number of attention heads)
N_ffn = 12 # 前馈神经数(Number of feedforward neural network)
dim_model = 512 # 模型维度(Model dimension)
rank_candidates = [8, 16, 32, 64] #秩选择集(Rank selection set)
Delta_candidates = [0.01, 0.02, 0.04]  # 量化位数选择集(Quantitative bit selection set)
T = 100 # 迭代轮数(Number of iterations)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 损失函数与目标函数权重(Loss function and objective function weights) ---------
rho1, rho2, rho3 = 0.3, 0.4, 0.3
beta1, beta2 = 0.5, 0.5
gamma1, gamma2 = 0.6, 0.4
chi1, chi2 = 0.7, 0.3 #损失函数与目标函数权重可自行根据实际情况进行动态调整(The loss function and objective function weights can be dynamically adjusted according to actual conditions.)

# --------- 输入数据((Input initialization)) ---------
x_i = torch.randn((n_heads, dim_model), device=device)
x_j = torch.randn((N_ffn, dim_model), device=device)
H3_xj = torch.randn_like(x_j)
Qo_xi = x_i.clone().detach()
Qo_xj = x_j.clone().detach()

# --------- 定义 Gumbel-Softmax 辅助函数(Define the Gumbel-Softmax auxiliary function) ---------
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0):

    y = gumbel_softmax_sample(logits, temperature)
    return y[:, 0]

# --------- 量化函数(Quantification function) ---------
def quantize(tensor, delta):
    return delta * torch.round(tensor / delta)

def attention_quantized(x, delta):
    q = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x), delta)
    k = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x), delta)
    v = quantize(torch.nn.Linear(dim_model, dim_model).to(device)(x), delta)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dim_model)
    probs_q = quantize(F.softmax(scores, dim=-1), delta)
    return quantize(torch.matmul(probs_q, v), delta)

def ffn_quantized(x, delta):
    w1 = torch.nn.Linear(dim_model, N_ffn).to(device)
    w2 = torch.nn.Linear(N_ffn, dim_model).to(device)
    h1 = quantize(w1(x), delta)
    act = F.relu(h1)
    return quantize(w2(act), delta)

def low_rank_approximation(x, r1, r2):
    U, S, V = torch.svd_lowrank(x, q=r1)
    approx = torch.matmul(U, torch.matmul(torch.diag(S), V.T))
    U2, S2, V2 = torch.svd_lowrank(approx, q=r2)
    return torch.matmul(U2, torch.matmul(torch.diag(S2), V2.T))

# --------- 初始化剪枝logits(Initialise pruning logits) ---------
w_i_logits = torch.randn(n_heads, 2, device=device, requires_grad=True)
w_j_logits = torch.randn(N_ffn, 2, device=device, requires_grad=True)

optimizer = torch.optim.Adam([w_i_logits, w_j_logits], lr=0.01)

# --------- 训练参数(Training parameter) ---------
temperature = 1.0
temperature_decay = 0.95

L_best = -float('inf')
best_results = {}

for t in range(T):
    optimizer.zero_grad()

    w_i = gumbel_softmax(w_i_logits, temperature)  # (n_heads,)
    w_j = gumbel_softmax(w_j_logits, temperature)  # (N_ffn,)

    for r in rank_candidates:
        for r_prime in rank_candidates:
            for delta in Delta_candidates:
                # Step 1: 应用剪枝权重(Apply pruning weights)
                head_pruned = w_i.unsqueeze(1) * x_i  # (n_heads, dim_model)
                ffn_pruned = w_j.unsqueeze(1) * x_j  # (N_ffn, dim_model)

                # Step 2: 量化(quantization)
                Q_xi = attention_quantized(head_pruned, delta)
                Q_xj = ffn_quantized(ffn_pruned, delta)

                # Step 3: 低秩近似(low-rank approximation)
                H3_hat = low_rank_approximation(H3_xj, r, r_prime)

                # Step 4: 感知损失(perceptual loss)
                LP = (torch.sum(torch.abs(w_i.unsqueeze(1)*x_i - x_i)) +
                      torch.sum(torch.abs(ffn_pruned - x_j)))

                LQ = beta1 * F.mse_loss(Q_xi, Qo_xi) + beta2 * F.mse_loss(Q_xj, Qo_xj)

                concat_q = torch.cat([head_pruned[i] for i in range(n_heads)], dim=0)
                concat_orig = torch.cat([x_i[i] for i in range(n_heads)], dim=0)
                Wo = torch.nn.Linear(concat_q.shape[-1], dim_model).to(device)
                LLF = gamma1 * torch.norm(Wo(concat_q - concat_orig)) + gamma2 * torch.norm(H3_hat - H3_xj)

                Delta_A = LP + LQ + LLF

                # Step 5: 压缩收益(Compression gains)
                delta_params = torch.sum(1 - w_i) + torch.sum(1 - w_j)  # 剪枝带来的参数减少
                delta_flops = 1.0 / r + 1.0 / r_prime
                delta_time = 1.0 / (r + r_prime)
                Delta_E = rho1 * delta_params + rho2 * delta_flops + rho3 * delta_time

                # Step 6: 目标函数(Objective function)
                L = chi1 * Delta_E - chi2 * Delta_A

                loss = -L
                loss.backward()

                if L.item() > L_best:
                    L_best = L.item()
                    best_results = {
                        'w_i': w_i.detach().cpu(),
                        'w_j': w_j.detach().cpu(),
                        'r': r,
                        'r_prime': r_prime,
                        'delta': delta,
                        'objective': L_best
                    }

    optimizer.step()
    temperature = max(temperature * temperature_decay, 0.1)

# --------- 输出结果(Output results) ---------
print(f"Best objective value: {best_results['objective']}")
print(f"Best ranks: r={best_results['r']}, r'={best_results['r_prime']}")
print(f"Best quantization delta: {best_results['delta']}")
print(f"Best pruning weights w_i (attention heads): {best_results['w_i']}")
print(f"Best pruning weights w_j (FFN layers): {best_results['w_j']}")