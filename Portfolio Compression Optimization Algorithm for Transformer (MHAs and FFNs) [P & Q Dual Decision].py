import torch
import torch.nn.functional as F
import numpy as np

# --------- 参数设定(Parameter settings) ---------
B_i = 4
B_j = 6
n_heads = 12 # 注意力头数(Number of attention heads)
N_ffn = 12 # 前馈神经数(Number of feedforward neural network)
dim_model = 512 # 模型维度(Model dimension)
T = 100 # 迭代轮数(Number of iterations)
tau = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 损失函数与目标函数权重(Loss function and objective function weights) ---------
rho1, rho2, rho3 = 0.3, 0.4, 0.3
beta1, beta2 = 0.5, 0.5
alpha1, alpha2 = 0.6, 0.4
chi1, chi2 = 0.7, 0.3 #损失函数与目标函数权重可自行根据实际情况进行动态调整(The loss function and objective function weights can be dynamically adjusted according to actual conditions.)

# --------- 输入初始化(Input initialization) ---------
head_i = torch.randn((n_heads, dim_model), device=device)
x_j = torch.randn((N_ffn, dim_model), device=device)
H3_xj = torch.randn_like(x_j)
Qo_xi = head_i.clone().detach()
Qo_xj = x_j.clone().detach()

# --------- 初始化可学习mask权重(Initialise learnable mask weights) ---------
w_i = torch.rand(n_heads, requires_grad=True, device=device)
w_j = torch.rand(N_ffn, requires_grad=True, device=device)

# --------- 定义函数(Define functions) ---------
def quantize(tensor, delta):
    return delta * torch.round(tensor / delta)

def gumbel_softmax(logits, tau=1.0):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return F.softmax((logits + gumbel) / tau, dim=-1)

def prune_and_apply(heads, w_mask):
    return w_mask.unsqueeze(1) * heads

def quantized_attention(x, delta):
    Wq = torch.nn.Linear(dim_model, dim_model).to(device)
    Wk = torch.nn.Linear(dim_model, dim_model).to(device)
    Wv = torch.nn.Linear(dim_model, dim_model).to(device)
    q = quantize(Wq(x), delta)
    k = quantize(Wk(x), delta)
    v = quantize(Wv(x), delta)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dim_model)
    attn = quantize(F.softmax(scores, dim=-1), delta)
    return quantize(torch.matmul(attn, v), delta)

def quantized_ffn(x, delta):
    W1 = torch.nn.Linear(dim_model, N_ffn).to(device)
    W2 = torch.nn.Linear(N_ffn, dim_model).to(device)
    h1 = quantize(W1(x), delta)
    act = F.relu(h1)
    return quantize(W2(act), delta)

# --------- 主循环优化(Main loop optimisation) ---------
L_best = -float('inf')
w_i_star = w_j_star = Q_xi_star = Q_xj_star = None
delta_star = None

candidate_b = [2, 3, 4, 5, 6, 7, 8]  # b = log2(1/Delta)

for t in range(T):
    wi_mask = gumbel_softmax(w_i, tau)
    wj_mask = gumbel_softmax(w_j, tau)

    if wi_mask.sum() > B_i:
        wi_mask = wi_mask * (B_i / wi_mask.sum())
    if wj_mask.sum() > B_j:
        wj_mask = wj_mask * (B_j / wj_mask.sum())

    pruned_heads = prune_and_apply(head_i, wi_mask)
    pruned_ffn = prune_and_apply(H3_xj, wj_mask)

    for b in candidate_b:
        delta = 1.0 / (2 ** b)

        Q_xi = quantized_attention(pruned_heads, delta)
        Q_xj = quantized_ffn(x_j, delta)

        LP = alpha1 * torch.sum(torch.abs(pruned_heads - head_i)) + \
             alpha2 * torch.sum(torch.abs(pruned_ffn - H3_xj))
        LQ = beta1 * F.mse_loss(Q_xi, Qo_xi) + beta2 * F.mse_loss(Q_xj, Qo_xj)
        Delta_A = LP + LQ

        delta_params = torch.sum(1.0 - wi_mask) + torch.sum(1.0 - wj_mask)
        delta_flops = 1.0 / (torch.sum(wi_mask) + 1e-6) + 1.0 / (torch.sum(wj_mask) + 1e-6)
        delta_time = 1.0 / (torch.sum(wi_mask + wj_mask) + 1e-6)
        Delta_E = rho1 * delta_params + rho2 * delta_flops + rho3 * delta_time

        L = chi1 * Delta_E - chi2 * Delta_A

        if L > L_best:
            L_best = L.item()
            w_i_star = wi_mask.detach().clone()
            w_j_star = wj_mask.detach().clone()
            Q_xi_star = Q_xi.clone()
            Q_xj_star = Q_xj.clone()
            delta_star = delta

# --------- 输出结果(Output results) ---------
print("Best Objective Value:", L_best)
print("Optimal Head Mask (w_i*):", w_i_star)
print("Optimal FFN Mask (w_j*):", w_j_star)
print("Optimal Delta (Quantization Step):", delta_star)
print("Optimal Bit-width b* = log2(1/Delta):", np.log2(1 / delta_star))
