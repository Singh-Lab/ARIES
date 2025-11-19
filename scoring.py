from utils import *
from itertools import combinations

AA = list("ARNDCQEGHILKMFPSTWYVBJZX*")
blosum_matrix = [
 [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1,-1,-1,-4],
 [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-2, 0,-1,-4],
 [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 4,-3, 0,-1,-4],
 [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4,-3, 1,-1,-4],
 [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-1,-3,-1,-4],
 [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0,-2, 4,-1,-4],
 [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1,-3, 4,-1,-4],
 [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-4,-2,-1,-4],
 [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0,-3, 0,-1,-4],
 [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3, 3,-3,-1,-4],
 [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4, 3,-3,-1,-4],
 [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0,-3, 1,-1,-4],
 [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3, 2,-1,-1,-4],
 [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3, 0,-3,-1,-4],
 [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-3,-1,-1,-4],
 [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0,-2, 0,-1,-4],
 [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1,-1,-1,-4],
 [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-2,-2,-1,-4],
 [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-1,-2,-1,-4],
 [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3, 2,-2,-1,-4],
 [-2,-1, 4, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4,-3, 0,-1,-4],
 [-1,-2,-3,-3,-1,-2,-3,-4,-3, 3, 3,-3, 2, 0,-3,-2,-1,-2,-1, 2,-3, 3,-3,-1,-4],
 [-1, 0, 0, 1,-3, 4, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-2,-2,-2, 0,-3, 4,-1,-4],
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4],
 [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
]

BLOSUM62 = {}
for i, a in enumerate(AA):
    for j, b in enumerate(AA):
        BLOSUM62[(a, b)] = blosum_matrix[i][j]

def blosum(a, b):
    return BLOSUM62.get((a.upper(), b.upper()), 0.0)

def pairwise_blosum(sa, sb):
    score = np.zeros((len(sa), len(sb)))
    for i in range(len(sa)):
        for j in range(len(sb)):
            score[i, j] = blosum(sa[i], sb[j])
    return score

def l2_similarity(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.sqrt(torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8))
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log(.5 * (F.softmax(S, dim=-1) + F.softmax(S, dim=-2)) + 1e-8) + S
    torch.cuda.empty_cache()
    return S