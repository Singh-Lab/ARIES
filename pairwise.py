from utils import *
from scoring import *

class WavefrontDP:
    def __init__(self, **kwargs):
        self.batch = kwargs.get('batch', True)

    def init_DP(self, M):
        raise NotImplementedError
    
    def init_DP_batch(self, M):
        raise NotImplementedError

    def recurrence(self, DP, M, i_idx, j_idx):
        raise NotImplementedError
    
    def recurrence_batch(self, DP, M, i_idx, j_idx):
        raise NotImplementedError

    def backtrack(self, DP, M, L1, L2, **kwargs):
        raise NotImplementedError

    def backtrack_batch(self, DP, M, L1, L2, **kwargs):
        raise NotImplementedError

    def align_single(self, M):
        assert isinstance(M, torch.Tensor), 'not a tensor'
        L1, L2 = M.shape
        DP = self.init_DP(M)
        for s in range(2, L1 + L2 + 1):
            i_idx = torch.arange(1, min(s, L1 + 1), device=M.device)
            j_idx = s - i_idx
            mask = (j_idx >= 1) & (j_idx <= L2)
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            if i_idx.numel() == 0:
                continue
            self.recurrence(DP, M, i_idx, j_idx)
        return self.backtrack(DP, M, L1, L2)
    
    def align_batch(self, M):
        assert isinstance(M, list), 'not a list of sim matrices'
        L1 = torch.tensor([m.shape[0] for m in M])
        L2 = torch.tensor([m.shape[1] for m in M])
        L1_max, L2_max, num_alns = torch.max(L1).item(), torch.max(L2).item(), len(M)
        M_pad = torch.zeros((num_alns, L1_max, L2_max), device=M[0].device)
        for i, m in enumerate(M):
            M_pad[i, :m.shape[0], :m.shape[1]] += m
        DP = self.init_DP_batch(M, M_pad)
        
        for s in range(2, L1_max + L2_max + 1):
            i_idx = torch.arange(1, min(s, L1_max + 1), device=M_pad.device)
            j_idx = s - i_idx
            mask = (j_idx >= 1) & (j_idx <= L2_max)
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            if i_idx.numel() == 0:
                continue
            self.recurrence_batch(DP, M_pad, i_idx, j_idx)
        best_scores, paths = self.backtrack_batch(DP, M_pad, L1, L2)
        del M, DP, M_pad
        torch.cuda.empty_cache()
        return best_scores, paths
    
    def align(self, M):
        return self.align_batch(M) if self.batch else self.align_single(M)
    
    # make sure to send e1, e2 to cuda first
    def align_embeddings(self, e1, e2, w, reciprocal=200., blur=3., return_scores=False):
        if self.batch:
            assert isinstance(e2, list) and isinstance(e1, torch.Tensor), "input 2 must be list, input 1 tensor"
            M = [l2_similarity(e1, e, w, reciprocal, blur, device='cuda') for e in e2]
        else:
            assert isinstance(e1, torch.Tensor) and isinstance(e2, torch.Tensor), "both inputs must be tensors"
            M = l2_similarity(e1, e2, w, reciprocal, blur, device='cuda')
        if return_scores:
            return *self.align(M), M
        else:
            return self.align(M)

class DTW(WavefrontDP):
    def __init__(self, **kwargs):
        super(DTW, self).__init__(**kwargs)

    def init_DP(self, M):
        L1, L2 = M.shape
        DP = torch.zeros((L1 + 1, L2 + 1), device=M.device)
        DP[0, 1:] = -float('inf')
        DP[1:, 0] = -float('inf')
        return DP
    
    def init_DP_batch(self, M, M_pad):
        L1 = torch.tensor([m.shape[0] for m in M])
        L2 = torch.tensor([m.shape[1] for m in M])
        L1_max, L2_max, num_alns = torch.max(L1).item(), torch.max(L2).item(), len(M)
        DP = torch.zeros((num_alns, L1_max + 1, L2_max + 1), device=M_pad.device)
        DP[:, 0, 1:] = -float('inf')
        DP[:, 1:, 0] = -float('inf')
        return DP
    
    def recurrence(self, DP, M, i_idx, j_idx):
        diag, vert, hori = DP[i_idx-1, j_idx-1], DP[i_idx-1, j_idx], DP[i_idx, j_idx-1]
        DP[i_idx, j_idx] = torch.maximum(torch.maximum(diag, vert), hori) + M[i_idx-1, j_idx-1]
    
    def recurrence_batch(self, DP, M, i_idx, j_idx):
        diag, vert, hori = DP[:, i_idx-1, j_idx-1], DP[:, i_idx-1, j_idx], DP[:, i_idx, j_idx-1]
        DP[:, i_idx, j_idx] = torch.stack([diag, vert, hori], dim=0).max(dim=0).values + M[:, i_idx-1, j_idx-1]

    def backtrack(self, DP, M, L1, L2, **kwargs):
        i, j = L1, L2
        path = []
        while i and j:
            path.append((i - 1, j - 1))
            prev_scores = torch.tensor([
                DP[i - 1, j - 1],
                DP[i - 1, j],
                DP[i, j - 1]
            ], device=DP.device)
            move = torch.argmax(prev_scores).item()
            if move == 0 or move == 1:
                i -= 1
            if move == 0 or move == 2:
                j -= 1
        while i:
            path.append((i - 1, j))
            i = i - 1
        while j:
            path.append((i, j - 1))
            j = j - 1
        path.reverse()
        return DP[L1, L2].item(), path
    
    def backtrack_batch(self, DP, M, L1, L2, **kwargs):
        best_scores, paths = [], []
        for k in range(M.shape[0]):
            score, path = self.backtrack(DP[k], M[k], L1[k].item(), L2[k].item())
            best_scores.append(score)
            paths.append(path)
        return best_scores, paths

class TurnPenaltyDTW(WavefrontDP):
    def __init__(self, **kwargs):
        super(TurnPenaltyDTW, self).__init__(**kwargs)
        self.tau = kwargs.get('tau', 1.)
    
    def init_DP(self, M):
        L1, L2 = M.shape
        self.penalty = torch.median(M) * self.tau
        DP = torch.full((3, L1 + 1, L2 + 1), -float('inf'), device=M.device)
        DP[:, 0, 0] = 0.0
        return DP
    
    def init_DP_batch(self, M, M_pad):
        L1 = torch.tensor([m.shape[0] for m in M])
        L2 = torch.tensor([m.shape[1] for m in M])
        self.penalty = torch.tensor([torch.median(m) * self.tau for m in M], device=M_pad.device)
        L1_max, L2_max, num_alns = torch.max(L1).item(), torch.max(L2).item(), len(M)
        DP = torch.full((num_alns, 3, L1_max + 1, L2_max + 1), -float('inf'), device=M_pad.device)
        DP[:, :, 0, 0] = 0.0
        return DP
    
    def recurrence(self, DP, M, i_idx, j_idx):
        move = torch.stack([
            DP[:, i_idx - 1, j_idx - 1],
            DP[:, i_idx - 1, j_idx],
            DP[:, i_idx, j_idx - 1]
        ], dim=0)
        move[0, 0] -= self.penalty
        move[0, 2] -= self.penalty
        move[1, 1] -= self.penalty
        move[1, 2] -= self.penalty
        move[2, 0] -= self.penalty
        move[2, 1] -= self.penalty
        DP[:, i_idx, j_idx] = move.max(dim=1).values + M[i_idx - 1, j_idx - 1]

    def recurrence_batch(self, DP, M, i_idx, j_idx):
        move = torch.stack([
            DP[:, :, i_idx - 1, j_idx - 1],
            DP[:, :, i_idx - 1, j_idx],
            DP[:, :, i_idx, j_idx - 1]
        ], dim=1)
        move[:, 0, 0] -= self.penalty[:, None]
        move[:, 0, 2] -= self.penalty[:, None]
        move[:, 1, 1] -= self.penalty[:, None]
        move[:, 1, 2] -= self.penalty[:, None]
        move[:, 2, 0] -= self.penalty[:, None]
        move[:, 2, 1] -= self.penalty[:, None]
        DP[:, :, i_idx, j_idx] = move.max(dim=2).values + M[:, i_idx - 1, j_idx - 1][:, None, :]
    
    def backtrack(self, DP, M, L1, L2, **kwargs):
        penalty = kwargs.get('penalty', self.penalty)
        state = torch.argmax(DP[:, L1, L2]).item()  # 0: M, 1: V, 2: H
        path, i, j = [], L1, L2
        while i and j:
            path.append((i - 1, j - 1))
            if state == 0:
                cand = DP[:, i - 1, j - 1]
                cand[1] -= penalty
                cand[2] -= penalty
                i, j = i - 1, j - 1
            elif state == 1: 
                cand = DP[:, i - 1, j]
                cand[0] -= penalty
                cand[2] -= penalty
                i = i - 1
            else:
                cand = DP[:, i, j - 1]
                cand[0] -= penalty
                cand[1] -= penalty
                j = j - 1
            state = torch.argmax(cand).item()
        while i:
            path.append((i - 1, j))
            i = i - 1
        while j:
            path.append((i, j - 1))
            j = j - 1
        path.reverse()
        return DP[state, L1, L2].item(), path
    
    def backtrack_batch(self, DP, M, L1, L2, **kwargs):
        best_scores, paths = [], []
        for k in range(M.shape[0]):
            score, path = self.backtrack(DP[k], M[k], L1[k].item(), L2[k].item(), penalty=self.penalty[k].item())
            best_scores.append(score)
            paths.append(path)
        return best_scores, paths    