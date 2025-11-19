from utils import *
from msa_tools import *
from plm_wrapper import *
from scoring import *
from pairwise import *
from clustal import *

class ARIES:
    def __init__(self, **kwargs):
        self.plm = kwargs['plm']
        self.w = kwargs.get('window', 5)
        self.h = kwargs.get('num_hidden_states', 9)
        self.b = kwargs.get('batch', 1)
        self.r = kwargs.get('reciprocal', 200.)
        self.g = kwargs.get('blur', 3.)
        self.medoid_metric = kwargs.get('medoid_metric', 'edit')
        aligner_tokens = kwargs.get('aligner', 'dtw').split('-')
        if aligner_tokens[0] == 'dtw':
            self.aligner = DTW(batch=True)
        elif aligner_tokens[0] == 'turnpen':
            self.aligner = TurnPenaltyDTW(tau=float(aligner_tokens[1]), batch=True)
        
    def get_embeddings(self, seqs, return_logits=False):
        padded_seqs = polyX_paddings(seqs, self.w)
        embeddings, logits = self.plm(padded_seqs, num_hidden_states=self.h, batch=self.b)
        embeddings = [embeddings[i, :len(padded_seqs[i]), :].to('cpu') for i in range(len(padded_seqs))]
        if return_logits:
            logits = [logits[i, :len(padded_seqs[i]), :].to('cpu') for i in range(len(padded_seqs))]    
            return embeddings, logits
        return embeddings

    def build_msa(self, embeddings, seqs, consensus_embedding, verbose=True):
        if verbose:
            print(f'>>>> Aligning with template')
        scores, paths, pairwise_scores = self.aligner.align_embeddings(consensus_embedding, embeddings, self.w, self.r, self.g, return_scores=True)
        num_seqs = len(seqs)
        template_length = consensus_embedding.shape[0] - 2 * self.w
        edges = []
        for i in range(num_seqs):
            edges.append(torch.zeros(template_length, len(seqs[i]), dtype=int))
            for u, v in paths[i]:
                edges[i][u][v] = 1  # edge from template[u] to si[v]
        if verbose:
            print(f'>>>> Assigning columns')
        for i in range(len(edges)):
            for j in range(len(seqs[i])):
                template_residues = torch.where(edges[i][:, j])[0]
                best_match = template_residues[torch.argmax(pairwise_scores[i][template_residues, j])]                
                edges[i][template_residues, j] = 0
                edges[i][best_match, j] = 1
        
        if verbose:
            print(f'>>>> Inferring anchors')
        cols = []
        for t in range(template_length):
            col_left, col_right = [], []
            for i in range(num_seqs):
                edges_from_t = torch.where(edges[i][t] > 0)[0]
                if edges_from_t.numel() == 0:
                    col_left.append('')
                    col_right.append('')
                    continue
                best_match_idx = torch.argmax(pairwise_scores[i][t][edges_from_t])                
                left_idx = edges_from_t[:best_match_idx + 1].tolist()
                right_idx = edges_from_t[best_match_idx + 1:].tolist()
                col_left.append(''.join([seqs[i][j] for j in left_idx]))
                col_right.append(''.join([seqs[i][j] for j in right_idx]))
            left_width = max([len(s) for s in col_left])
            right_width = max([len(s) for s in col_right])
            if (left_width + right_width) == 0:
                continue
            else:
                cols.append((col_left, col_right))
        
        if verbose:
            print(f'>>>> Centering columns')
        for t, (col_left, col_right) in enumerate(cols):
            left_width = max([len(s) for s in col_left])
            right_width = max([len(s) for s in col_right])
            col = []
            for i in range(num_seqs):
                pad_left = (left_width - len(col_left[i])) * '-'
                pad_right = (right_width - len(col_right[i])) * '-'             
                col.append(pad_left + col_left[i] + col_right[i] + pad_right)
            cols[t] = col
        alns = [
            ''.join([c[i] for c in cols])
            for i in range(num_seqs)
        ]
        del cols
        return alns
    
    def align(self, seqs, **kwargs):
        print(f'Embed sequences with PLM')
        t = time.time()
        embeddings = self.get_embeddings(seqs)
        emb_time = time.time() - t
        print(f'Building template')
        t = time.time()        
        msa_name = kwargs['msa_name']
        path = f'/home/qh1002/Code/recap/temp/{msa_name}_clustalw.dnd'
        if not os.path.exists(path):
            run_clustalo(seqs, msa_name)
        k = math.ceil(math.log2(len(seqs)))    
        medoids = topk_medoids_from_dnd(path, k)
        medoid_embs = [embeddings[m] for m in medoids]
        medoid_seqs = [seqs[m] for m in medoids]
        global_medoid_emb = embeddings[medoids[0]]
        medoid_aln = self.build_msa(medoid_embs, medoid_seqs, global_medoid_emb)
        consensus = [s.replace('-', 'X') for s in medoid_aln]
        consensus_emb = torch.stack(self.get_embeddings(consensus), dim=0)
        mask = torch.tensor([[1 if c != '-' else 0 for c in s] for s in polyX_paddings(medoid_aln, self.w)]).to(consensus_emb.device)
        consensus_emb = (consensus_emb * mask[:, :, None]).sum(dim=0) / mask.sum(dim=0)[:, None]
        aln_time = time.time() - t
        print(f'Building MSA given template')
        return self.build_msa(embeddings, seqs, consensus_emb), (emb_time, aln_time)