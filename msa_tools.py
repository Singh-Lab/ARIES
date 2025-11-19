from utils import *
from collections import defaultdict, Counter
from Bio import Phylo
from io import StringIO

def sequence_identity(s1, s2, ignore_gaps=True):
    assert len(s1) == len(s2), 'sequences must align'
    matches, total = 0, 0
    for a, b in zip(s1, s2):
        if ignore_gaps and ('-' in (a, b)):
            continue
        total += 1
        matches += (a==b)
    return matches / total if total > 0 else 0.0

def parse_clustal_aln(aln_path):
    sequences = {}
    with open(aln_path, "r") as f:
        lines = f.readlines()
    # Skip header (first line) and blank line after
    lines = [line.rstrip("\n") for line in lines if line.strip() and not line.startswith("CLUSTAL")]
    for line in lines:
        # Skip consensus lines (start with space)
        if line.startswith(" "):
            continue
        parts = line.split()
        if len(parts) >= 2:
            name, seq_chunk = parts[0], parts[1]
            if name not in sequences:
                sequences[name] = ""
            sequences[name] += seq_chunk
    return sequences

def write_clustal_aln(records, aln_path, header="", line_width=60, name_width=20):
    if not records:
        raise ValueError("No sequences provided.")
    L = len(records[0][1])
    if len({len(seq) for _, seq in records}) != 1:
        raise ValueError("All sequences must have the same aligned length.")
    with open(aln_path, "w") as out:
        out.write(f"{header} multiple sequence alignment\n\n")
        for start in range(0, L, line_width):
            end = min(L, start + line_width)
            for name, seq in records:
                chunk = seq[start:end]
                label = name[:name_width].ljust(name_width)
                out.write(f"{label} {chunk}\n")
            out.write("\n") 

def pairwise_sp_score(ref1, ref2, pred1, pred2, normalize=True):
    assert (ref1.replace('-', '').replace('.', '') == pred1.replace('-', '').replace('.', '')), 'mismatched ungapped sequences' 
    assert (ref2.replace('-', '').replace('.', '') == pred2.replace('-', '').replace('.', '')), 'mismatched ungapped sequences' 
    gap_chars = ['-', '.']
    def extract_aligned_pairs(s1, s2):
        assert len(s1) == len(s2), 'mismatched aligned lengths'
        c1, c2 = -1, -1
        pairs = set()
        for i in range(len(s1)):
            if s1[i] in gap_chars and s2[i] in gap_chars:
                continue
            if s1[i] in gap_chars:
                c2 += 1
            elif s2[i] in gap_chars:
                c1 += 1
            else:
                c1 += 1
                c2 += 1
                pairs.add((c1, c2))
        return pairs
    ref_pairs = extract_aligned_pairs(ref1, ref2)
    pred_pairs = extract_aligned_pairs(pred1, pred2)
    correct_pairs = ref_pairs & pred_pairs
    return len(correct_pairs) / (len(ref_pairs) if normalize else 1), ref_pairs, correct_pairs

def multiple_sp_score(refs, preds, normalize=True):
    num_ref_pairs, num_correct_pairs = 0, 0
    for i in range(len(refs) - 1):
        for j in range(i + 1, len(refs)):
            _, ref_pairs, correct_pairs = pairwise_sp_score(refs[i], refs[j], preds[i], preds[j])
            num_ref_pairs += len(ref_pairs)
            num_correct_pairs += len(correct_pairs)
    return num_correct_pairs / (num_ref_pairs if normalize else 1)

def reorder_by_sequences(refs, preds, gap_chars=('-', '.')):
    def ungap(s):
        for g in gap_chars:
            s = s.replace(g, '')
        return s

    ref_ungapped = [ungap(s) for s in refs]
    pred_ungapped = [ungap(s) for s in preds]
    # Multiset equality check (same sequences, same multiplicities)
    assert Counter(ref_ungapped) == Counter(pred_ungapped), \
        "refs/preds don't contain the same set of ungapped sequences"
    # Map ungapped -> queue of indices in preds
    idx_map = defaultdict(list)
    for j, u in enumerate(pred_ungapped):
        idx_map[u].append(j)
    # Reorder by popping from each queue in ref order
    reordered = []
    for u in ref_ungapped:
        j = idx_map[u].pop(0)
        reordered.append(preds[j])
    return reordered

def _msa_to_column_multiset(msa, gap_chars=('-', '.'), min_residues=2):
    # min_residues=0 includes all columns, min_residues=1 includes single-res columns, min_residues=2 includes only multi-res cols
    n_seq = len(msa)
    L = len(msa[0])
    assert all(len(s) == L for s in msa), "all sequences in an MSA must have equal aligned length"

    counters = [-1] * n_seq  # per-sequence ungapped residue counters
    sigs = []
    for col in range(L):
        sig = []
        non_gap = 0
        for s_idx in range(n_seq):
            ch = msa[s_idx][col]
            if ch in gap_chars:
                sig.append(-1)
            else:
                counters[s_idx] += 1
                sig.append(counters[s_idx])
                non_gap += 1
        if non_gap >= min_residues:
            sigs.append(tuple(sig))
    return Counter(sigs)


def multiple_tc_score(refs, preds, normalize=True, gap_chars=('-', '.'), min_residues=2):
    preds = reorder_by_sequences(refs, preds[:len(refs)], gap_chars=gap_chars)
    ref_sig = _msa_to_column_multiset(refs, gap_chars=gap_chars, min_residues=min_residues)
    pred_sig = _msa_to_column_multiset(preds, gap_chars=gap_chars, min_residues=min_residues)
    num_correct = sum(min(ref_sig[sig], pred_sig.get(sig, 0)) for sig in ref_sig)
    num_ref_cols = sum(ref_sig.values())
    return num_correct / (num_ref_cols if normalize else 1)

def topk_medoids_from_dnd(dnd_path, k=1):
    tree = Phylo.read(dnd_path, "newick")
    adj = defaultdict(list)
    node_id = {}
    leaves = []
    all_nodes = list(tree.find_clades(order="level"))
    for i, node in enumerate(all_nodes):
        node_id[node] = i
        if not node.clades:  # leaf
            leaves.append(i)
    n = len(all_nodes)

    for node in all_nodes:
        u = node_id[node]
        for child in node.clades:
            v = node_id[child]
            w = child.branch_length or 0.0
            adj[u].append((v, w))
            adj[v].append((u, w))
    # ---- Arrays for DP ----
    size = [0] * n          # number of descendant nodes
    dist_sum = [0.0] * n    # total distance from this node to nodes in its subtree
    # ---- First DFS: postorder accumulation ----
    def dfs1(u, parent):
        size[u] = 1
        dist_sum[u] = 0.0
        for v, w in adj[u]:
            if v == parent:
                continue
            dfs1(v, u)
            size[u] += size[v]
            dist_sum[u] += dist_sum[v] + w * size[v]
    # ---- Second DFS: rerooting to propagate total distances ----
    def dfs2(u, parent):
        for v, w in adj[u]:
            if v == parent:
                continue
            dist_sum[v] = dist_sum[u] + w * (n - 2 * size[v])
            dfs2(v, u)
    # ---- Run the two DFS passes ----
    root = 0
    dfs1(root, -1)
    dfs2(root, -1)
    # ---- Collect leaf distances and pick top-k ----
    leaf_dists = [(all_nodes[i].name, dist_sum[i]) for i in leaves]
    leaf_dists.sort(key=lambda x: x[1])
    medoids = leaf_dists[:k]
    return [int(s[3:]) for s, d in medoids]

def reflect_paddings(seqs, w=0):
    return [s[1:w+1][::-1] + s + s[-w-1:-1][::-1] for s in seqs]

def polyX_paddings(seqs, w=0):
    polyX_padding = 'X' * w
    return [polyX_padding + s + polyX_padding for s in seqs]

def randomAA_paddings(seqs, w=0):
    pad_left = ''.join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(w))
    pad_right = ''.join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(w))
    return [pad_left + s + pad_right for s in seqs]

    