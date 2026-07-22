from utils import *
from torch.utils.data import Dataset, DataLoader

class MSADataset(Dataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022):
        self.min_len = min_len
        self.max_len = max_len
        self.msa_dir = msa_dir
        self.entries = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class FastaFolderDataset(MSADataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022, exts=(".fasta",), ref_dir=None, include_refs=False):
        self.exts = exts
        self.ref_dir = ref_dir
        self.include_refs = include_refs
        self.dataset_tag = self._infer_dataset_tag(msa_dir, ref_dir)
        super().__init__(msa_dir, min_len, max_len)

    def load_dataset(self):
        entries = []
        for fname in os.listdir(self.msa_dir):
            if not fname.endswith(self.exts):
                continue

            msa_name = os.path.splitext(fname)[0]
            msa = self.load_fasta(os.path.join(self.msa_dir, fname))
            if not msa:
                continue

            msa = {k: s.upper() for k, s in msa.items()}
            ungapped = [s.replace(".", "").replace("-", "") for s in msa.values()]
            lengths = [len(s) for s in ungapped]
            max_observed_len = int(np.max(lengths))
            if (np.min(lengths) < self.min_len) or (max_observed_len > self.max_len):
                print(
                    f"Warning: Skipping {msa_name}: max sequence length in file is "
                    f"{max_observed_len}, which exceeds --maxlen={self.max_len}"
                )
                continue

            if not self.include_refs:
                entries.append((msa_name, msa, ungapped))
                continue

            ref_seqs, ref_ids = self.load_ref_msa(msa_name)
            ref_ordered = None
            ref_indices = None

            if ref_seqs:
                if self.dataset_tag and "QUANTEST" in self.dataset_tag.upper():
                    msa_ungapped = [s.replace("-", "").replace(".", "") for s in msa.values()]
                    used = set()
                    ref_indices = []
                    ref_ordered = []

                    for rs in ref_seqs:
                        rs_ungapped = rs.replace("-", "").replace(".", "")
                        match_idx = None
                        for i, mu in enumerate(msa_ungapped):
                            if i in used:
                                continue
                            if mu == rs_ungapped:
                                match_idx = i
                                break
                        if match_idx is None:
                            ref_indices = None
                            ref_ordered = None
                            break
                        used.add(match_idx)
                        ref_indices.append(match_idx)
                        ref_ordered.append(rs)
                else:
                    ref_map = dict(zip(ref_ids, ref_seqs))
                    ref_indices = [i for i, k in enumerate(msa.keys()) if k in ref_map]
                    ref_ordered = [ref_map[k] for k in msa.keys() if k in ref_map]
                    if not ref_ordered:
                        ref_ordered = None
                        ref_indices = None

            entries.append((msa_name, msa, ungapped, ref_ordered, ref_indices))
        return entries

    def load_fasta(self, fasta_path):
        msa = {}
        try:
            for r in SeqIO.parse(fasta_path, "fasta"):
                msa[r.id] = str(r.seq).replace("*", "")
            return msa if msa else None
        except Exception as e:
            print(f"Failed to parse {fasta_path}: {e}")
            return None

    def _infer_dataset_tag(self, msa_dir, ref_dir):
        for c in (msa_dir, ref_dir):
            if not c:
                continue
            name = os.path.basename(os.path.dirname(c)) if os.path.basename(c) == "inputs" else os.path.basename(c)
            if name:
                return name
        return None

    def _remove_all_gap_cols(self, seqs, gap_chars="-."):
        if not seqs:
            return seqs
        L = len(seqs[0])
        keep = [not all(s[i] in gap_chars for s in seqs) for i in range(L)]
        return [''.join(s[i] for i in range(L) if keep[i]) for s in seqs]

    def load_ref_msa(self, msa_name):
        if not self.ref_dir:
            return None, None

        for ext in (".aln", ".fasta", ".fa"):
            path = os.path.join(self.ref_dir, f"{msa_name}{ext}")
            if not os.path.exists(path):
                continue
            try:
                records = list(SeqIO.parse(path, "fasta"))
                if not records:
                    return None, None

                # strip '*' everywhere so refs match model outputs
                seqs = [str(r.seq).upper().replace("*", "") for r in records]
                ids = [r.id for r in records]

                if self.dataset_tag and "QUANTEST" in self.dataset_tag.upper():
                    seqs = self._remove_all_gap_cols(seqs[:3], gap_chars="-.")
                    ids = ids[:3]

                return seqs, ids
            except Exception as e:
                print(f"Failed to parse {path}: {e}")
                return None, None

        return None, None


def _resolve_dataset_dir(input_dir):
    if os.path.isdir(input_dir):
        return input_dir
    raise ValueError(
        f"Input folder not found: {input_dir}. Pass a path to a folder of .fasta "
        f"files (e.g. ./datasets/HOMSTRAD/inputs)."
    )


def _resolve_ref_dir(input_dir):
    # Auto-detect references when the input folder is named 'inputs' with a
    # sibling 'reference_outputs' (the layout of the bundled datasets).
    if os.path.basename(os.path.normpath(input_dir)) == "inputs":
        ref_dir = os.path.join(os.path.dirname(os.path.normpath(input_dir)), "reference_outputs")
        if os.path.isdir(ref_dir):
            return ref_dir
    return None


def get_dataset(dataset_name, include_refs=False, ref_dir=None, max_len=1022):
    msa_dir = _resolve_dataset_dir(dataset_name)
    if include_refs:
        ref_dir = ref_dir if ref_dir is not None else _resolve_ref_dir(dataset_name)
    else:
        ref_dir = None
    return DataLoader(
        FastaFolderDataset(msa_dir, ref_dir=ref_dir, include_refs=include_refs, max_len=max_len),
        batch_size=1,
        collate_fn=lambda b: b
    )


def prepare_ref_eval(pred_alns, ref_indices=None, ref_mask=None):
    eval_alns = pred_alns
    if ref_indices is not None:
        eval_alns = [eval_alns[i] for i in ref_indices]
    if ref_mask is not None:
        eval_alns = [''.join(s[i] for i, keep in enumerate(ref_mask) if keep) for s in eval_alns]
    return eval_alns