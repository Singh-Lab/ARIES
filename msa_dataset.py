from utils import *
import re
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

DATASET_DIRS = {
    'RV11': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV11", #<20% sequence identity
    'RV20': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV20", #families aligned with a highly divergent "orphan" sequence
    'RV30': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV30", #subgroups with <25% residue identity between groups
    'RV40': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV40", #sequences with N/C-terminal expansions
    'RV50': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV50", #internal insertions
    'RV12': "/scratch/gpfs/ia3026/plm-aln/datasets/balibase/bb3_release/RV12", #20-40% sequence identity
    'HOMSTRAD': "/scratch/gpfs/ia3026/plm-aln/datasets/homstrad", 
    'QUANTEST': "/scratch/gpfs/ia3026/plm-aln/datasets/QuanTest2", 
    'QUANTEST20': ("/scratch/gpfs/ia3026/plm-aln/datasets/QuanTest2", "/scratch/gpfs/ia3026/plm-aln/datasets/QuanTest2_20seqs"),
    'QUANTEST1000': ("/scratch/gpfs/ia3026/plm-aln/datasets/QuanTest2", "/scratch/gpfs/ia3026/plm-aln/datasets/QuanTest2_full")
}

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
        msa_name, msa, ungapped = self.entries[idx]
        return msa_name, msa, ungapped

class BalibaseDataset(MSADataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022):
        super(BalibaseDataset, self).__init__(msa_dir, min_len, max_len)
    
    def load_dataset(self):
        entries = []
        for fname in os.listdir(self.msa_dir):
            if not re.match(r"BB\d{5}\.msf$", fname):  # Skip truncated or malformed
                continue
            base = fname[:-4]  # BBnnnnn
            msf_path = os.path.join(self.msa_dir, f"{base}.msf")
            tfa_path = os.path.join(self.msa_dir, f"{base}.tfa")
            if os.path.exists(msf_path) and os.path.exists(tfa_path):
                msa = self.load_msa(msf_path)
                if msa:
                    ungapped = [s.replace(".", "").replace("-", "").upper() for s in msa.values()]
                    lengths = [len(s) for s in ungapped]
                    if (np.min(lengths) >= self.min_len) and (np.max(lengths) <= self.max_len):
                        entries.append((base, msa, ungapped))
        return entries

    def load_msa(self, msa_path):
        msa = {}
        try:
            lines = [l.strip() for l in open(msa_path, "r").readlines()]
            if "//" not in lines:
                return None
            lines = [l for l in lines[lines.index('//') + 1:] if l != ""]
            for l in lines:
                tokens = l.split()
                if not tokens:
                    continue
                name = tokens[0]
                seq = ''.join(tokens[1:])
                msa.setdefault(name, "")
                msa[name] += seq
            return msa if msa else None
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
            return None

class HomstradDataset(MSADataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022, msa_ext=".aln", prefix=">P1;"):
        self.msa_ext = msa_ext
        self.prefix = prefix
        super(HomstradDataset, self).__init__(msa_dir, min_len, max_len)
        
    def load_dataset(self):
        entries = []
        for fname in os.listdir(self.msa_dir):
            msa_name = fname.split('.')[0]
            if not fname.endswith(self.msa_ext):
                continue
            msa = self.load_msa(os.path.join(self.msa_dir, fname))
            if msa:
                msa = {k: s.upper() for k, s in msa.items()}
                ungapped = [s.replace(".", "").replace("-", "") for s in msa.values()]
                lengths = [len(s) for s in ungapped]
                if (np.min(lengths) >= self.min_len) and (np.max(lengths) <= self.max_len):
                    entries.append((msa_name, msa, ungapped))
        return entries

    def load_msa(self, msa_path):
        msa = {}
        try:
            lines = [l.strip() for l in  open(msa_path, "r").readlines()]
            current_id = None
            for line in lines:
                if line.startswith(self.prefix):
                    current_id = line[4:]
                    msa[current_id] = ""
                elif current_id is not None:
                    msa[current_id] += line.replace("*", "")  # remove trailing *
            return msa if msa else None
        except Exception as e:
            print(f"Failed to parse {msa_path}: {e}")
            return None

class QuanTestRefDataset(HomstradDataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022, msa_ext=".ref", prefix=">seq"):
        super(QuanTestRefDataset, self).__init__(msa_dir, min_len, max_len, msa_ext, prefix)

class QuanTestDataset(MSADataset):
    def __init__(self, msa_dir, min_len=0, max_len=1022, msa_ext=".vie.20seqs.fasta", prefix=">seq"):
        ref_dir, full_dir = msa_dir
        ref_dataset = QuanTestRefDataset(ref_dir, min_len, max_len, msa_ext=".ref", prefix=">seq")
        full_dataset = QuanTestRefDataset(full_dir, min_len, max_len, msa_ext=msa_ext, prefix=">seq")
        ref_entries = {
            msa_name: msa for msa_name, msa, _ in ref_dataset.entries
        }
        full_entries = {
            msa_name: msa for msa_name, msa, _ in full_dataset.entries
        }
        self.entries = []
        for msa_name in ref_entries.keys():
            self.entries.append((msa_name, ref_entries[msa_name], list(full_entries[msa_name].values())))

def get_dataset(dataset_name):
    if dataset_name == 'HOMSTRAD':
        return DataLoader(HomstradDataset(DATASET_DIRS[dataset_name]), batch_size=1, collate_fn = lambda b: b)
    elif dataset_name == 'QUANTEST':
        return DataLoader(QuanTestRefDataset(DATASET_DIRS[dataset_name]), batch_size=1, collate_fn = lambda b: b)
    elif dataset_name == 'QUANTEST20':
        return DataLoader(QuanTestDataset(DATASET_DIRS[dataset_name], msa_ext=".vie.20seqs.fasta"), batch_size=1, collate_fn = lambda b: b)
    elif dataset_name == 'QUANTEST1000':
        return DataLoader(QuanTestDataset(DATASET_DIRS[dataset_name], msa_ext=".vie"), batch_size=1, collate_fn = lambda b: b)
    else:
        return DataLoader(BalibaseDataset(DATASET_DIRS[dataset_name]), batch_size=1, collate_fn = lambda b: b)