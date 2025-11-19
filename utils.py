# set up directories
import os
os.environ['HF_HOME'] = '/scratch/gpfs/qh1002/.cache/huggingface'
temp_dir =  f'/home/qh1002/Code/recap/temp'
os.makedirs(temp_dir, exist_ok=True)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torch.nn.utils.rnn import pad_sequence
# Analytics
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from itertools import combinations, permutations, product
# System utilities
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from itertools import product, combinations
from tqdm import tqdm, trange
from copy import deepcopy
from pprint import pprint
from functools import partial
from io import StringIO
from datetime import datetime
from collections import Counter
import random, gc, re, sys, requests, glob, time, math, subprocess
# BioPython
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Levenshtein import distance as edit_distance
from Bio.Align import substitution_matrices

# PyTorch utilities
def assess_cuda_memory(header):
    print(f'{header}-------------------------------')
    mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2
    mem_cache = torch.cuda.memory_reserved() / 1024 ** 2
    print(f'Allocated: {mem_alloc:.2f}MB, Cache: {mem_cache:.2f}')

def freeze_module(module, unfreeze=False):
    for param in module.parameters():
        param.requires_grad = unfreeze
def set_seed(seed=2603):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def num_params(net):
    return sum(p.numel() for p in net.parameters())

def spawn_fasta(seqs, fasta_path):
    records = [SeqRecord(Seq(s if len(s) > 0 else '-' * maxlen), id=f'seq{i}', description="") for i, s in enumerate(seqs)]
    with open(fasta_path, 'w') as output_handle:
        SeqIO.write(records, output_handle, "fasta")