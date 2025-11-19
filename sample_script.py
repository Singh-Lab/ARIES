from msa_dataset import *
from msa_tools import *
from plm_wrapper import *
from aries import *
from clustal import *

def evaluate_msa(alns, refs):        
    sp_score = multiple_sp_score(refs, alns, normalize=True)
    tc_score = multiple_tc_score(refs, alns, normalize=True)
    return sp_score, tc_score

set_seed(123)

config = {
    'plm': PLMWrapper('esm2-650M').to('cuda'),
    'aligner': 'dtw',
    'num_hidden_states': 9,
    'window': 5,
    'reciprocal': 200,
    'batch': 32
}
dataset = 'RV11'

clustalo_aligner = MSAClustalO()
clustalw_aligner = MSAClustalW()
aries_aligner = ARIES(**config)
loader = get_dataset(dataset)
with torch.no_grad():
    for i, batch in enumerate(loader):
        msa_name, msa, ungapped = batch[0]
        print(f'-----------------------------------------------------------------------------------------')
        print(f'[{i+1}/{len(loader)}]: Aligning {msa_name} with {len(msa)} seqs')
        
        clustalo_aln = clustalo_aligner.align(seqs=ungapped, aln_name=msa_name)
        clustalw_aln = clustalw_aligner.align(seqs=ungapped, aln_name=msa_name)
        aries_aln, runtime = aries_aligner.align(seqs=ungapped, msa_name=msa_name)        
        
        clustalo_sp, clustalo_tc = evaluate_msa(clustalo_aln, list(msa.values()))
        clustalw_sp, clustalw_tc = evaluate_msa(clustalo_aln, list(msa.values()))
        aries_sp, aries_tc = evaluate_msa(clustalo_aln, list(msa.values()))
        print(f'SP Score --- ClustalO: {clustalo_sp:.3f} ClustalW: {clustalw_sp:.3f} ARIES: {aries_sp:.3f}')
        print(f'SP Score --- ClustalO: {clustalo_tc:.3f} ClustalW: {clustalw_tc:.3f} ARIES: {aries_tc:.3f}')
