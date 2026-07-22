import argparse
import os
from typing import List

from msa_dataset import get_dataset, prepare_ref_eval, _resolve_ref_dir
from msa_tools import evaluate_msa
from scoring import load_core_domain, project_to_domain
from plm_wrapper import PLMWrapper
from transformers.utils import logging as hf_logging
from aries import ARIES
from clustal import MSAClustalO, MSAClustalW
from utils import set_seed


def resolve_device(device):
    """Resolve the requested device to one that is actually available.

    'auto' picks the best available backend (cuda > mps > cpu). An explicit
    device is honored when available, otherwise we warn and fall back to cpu.
    """
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if device == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not cuda_ok:
        print("Warning: CUDA not available, falling back to CPU.")
        return "cpu"
    if device.startswith("mps") and not mps_ok:
        print("Warning: MPS not available, falling back to CPU.")
        return "cpu"
    return device
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import torch


def write_fasta(alns: List[str], ids: List[str], out_path: str):
    records = [SeqRecord(Seq(alns[i]), id=ids[i], description="") for i in range(len(ids))]
    with open(out_path, "w") as handle:
        SeqIO.write(records, handle, "fasta")


def parse_args():
    desc = (
        "Run ARIES on a dataset or a folder of FASTA files. "
        "Writes ARIES alignments to --output-dir, and optionally runs "
        "ClustalO/ClustalW comparisons and scoring against references."
    )
    epi = (
        "Examples:\n"
        "  python run_aries.py --input ./datasets/BAliBASE/inputs --output-dir ./tmp/aries_out\n"
        "  python run_aries.py --input ./datasets/HOMSTRAD/inputs --output-dir ./tmp/aries_out --compare clustalo\n"
        "  python run_aries.py --input ./datasets/QuanTest2/inputs --ref-dir ./datasets/QuanTest2/reference_outputs "
        "--output-dir ./tmp/aries_out\n"
        "  python run_aries.py --input /path/to/fastas --output-dir ./tmp/aries_out --plm esm2-150M --batch 8\n"
    )
    p = argparse.ArgumentParser(
        description=desc,
        epilog=epi,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Path to a folder of input FASTA (.fasta) files (e.g. ./datasets/HOMSTRAD/inputs). "
            "Each .fasta is aligned independently."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to write ARIES alignments (FASTA). Created if missing.",
    )
    p.add_argument(
        "--ref-dir",
        default=None,
        help=(
            "Optional reference alignment directory. If provided, scoring is enabled. "
            "Reference files must be FASTA (.aln/.fasta/.fa) and match input file stems. "
            "If omitted and --input ends in 'inputs' with a sibling 'reference_outputs' "
            "folder, references are auto-detected and scoring is enabled."
        ),
    )
    p.add_argument(
        "--xml-dir",
        default=None,
        help=(
            "Path to BAliBASE bb3_release directory. If provided, SP/TC scores are computed "
            "against core-block regions only (using XML colsco-data masks)."
        ),
    )
    p.add_argument(
        "--compare",
        nargs="*",
        choices=["clustalo", "clustalw"],
        default=[],
        help="Run comparison aligners in addition to ARIES (no output files written for them).",
    )

    # ARIES config
    p.add_argument("--plm", default="esm2-650M", help="PLM name (ex: esm2-35M, esm2-150M, esm2-650M, protbert, prottrans, or prottrans-half, or huggingface).")
    p.add_argument("--num-hidden-states", type=int, default=9, help="Number of hidden states to concat.")
    p.add_argument("-w", "--window", type=int, default=5, help="Context window size for similarity.")
    p.add_argument("-r", "--reciprocal", type=float, default=200.0, help="Reciprocal weighting for similarity.")
    p.add_argument("--batch", type=int, default=32, help="PLM batch size.")
    p.add_argument("--blur", type=float, default=3.0, help="Gaussian blur sigma for similarity.")
    p.add_argument(
        "--pad-char",
        default="X",
        help=(
            "Padding character (default: X). "
            "Pass '!' to use the tokenizer's native pad token."
        ),
    )

    def _parse_medoid_topk(value: str):
        v = value.strip().lower()
        if v == "log":
            return "log"
        if v in ("ln", "logn"):
            return "ln"
        try:
            k = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "medoid-topk must be 'log', 'ln', 'logn', or an integer"
            ) from exc
        if k <= 0:
            raise argparse.ArgumentTypeError("medoid-topk integer must be > 0")
        return k

    p.add_argument(
        "--medoid-topk",
        type=_parse_medoid_topk,
        default="ln",
        help=(
            "Medoid top-k selection for template synthesis: 'log' (ceil(log2(n))), 'ln' (ceil(ln(n))), "
            "or a positive integer k."
        ),
    )
    p.add_argument("--maxlen", type=int, default=1022, help="Max sequence length to include from dataset. Any alignments including sequenes longer than this will be skipped. Sequences longer than 1022 will be processed via PLM tiling.")

    # runtime
    p.add_argument("--device", default="auto", help="Device for PLM/ARIES: 'auto' (cuda > mps > cpu), or an explicit 'cuda', 'mps', or 'cpu'.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    return p.parse_args()


def aries(
    input_path,
    output_dir,
    ref_dir=None,
    xml_dir=None,
    compare=None,
    plm="esm2-650M",
    num_hidden_states=9,
    window=5,
    reciprocal=200.0,
    batch=32,
    blur=3.0,
    pad_char="X",
    medoid_topk="ln",
    device="auto",
    seed=123,
    maxlen=1022,
):
    os.makedirs(output_dir, exist_ok=True)

    hf_logging.set_verbosity_error()
    set_seed(seed)

    compare = compare or []
    if maxlen > 1022:
        print("Warning: --maxlen > 1022: sequences above 1022 residues will be processed via PLM tiling.")
    device = resolve_device(device)
    try:
        plm = PLMWrapper(plm).to(device)
    except RuntimeError as e:
        if device != "cpu":
            print(f"Warning: failed to place model on '{device}' ({e}); falling back to CPU.")
            device = "cpu"
            plm = PLMWrapper(plm).to(device)
        else:
            raise
    config = {
        "plm": plm,
        "aligner": "aries",
        "num_hidden_states": num_hidden_states,
        "window": window,
        "reciprocal": reciprocal,
        "batch": batch,
        "blur": blur,
        "pad_char": pad_char,
        "medoid_mode": "dnd",
        "medoid_topk": medoid_topk,
        "sim_metric": "l2-gm",
    }

    resolved_ref_dir = ref_dir if ref_dir is not None else _resolve_ref_dir(input_path)
    include_refs = resolved_ref_dir is not None
    resolved_xml_dir = xml_dir
    if resolved_xml_dir is None and os.path.basename(os.path.normpath(input_path)) == "inputs":
        cand = os.path.join(os.path.dirname(os.path.normpath(input_path)), "xml")
        resolved_xml_dir = cand if os.path.isdir(cand) else None
    loader = get_dataset(input_path, include_refs=include_refs, ref_dir=resolved_ref_dir, max_len=maxlen)

    clustalo_aligner = MSAClustalO() if "clustalo" in compare else None
    clustalw_aligner = MSAClustalW() if "clustalw" in compare else None
    aries_aligner = ARIES(**config)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            item = batch[0]
            msa_name = item[0]
            msa = item[1]
            ungapped = item[2]
            ref_msa = item[3] if len(item) > 3 else None
            ref_indices = item[4] if len(item) > 4 else None
            ref_mask = item[5] if len(item) > 5 else None

            ids = list(msa.keys())
            print("-" * 90)
            print(f"[{i+1}/{len(loader)}]: Aligning {msa_name} with {len(msa)} seqs")

            clustalo_aln = None
            clustalw_aln = None
            if clustalo_aligner is not None:
                clustalo_aln = clustalo_aligner.align(seqs=ungapped, aln_name=msa_name)
            if clustalw_aligner is not None:
                clustalw_aln = clustalw_aligner.align(seqs=ungapped, aln_name=msa_name)

            aries_aln, runtime = aries_aligner.align(seqs=ungapped, msa_name=msa_name)

            out_path = os.path.join(output_dir, f"{msa_name}.fasta")
            write_fasta(aries_aln, ids, out_path)

            if resolved_ref_dir is not None and ref_msa:
                if resolved_xml_dir is not None:
                    xml_path = os.path.join(resolved_xml_dir, f"RV{msa_name[2:4]}", f"{msa_name}.xml")
                    loaded = load_core_domain(xml_path, ids)
                else:
                    loaded = None

                if loaded is not None:
                    domain_targets, domain_ref, idx_sets = loaded
                    def _score(aln):
                        proj = project_to_domain(aln, idx_sets, domain_targets)
                        return evaluate_msa(proj, domain_ref) if proj is not None else (float('nan'), float('nan'))
                    aries_sp, aries_tc = _score(aries_aln)
                    parts = [f"ARIES SP={aries_sp:.3f} TC={aries_tc:.3f}"]
                    if clustalo_aln is not None:
                        sp, tc = _score(clustalo_aln)
                        parts.append(f"ClustalO SP={sp:.3f} TC={tc:.3f}")
                    if clustalw_aln is not None:
                        sp, tc = _score(clustalw_aln)
                        parts.append(f"ClustalW SP={sp:.3f} TC={tc:.3f}")
                else:
                    aries_eval = prepare_ref_eval(aries_aln, ref_indices, ref_mask)
                    aries_sp, aries_tc = evaluate_msa(aries_eval, ref_msa)
                    parts = [f"ARIES SP={aries_sp:.3f} TC={aries_tc:.3f}"]
                    if clustalo_aln is not None:
                        clustalo_eval = prepare_ref_eval(clustalo_aln, ref_indices, ref_mask)
                        clustalo_sp, clustalo_tc = evaluate_msa(clustalo_eval, ref_msa)
                        parts.append(f"ClustalO SP={clustalo_sp:.3f} TC={clustalo_tc:.3f}")
                    if clustalw_aln is not None:
                        clustalw_eval = prepare_ref_eval(clustalw_aln, ref_indices, ref_mask)
                        clustalw_sp, clustalw_tc = evaluate_msa(clustalw_eval, ref_msa)
                        parts.append(f"ClustalW SP={clustalw_sp:.3f} TC={clustalw_tc:.3f}")
                print(" | ".join(parts))
            elif resolved_ref_dir is not None:
                print("No reference alignment found; skipping evaluation.")


def main():
    args = parse_args()
    aries(
        input_path=args.input,
        output_dir=args.output_dir,
        ref_dir=args.ref_dir,
        xml_dir=args.xml_dir,
        compare=args.compare,
        plm=args.plm,
        num_hidden_states=args.num_hidden_states,
        window=args.window,
        reciprocal=args.reciprocal,
        batch=args.batch,
        blur=args.blur,
        pad_char=args.pad_char,
        medoid_topk=args.medoid_topk,
        device=args.device,
        seed=args.seed,
        maxlen=args.maxlen,
    )


if __name__ == "__main__":
    main()
