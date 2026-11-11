## Dataset Structure

Each folder in this repository corresponds to a dataset. Within each dataset directory:

- **`input/`** contains the **ungapped input FASTA files**.  
- **`reference_outputs/`** contains the **reference alignment (`.aln`) files** used as gold-standard alignments for evaluation.

---

## BAliBASE

The BAliBASE dataset follows the standard RV (Reference Version) naming scheme:

- Files are grouped by RV using the **first four characters** of the filename.
- For example:
  - `BB11*.fasta` or `BB11*.aln` --> **RV11**
  - `BB12*.fasta` or `BB12*.aln` --> **RV12**, etc.

---

## QuanTest2

The **QuanTest2** dataset has the following structure:

- The **`input/`** directory contains sets of **1000 sequences**:
  - The **first 3 sequences** are **HOMSTRAD reference sequences** with known structures.
  - The **remaining 997 sequences** come from **Pfam**.

- The **`reference_outputs/`** directory contains the **reference alignment** for the **first three HOMSTRAD sequences only**.
