# NEXUS v2 adaptation pilot

An exploratory, single-seed comparison of **Flexible-TADA versus LoRA inside a
sentiment classifier with the original scalar-knowledge fusion design**.
This repository contains experiment code, not completed benchmark results.

[Open in Colab](https://colab.research.google.com/github/sepehrmustafavi/nexusv2/blob/main/notebooks/NEXUS_Pilot_Colab.ipynb)

## Experiment protocol

One base model: `Qwen/Qwen2-0.5B` (base, not Instruct). One dataset:
`stanfordnlp/sst2`. One seed: 42. Four arms, with two backbone learning rates per
arm: **eight training runs total**.

| Method | Knowledge | Backbone learning rates |
|---|---|---|
| Flexible-TADA | Off | 2e-5, 1e-4 |
| Flexible-TADA | On | 2e-5, 1e-4 |
| LoRA | Off | 1e-4, 4e-4 |
| LoRA | On | 1e-4, 4e-4 |

- Flexible-TADA trains input token embeddings, the complete final transformer
  block, and a classification head. Final RMSNorm stays frozen.
- LoRA trains rank-8 updates on `q_proj`/`v_proj` in every block, plus the same
  classification head. Alpha 16; LoRA dropout 0.1; no quantization.
- Pooling uses the **last non-padding token**. There is no decoder CLS token.
- Both arms have an identical `(hidden_size + 1) -> 2` linear head, initialized
  identically. Knowledge-off supplies a zero scalar. Knowledge-on supplies the
  mean polarity over unique, longest-matching concepts (up to five words).
- **The fixed knowledge resource is SenticNet 6 (200,000 concepts), distributed
  in the original `senticnet` 1.6 package. This package does NOT contain
  SenticNet 7.** The lexicon hash and size are recorded. This deliberate,
  reproducible pilot is not an exact replication of the old manuscript, which
  names SenticNet 7 and does not fully specify its keyword extractor.
- No new gate, projection, graph traversal, negation correction, or intrinsic
  explainability mechanism is introduced. Knowledge sees only the text visible
  after model truncation, never the truncated-away suffix.
- This is **supervised task tuning**. It does not reproduce original TADA's
  unsupervised domain-adaptive pretraining followed by task fine-tuning.
- Maximum length 128; four epochs; microbatch 2; accumulation 16 (effective batch
  32); head LR 1e-3 for all candidates; AdamW decay .01; warmup 10%; clip 1.0.
- Weights and optimizer states stay FP32. T4 uses FP16 autocast and GradScaler;
  non-reentrant gradient checkpointing reduces activation memory. There is no
  incorrect `no_grad` around intermediate frozen layers.

## Selection and reporting

5% of normalized sentence groups from the original training split are held out
for internal tuning. Exact normalized matches to official validation sentences
are excluded from training/tuning. Duplicate normalized sentences stay together.
SST-2 also contains related phrases: this procedure does **not** eliminate every
source-review/phrase relation. It is a pilot split, not a leakage-proof new test.

The best epoch and learning rate for each arm are selected by **internal-tune
Macro-F1 only**. The official validation set (872 items) is used for reporting
only after all eight training runs are complete and `--finalize` is requested.
It is not an untouched test set for a future paper. Do not tune the design on
this set and later present it as independent confirmation.

Model and dataset commit SHAs are resolved once and stored in `protocol.json`.
Prepared input IDs, labels, concept lists, split IDs and feature values are
persisted in `data.json`. Code/data hashes and package versions guard against
accidentally resuming a different experiment in the same output directory.

Compare both on-knowledge arms, but also examine the on-minus-off delta within
each method. Separately selected learning rates make these **descriptive
best-of-budget deltas**, not an identified causal interaction. A small gap on
one seed is inconclusive. Follow up with more seeds and harder tasks before
making a paper-level claim. Never interpret these runs as XAI validation.

## Run on free Colab

1. Open the notebook using the link above, and sign in to your Google account.
2. Choose **Runtime > Change runtime type > T4 GPU > Save**. The available GPU
   depends on your free-tier quota.
3. Run cells from top to bottom. Authorize the Google Drive mount when prompted.
4. The notebook installs pinned dependencies, performs offline smoke tests,
   checks the GPU, and starts the experiment queue.
5. The default session budget is **two hours**. The program checkpoints and
   returns when that budget is reached. Repeat the training cell (or reconnect,
   mount Drive and run all cells again). Finished candidates are skipped.
6. When it reports **Completed 8/8**, run the finalization cell. Earlier attempts
   are intentionally refused. The last cell displays and downloads the report.

Default output: `/content/drive/MyDrive/nexus-v2-pilot/qwen2_sst2_seed42_v1`.
You do not need a Hugging Face or GitHub token for this public model and repo.
Do not put credentials into a notebook or commit them to GitHub.

## Resume and storage

### Running from the delivered ZIP before GitHub publication

Extract `nexus-v2-pilot.zip` on your computer. In Colab choose **File > Upload
notebook** and select `nexus-v2/notebooks/NEXUS_Pilot_Colab.ipynb`. In its source
setup cell set `USE_LOCAL_ZIP = True`. Run the cells; when asked, upload the
**original complete ZIP**, not just the notebook. The code extracts under
`/content/nexus-v2` and records its source revision in Drive. Other instructions
are unchanged. Keep that same ZIP for later reconnections. GitHub clone mode
requires that the public repository actually exists; this local-ZIP mode does
not require GitHub authentication or publication.

Resumable snapshots are written every 100 optimizer updates and at epoch ends.
A soft two-hour stop also writes a snapshot. A sudden Colab termination can
lose work since the latest complete snapshot. Keeping a tab open does not
guarantee GPU access or uninterrupted execution on the free tier.

Snapshots include trainable weights only, optimizer, scheduler, AMP scaler,
RNGs, epoch and next-example offset. Each file has a SHA-256 completion marker;
the latest two snapshots are retained so a partial/corrupt newest file can fall
back to the previous complete one. Best-epoch weights are saved separately.
After candidate completion its resumable optimizer snapshots are removed and
only selected best weights remain. Reconnect and rerun; there is **no automatic
reacquisition of a terminated Colab VM** and no keep-alive workaround.

Flexible-TADA updates a large vocabulary embedding matrix, so its checkpoints
are much larger than LoRA's. Keep approximately **10 GB of free Drive space**
for this pilot; actual usage depends on retained snapshots and partial files.
Do not run two Colab sessions against the same output directory. If you change
code or settings, use a **new output directory**. To resume, keep the original
code checkout and exact settings.

Resume is tested for numerical equivalence on CPU. Across GPU types or software
changes, bitwise equivalence is not promised. Do not load `.pt` checkpoints from
untrusted sources: they contain Python optimizer/RNG state.

## Command line

```bash
python -m pip install -r requirements.txt
python -m nexus_pilot.run --output /path/to/pilot --list
python -m nexus_pilot.run --output /path/to/pilot --max-hours 2
# Repeat the previous command until 8/8 are complete.
python -m nexus_pilot.run --output /path/to/pilot --finalize
```

Use `--only EXACT_RUN_ID` to process a particular candidate. `--prepare-only`
downloads and freezes the protocol/data without training. For debugging only,
`--train-limit 256 --epochs 1` creates a smaller experiment; use a distinct output
directory and **never mix those results with the full pilot**.

If CUDA runs out of memory, reduce microbatch to 1 and increase accumulation
to 32, use a new output directory, and run **all arms** under that protocol.
The code does not silently change settings after an OOM. If package imports
fail after installation, restart the Colab runtime and rerun cells. Resume
requires matching scientific package versions; use a new output for new ones.

## Output files

| File | Purpose |
|---|---|
| `protocol.json` | Frozen settings, source hashes, revisions, versions and coverage |
| `data.json` | Prepared rows with split IDs, token IDs, concepts and polarity |
| `<run>/history.json` | Internal tuning metrics by epoch |
| `<run>/complete.json` | Completion, selected epoch, parameters, elapsed time and peak VRAM |
| `<run>/best-epoch-N.pt` | Selected trainable weights, reload with the frozen base revision |
| `<selected-run>/predictions.csv` | Per-example official-validation predictions |
| `all_candidates.json` | Internal-tuning metrics of all eight candidates |
| `summary.csv`, `summary.json` | Four selected-arm results |
| `effects.json` | Descriptive within-method knowledge deltas in percentage points |

Time includes training, validation and checkpoint I/O and is accumulated across
resumes; it excludes model loading and final reporting. Interrupted work after a
snapshot is not fully accounted for. Peak VRAM means PyTorch allocated memory,
not the entire GPU process footprint. These measurements are operational pilot
diagnostics, not a controlled hardware-efficiency benchmark.

## Tests and limitations

```bash
python -m unittest discover -s tests -v
```

Tests use an explicitly random, tiny Qwen2 model. They cover last-token pooling,
freeze boundaries and backward flow, zero-feature behavior, concept matching,
group split isolation, checkpoint corruption fallback, interrupted/resumed
training equivalence, and the full eight-candidate/four-report pipeline. They
do not download or benchmark Qwen2-0.5B. See `VALIDATION.md` for the validation
performed before this repository was delivered.

## Research resources

- Qwen2-0.5B: https://huggingface.co/Qwen/Qwen2-0.5B
- SST-2: https://huggingface.co/datasets/stanfordnlp/sst2
- LoRA: https://openreview.net/forum?id=nZeVKeeFYf9
- Original TADA: https://aclanthology.org/2023.findings-acl.31/
- SenticNet resources and applicable terms: https://sentic.net/downloads/

External models, datasets and knowledge resources retain their respective
licenses. This repository does not redistribute those resources or the user's
manuscripts. There are no precomputed or invented benchmark results here.
