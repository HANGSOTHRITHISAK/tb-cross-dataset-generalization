# M3A: full-data preflight

Date: 2026-10-13. M3A checks full-partition execution feasibility and continuation
correctness before a separately authorized M3B experiment. Its single-epoch
validation metrics are **preflight diagnostics**, not final research results.

## Approved contract

The primary baseline uses unweighted cross-entropy, the existing ImageNet
DenseNet-121 with three outputs, and the frozen Healthy / Sick non-TB / TB labels.
Training-only affine augmentation remains within 5 degrees and 2% translation,
with no horizontal flipping. Validation uses deterministic full-image resizing,
RGB conversion, and fixed ImageNet normalization. Selection uses validation
macro-F1, then lower validation loss, then the earlier epoch on an exact tie.

The complete frozen partitions contain 5,780 training images with class counts
`[2660, 2560, 560]` and 1,239 validation images with counts `[570, 549, 120]`.
The audited adapter checks assignment IDs, labels, checksums, duplicate families,
and summary provenance; all development image bytes are checked against their
audited SHA-256 before training. Only these two partitions enter datasets.
Held-out internal-test and external image bytes are inaccessible to the runner.
Metadata parsing does include the complete frozen assignment table.

M3A fixes seed 42, batch 2, 224px, FP32, AdamW learning rate 1e-4 and weight decay
1e-4. No rebalance, oversampling, exclusion, scheduler, precision comparison or
imbalance intervention is performed. These are provisional engineering settings,
not final M3B hyperparameter approvals. Existing M1/M2 runners and verification
artifacts are preserved.

## Infrastructure and bounded execution

`src.training.epoch.EpochTrainer` maintains one AdamW instance and provides one
complete training epoch, one validation pass, and epoch-boundary continuation.
Training uses a seeded shuffled DataLoader and validation uses a separately seeded
ordered loader. Both use zero workers and `drop_last=False`; the final singleton
validation batch is retained. The engine checks exact sample coverage and rejects
partial validation for selection. Logits, loss, gradients, updated parameters and
checkpoint tensor state must be finite. A failed step stops without retries.

`python -m src.training.preflight` launches the worker in a hidden subprocess.
An independent 3,600-second deadline includes worker imports, integrity checks,
initialization, training, validation, checkpoint work and cleanup. The parent kills
the worker at the deadline, waits for process termination, and writes a failed
receipt. Termination/receipt cleanup may take a few additional seconds; training
is not extended. Operation-boundary guards also enforce elapsed time. The worker
CLI requires supervisor launch. Calling the Python API directly only provides
operation-boundary time checks; real execution here uses the supervisor.

The runner initializes a fresh cached ImageNet model after the short compatibility
check, trains at most one complete epoch, validates once, saves a local checkpoint,
reloads all continuation state, compares hashes, and stops. Reload verification
does not repeat real validation or make optimizer updates. `epochs: 1` is a total
cap, including a resumed checkpoint's completed epochs. The CLI can reload a
completed M3A checkpoint but cannot start epoch two. The reusable engine is tested
with multiple fixture epochs; it has no independent entry point or approved
research schedule. No M3B runner or schedule is activated.

The existing GPU monitor samples temperature, utilization, device memory and
hardware thermal slowdown about once per second. M3A stops at 80 C, any active
hardware thermal slowdown, over 5,120 MiB Torch reserved memory, or missing required
temperature/thermal telemetry. Final monitoring can invalidate success. Receipts
strip GPU UUIDs and omit exception text containing machine-local paths. GPU clocks,
power limits, drivers and OS settings are not modified. Polling cannot establish
instantaneous thermal maxima; telemetry is device-wide, Torch memory is per process.

Local receipts update every five seconds, with completed sample/step counters,
phase, GPU samples and memory peaks. A forcibly killed worker's last receipt is a
lower bound on completed work. Step times include batch fetch and synchronization;
training/validation throughput uses actual complete-phase wall time. Epoch and
total timing include separately documented checkpoint and setup costs.

## Checkpoints and resume evidence

`latest.pt` is written atomically only after complete training and validation.
It contains current model parameters/buffers, AdamW state, disabled FP32 scaler
state, completed epoch and optimizer-step counters, Python/NumPy/Torch CPU and all
CUDA RNG states, both loader-generator states, complete history, selected
validation metadata and a CPU copy of the best model state. Best selection state
is retained inside the continuation checkpoint, even when a later epoch is worse.

Restore rejects incompatible format/counters, changed configuration or config
file, changed frozen inputs/membership, changed source, changed initialization or
package/device environment, malformed RNG topology, non-finite tensors and model
hash mismatch. This deliberately supports exact continuation in the same recorded
environment; cross-version or cross-device equality is not claimed. Checkpoints
are local trusted artifacts loaded with `weights_only=True`; model weights never
enter Git. Any partial epoch must be discarded in favor of the last complete
boundary. M3A's checkpoint is preflight evidence and is not a final research model.

The stochastic tiny fixture uses real AdamW updates, train transforms, dropout,
Python randomness and NumPy randomness. Two uninterrupted epochs are compared
with one epoch, save, fresh engine construction, restore, and epoch two. Tests
compare batch order, losses, model hashes, optimizer state and final RNG state
exactly, on CPU and in a separately enabled actual CUDA fixture. This is resume
execution evidence on synthetic images, distinct from real full-data execution.
No additional full-data epoch is used to test continuation.

## Commands and local artifacts

From the repository root in PowerShell, reuse the established environment and
cached weights; no packages are upgraded and no data audits or splits rerun:

```powershell
.\.venv-m1\Scripts\python.exe -m pip check
.\.venv-m1\Scripts\python.exe -m pytest -q
.\.venv-m1\Scripts\ruff.exe check .
.\.venv-m1\Scripts\ruff.exe format --check src/training/pilot.py src/training/smoke.py src/training/epoch.py src/training/full_data.py src/training/preflight.py src/data/provenance.py tests/test_m2.py tests/test_m3a.py
$env:M3A_TEST_CUDA_FIXTURES = '1'
.\.venv-m1\Scripts\python.exe -m pytest tests/test_m3a.py -k interrupted_resume -q
Remove-Item Env:\M3A_TEST_CUDA_FIXTURES
.\.venv-m1\Scripts\python.exe -m src.training.pilot --config tmp/m3a-compatibility.yaml --device cuda --output-dir results/m3a-compatibility
.\.venv-m1\Scripts\python.exe -m src.training.preflight --config configs/preflight.yaml --output-dir results/m3a-cuda-fp32
```

The local compatibility YAML copies `configs/pilot.yaml`, changing only epochs to
1, both per-class quotas to `[2,2,2]`, maximum steps to 3, and duration to 60 seconds.
It is an ignored engineering check; its trained model is discarded. The complete
preflight uses only the committed `configs/preflight.yaml`.

Ignored local artifacts, relative to the repository root:

- `results/m3a-compatibility/receipt.json` and `best.pt`;
- `results/m3a-cuda-fp32/receipt.json` and `latest.pt`;
- `results/m3a-cuda-fp32.log` (supervised worker progress);
- `tmp/m3a-compatibility.yaml` (short-check settings).

Every new execution requires fresh output and log names. A completed checkpoint
can be verified again with `--resume results/m3a-cuda-fp32/latest.pt` and a fresh
output directory; no extra epoch is permitted. Relocated images require
`--image-root` pointing to the local release's `imgs` directory. Machine-local
image paths must never be copied into public reports.

## Actual results

The actual full-data CUDA execution passed at source commit
`e40a3e9bae834fdbc876df8e9113749c44f18fc2` on 2026-10-13, using the unchanged
Python 3.12.10 / Torch 2.13.0+cu126 / torchvision 0.28.0+cu126 environment,
CUDA 12.6, NVIDIA driver 610.88, and cached ImageNet weights. The separate short
compatibility check passed three updates on six training and six validation
images, took 27.80 seconds, and sampled 58–60 C with no thermal slowdown. That
check's weights were discarded before fresh initialization of the full run.

The full preflight consumed **5,780 unique training images exactly once** and
**1,239 validation images exactly once**, completing **2,890 finite optimizer
updates** and **620 validation batches**. The last validation batch had one image.
There was exactly one complete training epoch and one full validation pass.
No full-data continuation epoch, extra full validation, comparison, held-out
evaluation or M3B execution occurred. Frozen metadata and M1/M2 artifacts match
`origin/main`; no split generation or audit was rerun.

| Observed quantity | Result |
| --- | ---: |
| Supervisor wall time, including imports | 1,835.54 s (30.59 min), under the 3,600 s cap |
| Worker duration | 1,821.55 s |
| Training phase | 1,626.61 s (27.11 min), 3.553 images/s |
| Validation phase | 47.00 s, 26.360 images/s |
| Epoch including checkpoint creation | 1,675.37 s |
| Training step time including fetch | Median 0.523 s; mean 0.556 s; range 0.455–2.667 s |
| Peak Torch allocated / reserved memory | 402.28 / 436 MiB |
| Peak sampled device memory | 592 MiB (device-wide) |
| GPU telemetry | 1,546 samples; 58–65 C, median 63 C |
| Sampled utilization | 0–43%, mean 19.84%, median 21% across the monitored interval |
| Thermal slowdown / telemetry failures | None; every sampled slowdown value was `Not Active` |
| OOM / non-finite / duration / memory stops | None |

Setup includes full development-image checksum verification and initialization;
interpreter imports precede the worker's timer but are included by the supervisor.
Overall memory peaks include checkpoint reload; training held 426 MiB reserved.
This is actual sustained execution with conservative finite checks, not a benchmark
for optimal GPU utilization. No hardware or operating-system settings were changed.

**One-epoch preflight diagnostics only:** unweighted training loss **0.356226**;
validation loss **0.152113**, accuracy **0.947538**, and macro-F1 **0.908306**.
Per-class recall is Healthy **0.984211**, Sick non-TB **0.910747**, TB **0.941667**.
The confusion matrix uses true rows / predicted columns in that order:

```text
[[561,   2,   7],
 [  3, 500,  46],
 [  4,   3, 113]]
```

The single available complete validation epoch selected the checkpoint under the
approved rule. These results cannot establish convergence, a final epoch cap or
patience, threshold policy, test performance, or external generalization. They do
not authorize a loss or augmentation revision.

`latest.pt` is 113,027,673 bytes (107.79 MiB) and remains ignored. Its SHA-256 is
`84e8e00058f047d341249fc5b8f0459f1b874ad6262e65447496b5e65dcecb97`.
Reload matched the model state and all continuation state exactly, including
optimizer, scaler, Python/NumPy/Torch CPU/CUDA RNGs, both loader generators,
counters, history and selected-best state. No real images were decoded again for
this verification. Deterministic training continuation was separately established
by the CPU and actual CUDA stochastic fixtures, each comparing a resumed second
fixture epoch with uninterrupted execution. Full-data second-epoch equivalence
was intentionally not tested because it is outside M3A authorization.

The sanitized [m3a-verification.json](m3a-verification.json) includes observed
settings, source/input/weight/checkpoint hashes, package versions, exact counts,
training-order hash, metrics, runtime summaries, telemetry summaries and hashes
of both original local receipts. Raw per-step/order/telemetry evidence remains in
the ignored local receipt (1,637,089 bytes). Only compact metadata is versioned;
no radiographs, GPU UUIDs, private paths or downloaded/model weights are committed.

Offline verification and actual execution are distinct. The final full local
suite passed **125 tests, 1 skipped** in **65.93 seconds**. Ruff across the repository,
formatting checks on all maintained M1/M2/M3A files, and `pip check` passed.
The actual CUDA resume fixture passed both
CPU and CUDA cases (2 passed); default offline/CI suites skip only that opt-in
CUDA case. Mocked tests cover timeout, OOM, numeric/resource failures and missing
thermal telemetry; these are not actual hardware failures. The final diff review
checks that all `data/` files, existing M1/M2 runners and verification receipts are
unchanged. Remote CI is reported separately from local checks.

## Changed files

| File | Purpose |
| --- | --- |
| `configs/preflight.yaml` | Approved bounded starting settings and frozen input references |
| `src/training/epoch.py` | Persistent optimizer, complete epochs, checkpoint and RNG continuation |
| `src/training/full_data.py` | Audited full partitions, provenance, guards, receipts and exact reload |
| `src/training/preflight.py` | Independent hard-deadline supervisor |
| `tests/test_m3a.py` | Complete counts, isolation, CPU/CUDA resume, binding rejection and failures |
| `.github/workflows/tests.yml` | Add M3A formatting checks to existing offline CI |
| `docs/m3a-preflight.md`, `docs/m3a-verification.json` | Runbook, actual versus fixture evidence, proposals and compact receipt |
| `README.md`, `docs/current-project-state.md` | Milestone status and the M3B approval gate |

## Proposed M3B decisions for Research HQ

These are recommendations for later approval, not executed choices. Leave
`configs/baseline.yaml`'s historical 20 epochs / patience 5 placeholders untouched
and do not interpret them as authorization.

| Choice | Proposal and rationale |
| --- | --- |
| Starting settings | Retain batch 2, 224px, FP32, seed 42, AdamW 1e-4 and weight decay 1e-4 if M3A is safe. Avoid changing several variables after a feasibility check. |
| Maximum epochs | Propose a cap of 10 initially, about 4.7 hours from the measured 27.92-minute epoch before additional setup/interruptions. This is a planning extrapolation, subject to Research HQ's time budget; one epoch cannot establish convergence or an optimal cap. |
| Early stopping | Propose patience 3 complete validation epochs under the approved selection rule. This is a bounded starting proposal; M3A cannot demonstrate appropriate patience. |
| Scheduler / learning rate | Keep a constant 1e-4 initially, with no scheduler, to preserve an interpretable primary baseline. Revisit only with separately approved training evidence. |
| Optimizer / architecture | Retain AdamW and full-model ImageNet DenseNet-121 fine-tuning. No architecture or optimizer comparison is needed to resolve execution feasibility. |
| Primary loss / augmentation | Preserve the approved unweighted CE and conservative training-only affine transform. Do not use one-epoch recall to silently revise the primary baseline. |
| Imbalance comparison | After the primary baseline, consider one separately approved inverse-frequency CE comparison with training-only weights; no simultaneous oversampling or targeted augmentation. M3A executes none. |
| Hyperparameter sweeps | Do not start a sweep; any comparison needs explicit scientific scope and computational budget. |
| Binary threshold | Leave unset. Research HQ must predeclare a validation-only selection objective before threshold selection, and freeze it before held-out evaluation. |
| M3A checkpoint reuse | Start M3B from fresh ImageNet weights and seed 42 after approval. Keep the one-epoch checkpoint as engineering evidence. |
| Held-out evaluation | Authorize it separately only after development/model/threshold choices are frozen; M3A provides no internal-test, Shenzhen or Montgomery metrics. |

Engineering assumptions: zero loader workers, two CPU threads, same-environment
resume, periodic receipts, and exact-state reload without a second full validation
pass. These are implementation choices, not additional scientific approvals.
