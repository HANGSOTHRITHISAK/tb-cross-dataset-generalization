# M2: bounded training and validation pilot

Date: 2026-10-13. This milestone verifies training infrastructure using a small
development subset. Its metrics are not estimates of final research performance.

## Execution contract

`python -m src.training.pilot` fine-tunes the existing ImageNet-initialized
DenseNet-121 with a three-output head. AdamW and its state persist across batches
and epochs. FP32 is the initial pilot configuration; explicit `precision: fp16`
uses CUDA autocast and GradScaler with initial scale 1,024. Non-finite logits,
losses, gradients or updated parameters stop execution; there is no skipped-step
calibration or automatic retry. CUDA requests never fall back to CPU.

The pilot runner accepts 1–3 epochs, batch size 2–4, image size 32–224, positive
per-class subset counts up to 128, at most 1,000 optimizer steps and at most
600 seconds. The committed configuration is stricter: 3 epochs, batch 2, 224px,
81 steps and 180 seconds. Duration includes input verification and initialization;
guards act at operation boundaries and cannot interrupt kernels or file I/O.
Reaching a hard budget before completion produces a failed receipt, retaining
progress but marking any saved checkpoint unusable for a successful pilot claim.
Full baseline training requires a separately reviewed M3 runner/configuration.

The existing frozen adapter validates metadata only for the complete 8,257-image
pool. Subset selection takes sorted sample IDs per class, independently samples
without replacement with Python seeds 42 (training) and 43 (validation), then
sorts the resulting membership. Each subset has 24 Healthy, 24 Sick non-TB and
6 TB images (54 each). All selected image bytes are checked against their audited
SHA-256 before decoding. Training uses a seeded shuffled DataLoader, zero workers,
complete batches and no oversampling; each selected image appears once per epoch.
Validation membership is fixed, independent of training, and uses deterministic
preprocessing. Internal-test, Shenzhen and Montgomery image bytes are never read.
No split, manifest, audit or duplicate assignment is regenerated or altered.

Only training images drive optimizer updates. Augmentation remains the M1
rotation within 5 degrees and translation within 2%, preserving laterality and
the full image. RGB conversion, bilinear full-image resize and fixed ImageNet
normalization apply to both partitions. There are no learned preprocessing
statistics. TB-targeted augmentation is a possible later controlled experiment,
requiring a training-only rationale and validation comparison before adoption.

## Validation and selection

Validation always runs in eval/inference mode and FP32. Report unweighted mean
cross-entropy, accuracy, macro-F1, per-class recall and confusion matrix in the
fixed order `[healthy, sick_non_tb, tb]`. Matrix rows are true labels and columns
are predictions. Undefined recall/F1 is explicitly zero; all three classes
remain in the macro average even if absent. A partial validation pass never
selects a checkpoint.

Predeclared selection maximizes validation macro-F1, breaks ties by lower
validation loss, then retains the earlier epoch on an exact tie. Training loss
cannot select a checkpoint. Optional early stopping counts epochs without an
improvement under this same rule; patience is disabled for this pilot. A tiny
three-epoch run cannot establish final patience, epoch cap or convergence.

`best.pt` is atomically replaced after an improved complete validation pass.
It contains model/optimizer/scaler state and metadata: selected epoch and step,
validation results, exact settings, input hashes, initialization and model-state
hash. After the epoch loop, the runner reloads the best checkpoint, checks its
state hash, and repeats validation; loss and macro-F1 must match exactly within
this environment. Stored Torch RNG state is diagnostic; resume training is not
implemented, and the checkpoint is not a complete resumable RNG/loader snapshot.

Training frequencies come exclusively from all 5,780 frozen training images:
`[2660, 2560, 560]`. Candidate CE weights are `N/(3*n_c)`, approximately
`[0.7243, 0.7526, 3.4405]`. Weighting is implemented and disabled for the initial
pilot; no oversampling is combined with it. Weighted training loss aggregates
batch losses by their CE weight denominator. Validation loss remains unweighted
for comparability. Fixed pilot quotas slightly overrepresent TB; they are bounded
class coverage, not a new frozen split or a full training sampling policy.

## Receipts, monitoring and provenance

Each run requires a fresh output directory. Atomic JSON receipts contain exact
config and config-file hash, Git revision, source hashes, installed package
versions, input byte/LF hashes, subset IDs/counts, seed/determinism controls,
initial/final state hashes, weight hash, step losses/runtimes, epoch history,
selection metadata, checkpoint hash and checkpoint validation. CUDA runs record
total device capacity, process allocated/reserved peaks (epoch peaks are
cumulative), and best-effort one-second `nvidia-smi` snapshots. Step compute times
exclude DataLoader fetch; epoch times include data I/O and checkpoint work.

The initial thermal stop is 80 C or an active hardware thermal-slowdown signal.
Reserved Torch memory is guarded at 5,120 MiB, leaving headroom on the 6,144 MiB
laptop. OOM, resource limits, numeric failures and hard execution budgets stop
without retries and record failure category, phase and completed finite steps.
The final thermal sample can invalidate success. Device-wide telemetry includes
other processes; polling cannot establish true instantaneous thermal peaks.
Unavailable telemetry records a reason/null readings and is best-effort, not a
proof of normal temperature. Monitor readings cover all visible NVIDIA devices.
Output creation/config parsing failures or an unwritable output can prevent
receipt creation. OS/PIL error text is omitted to keep private paths out of
receipts; non-finite tensor descriptions are retained.

M1's deprecated `image_evidence` argument is ignored. Both runners now derive
`verified_real_tbx11k` only from approved manifest/split LF hashes plus selected
raw-image checksums after the strict frozen join. Other valid inputs remain
usable as `unverified_or_fixture_inputs`; caller labels cannot promote fixtures
to verified real execution. The historical M1 receipts are retained unchanged;
their real execution evidence also includes recorded approved hashes/checksums.

Receipts/checkpoints stay under ignored `results/`. Checkpoint extensions and
temporary checkpoint files are ignored globally. Only a sanitized small receipt
and its original hash should be copied to `docs/`; GPU UUIDs, raw radiographs,
cached weights, private paths and large checkpoints must remain local.

## Reproduce

Reuse the M1 environment and cached ImageNet weights, without changing packages:

```powershell
.\.venv-m1\Scripts\python.exe -m pip check
.\.venv-m1\Scripts\python.exe -m pytest -q
.\.venv-m1\Scripts\ruff.exe check .
.\.venv-m1\Scripts\python.exe -m src.training.pilot --config configs/pilot.yaml --device cuda --output-dir results/m2-cuda-fp32
```

For relocated data append `--image-root $env:TBX11K_IMAGE_ROOT`, pointing to the
local release's `imgs` directory. The original M1 environment, weight cache and
other environments are preserved. No download is needed when the cache exists.
To test another explicit precision setting edit a local configuration copy and
use a fresh output directory. M2 does not authorize a hyperparameter sweep.

## Verification and M3

Implementation and offline fixture verification are distinct from actual
execution: the final CPU suite passed **92 tests** (43.47 seconds), including six
multi-step random-initialized DenseNet updates and
tiny-model repeatability, selection ties, validation isolation, class weights,
early stopping, numeric failures, mocked OOM/resource stops, hard budgets and
provenance. Ruff, maintained-file formatting and `pip check` passed. Existing CI
discovers the new offline tests and now checks maintained M1/M2 file formatting;
its remote outcome is reported separately from these local checks.

The actual real-image CUDA run used execution commit
`0f0729756b04bb37855a3049a053deec7298005f`, the unchanged M1 environment (Python
3.12.10, Torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, driver 610.88)
and cached ImageNet weights. Verified real TBX11K bytes: **54 training plus 54
validation images**, disjoint IDs, with `[24,24,6]` class representation in each.
There were **81 finite optimizer updates**, **3 complete validation passes**, and
one additional validation pass after checkpoint reload. The full run took
**72.97 seconds**, below its 180-second budget. No sweep or second GPU pilot ran.

| Epoch | Training loss | Validation loss | Accuracy | Macro-F1 | Train / validation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.998748 | 0.719536 | 0.759259 | 0.535948 | 15.91 / 2.25 |
| 2 | 0.658538 | 0.471300 | 0.814815 | 0.573996 | 14.75 / 2.02 |
| 3 (selected) | 0.642571 | 0.414623 | 0.888889 | 0.780610 | 14.14 / 2.03 |

Selected-checkpoint per-class recall: **Healthy 0.916667, Sick non-TB 1.000000,
TB 0.333333**. Its matrix (true rows / predicted columns, fixed class order) is:

```text
[[22,  2, 0],
 [ 0, 24, 0],
 [ 1,  3, 2]]
```

The checkpoint state hash matched after reload; validation loss, macro-F1 and
confusion matrix exactly reproduced the selected epoch. The local checkpoint
is about 84.4 MB, including optimizer state, and remains ignored. The sanitized
receipt, original receipt hash, exact subsets, packages, source hashes, full
history and checkpoint metadata are in [m2-verification.json](m2-verification.json).
Only six TB validation images were used: the metrics above are **not final
research findings**, cannot estimate full-validation or test performance, and do
not justify finalized hyperparameters or an imbalance intervention.

Peak Torch allocated/reserved memory: **402.28 / 426 MiB**. Peak sampled device
memory: **582 MiB**, including CUDA context/other processes. The monitor collected
51 samples during GPU work: **57–62 C**, **0–39% utilization**; no hardware thermal
slowdown, OOM, non-finite value or resource stop was observed. Median training-step
compute time was **0.487 seconds** (range 0.446–2.136 seconds). These are timings
with conservative finite/determinism checks, not a hardware throughput benchmark.
Runner setup/metadata, image decoding, checkpoint writes and reload account for
additional wall time; interpreter startup/imports precede the recorded runner
duration. The short run cannot establish sustained laptop cooling,
long-run memory behavior or an optimal batch size/precision.

M3 feasibility recommendation: retain **batch 2, 224px, FP32, AdamW 1e-4,
weight decay 1e-4**, zero loader workers initially, the conservative training-only
augmentation and validation-only macro-F1 selection. Use the complete frozen
training/validation partitions in a separately reviewed runner, with a duration
budget, the 80 C guard and memory monitoring. Keep weighting and oversampling
disabled initially unless the scientific imbalance choice is explicitly changed.
No need to increase batch size based on this pilot alone; multi-step FP16 was
implemented but only the historical M1 one-step FP16 GPU result is established.

Scaling observed training epoch times from 27 to 2,890 batches gives roughly
**25–28 minutes per full training epoch**, plus about **46–52 seconds of full
validation** if I/O and resource behavior scale linearly. This is a planning
extrapolation, not a measured full-epoch duration; sustained cooling, decoding and
finite checks may change it. A 20-epoch schedule could therefore require many
hours and must not start automatically. Baseline YAML's 20 epochs / patience 5
remain placeholders. Final epoch cap, patience, full-model fine-tuning choices,
class weighting, augmentation and any validation-selected threshold still need
scientific review. Held-out evaluation follows development decisions and model
freezing; M2 provides no held-out or external results.
