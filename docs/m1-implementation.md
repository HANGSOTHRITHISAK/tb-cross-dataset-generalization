# M1: frozen split to DenseNet-121 smoke

Date: 2026-10-13. This milestone verifies an executable engineering pathway;
it supplies no estimate of scientific model performance.

## Implemented behavior

`load_frozen_tbx11k` joins by stable sample ID and preserves CSV order and every
stored assignment. It rejects missing/duplicate IDs, contradictory labels,
checksums, repeated retained hashes/families, invalid family members and unsafe
relocation paths. Extra audited images are expected: the source manifest also
contains duplicate members, official unreleased test material and supplemental
images. Only retained frozen representatives enter the joined dataset. The CLI
binds the full audited manifest to its frozen summary and locks the assignment
file by SHA-256 in `configs/smoke.yaml`. No split-generation or overlap-audit
operation runs. Frozen files retain exact bytes across Git checkouts.

An optional `--image-root` identifies the local release's `imgs` directory;
paths below it use the audited `relative_release_path`. Without this option,
existing manifest-relative paths work. No machine paths or radiographs are
written to receipts or committed source.

All supported 8-bit images explicitly become RGB, resize to 224 x 224 using
bilinear interpolation, scale to float32 and receive fixed ImageNet mean/std
normalization. Evaluation has no randomness. Training alone uses a rotation
within 5 degrees and translation within 2%, without mirroring or random crops.
This is a provisional conservative preprocessing choice for the training pilot;
it needs collaborator review before full training. High-bit-depth/float PIL
modes fail rather than silently clipping. No project-data statistics are fitted.
These transforms intentionally resize the full image; they do not claim to be
torchvision's center-crop inference preset.

The model uses torchvision DenseNet-121 with `IMAGENET1K_V1` initialization and
a new three-output head. Missing weights fail with download instructions;
downloads require `--allow-weight-download`. Cached weights are checked against
the published filename hash prefix. Fixture tests explicitly use random weights.
The model never silently substitutes random initialization for ImageNet weights.
See the [official model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).

The smoke CLI selects a seeded tiny sample from **train only**, verifies each
selected image's SHA-256, checks batch dtype/shape, and performs exactly one full
model AdamW step. Logits, loss, gradients and updated parameters must be finite;
classifier weights must change. Batch size is limited to 2–4 and image size to
32–224; there is no epoch loop. Validation/internal-test and external image bytes
are never opened. Python, NumPy and Torch are seeded; deterministic algorithms
and cuDNN settings are enabled. Reproducibility applies within the recorded
environment, not bitwise equality across hardware or library versions.

CUDA is explicit with no CPU fallback. Default execution uses FP32; `--amp`
requests CUDA FP16, never BF16. Its initial GradScaler scale is 1,024 because a
one-step smoke cannot use several skipped steps to calibrate the default 65,536;
all non-finite gradients still cause failure. A receipt includes versions, input
hashes, sample IDs, settings, finite loss, update evidence, runtime, CUDA peak
allocated/reserved memory, and best-effort one-second `nvidia-smi` samples with
GPU UUIDs. Device-wide memory/utilization includes other processes; Torch peaks
describe this process. Unavailable sensors are null or carry an error type.
At 85 C or a hardware thermal-slowdown signal the runner stops at the next
operation boundary and reports failure; it cannot interrupt a running kernel.
Final thermal detection also invalidates success. No clocks/power limits change.
Receipts are atomic JSON in ignored `results/`; failure receipts retain an error
type and progress evidence. Failures before valid CLI configuration parsing or
an unwritable output destination may prevent receipt creation.

## Reproduce

From the repository root in PowerShell, create a separate Python 3.12 environment:

```powershell
py -3.12 -m venv .venv-m1
# Use the official primary download host directly.
.\.venv-m1\Scripts\python.exe -m pip install "https://download.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp312-cp312-win_amd64.whl" "https://download.pytorch.org/whl/cu126/torchvision-0.28.0%2Bcu126-cp312-cp312-win_amd64.whl"
.\.venv-m1\Scripts\python.exe -m pip install -r requirements-m1.txt
.\.venv-m1\Scripts\python.exe -m pip check
.\.venv-m1\Scripts\python.exe -m pytest -q
.\.venv-m1\Scripts\ruff.exe check .
.\.venv-m1\Scripts\python.exe -m src.training.smoke --device cuda --batch-size 2 --allow-weight-download --receipt results/m1-cuda-fp32.json
```

For relocated raw data append `--image-root $env:TBX11K_IMAGE_ROOT` after setting
that local variable to the `imgs` directory. To check FP16, run one additional
bounded command with `--amp --receipt results/m1-cuda-fp16.json`. For a CPU
real-image check use `--device cpu`. Full tests use offline fixture images and
random model weights; no radiographs, CUDA device or network downloads are
required by pytest. The repository-manifest integration test reads metadata only.

The [official installation matrix](https://pytorch.org/get-started/previous-versions/)
lists this matching Torch/torchvision CUDA 12.6 pair. CPU installations can install
the same base versions with `--index-url https://download.pytorch.org/whl/cpu`
before installing the remaining requirements. The existing local environment is
preserved; the M1 environment and weight cache are ignored/local-only.

## Actual verification

Environment: Windows, Python 3.12.10, pip 25.0.1, Torch 2.13.0+cu126,
torchvision 0.28.0+cu126, CUDA runtime 12.6, NVIDIA driver 610.88. The RTX 2060
Max-Q reports 6,144 MiB and compute capability 7.5; the wheel includes `sm_75`.
Exact observed package versions are pinned in `requirements-m1.txt` (CUDA/CPU
wheel tags are selected by the installation source). The original CPU-only
environment and existing local branches remain preserved.

- Original suite before implementation: **23 passed** (24.09 s).
- Final suite after the FP16 correction: **62 passed** (49.87 s), no skips.
- Ruff: **all checks passed**; new Python files also pass format checking.
- `pip check`: **no broken requirements**; final diff has no whitespace errors.
- Frozen join: **8,257 rows**, preserving every assignment and checksum; the
  frozen manifest, split, summary, adjudications and audit artifacts are unchanged.
- Actual GPU execution: ImageNet-initialized DenseNet-121, batch `[2, 3, 224, 224]`,
  float32 input, int64 labels `[2, 0]`, output `[2, 3]`. The seeded sample IDs are
  `tbx11k:tb/tb0036.png` and `tbx11k:health/h1703.png`; both raw hashes matched.
  Only training images were opened. Each successful run used a fresh model and
  verified one full-model optimizer update; no checkpoint was retained.

| Actual run | Result | Allocated / reserved peak MiB | GPU step seconds | Total runner seconds | Sampled temperature | Sampled utilization |
| --- | --- | ---: | ---: | ---: | --- | --- |
| FP32 | passed; loss 1.169044 | 348.5 / 410 | 2.85 | 30.82 | 57–58 C | 0–6% |
| FP16, initial scale 1,024 | passed; loss 1.167725 | 235.3 / 290 | 3.33 | 26.76 | 58–59 C | 0–2% |

The first FP16 attempt with PyTorch's default initial loss scale 65,536 failed
the non-finite-gradient guard **before any optimizer update**. Lowering the
initial scale to 1,024 resolved that numeric overflow; the failed receipt is
retained alongside the successful retry. No sampled hardware thermal slowdown,
overheating, OOM or abnormal hardware behavior was observed. Four sensor samples
were collected per run; these brief samples do not measure true utilization or
temperature peaks. Total runner time includes CPU setup and, for the FP32 run,
the initial weight download. The FP16 retry overlapped CPU fixture tests. These
timings are execution observations, not precision-mode speed benchmarks.

Original JSON receipts remain in ignored `results/`. Sanitized receipts,
original-receipt hashes, test outcomes and the failed-attempt explanation are
versioned in [m1-verification.json](m1-verification.json); GPU UUIDs are omitted
from the public copy. Torch's 2.59 GB CUDA wheel was obtained from the official
host with validated range requests after download trouble, then checked against
the full SHA-256 from the official index before installation. ImageNet weights
were downloaded into the local Torch cache and are not committed.

CPU fixture CI is added in `.github/workflows/tests.yml`. Its remote result is
separate from the local evidence above; it opens no real image and requires no
CUDA device. No further local verification blocks M1 review. Optional future
reruns use the commands above; they must not expand into full training.

## Research boundary and next milestone

No population, frozen assignment, label mapping, duplicate disposition, metric
definition or held-out boundary changes. No patient-level independence claim.
The baseline YAML's epoch/LR/early-stopping values remain provisional; batch size
2 replaces the unpiloted 32 as the initial 6 GB hardware setting. The smoke uses
its own configuration and supplies no pilot justification for final settings.

The next milestone is a short training/validation pilot with checkpoints,
validation-only selection, resource accounting and a documented freeze of the
final training settings. Full baseline training and all held-out evaluations are
pending. The audit's proposed like-for-like internal binary reporting should be
documented explicitly during the evaluation milestone alongside existing internal
three-class and external binary metrics. AUROC should use continuous TB
probability; any threshold must be selected on validation and frozen before
held-out testing. M1 does not adopt or implement new evaluation metrics.
