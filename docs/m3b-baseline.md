# M3B: reproducible baseline controller, implementation only

Date: 2026-10-13. Prerequisite M3A PR #17 was independently audited and merged as
`3f1e959793afc35d707a931b32e8636202ca33aa` before this feature branch was created.
This milestone authorizes implementation and fixture verification only.
**No real M3B training epoch or 10-epoch experiment has been executed.**

## Approved experiment

The authoritative executable configuration is `configs/m3b.yaml`: ImageNet
DenseNet-121, three outputs ordered Healthy / Sick non-TB / TB, all 5,780 frozen
training images and all 1,239 validation images, 224px RGB, fixed ImageNet
normalization, training-only affine within 5 degrees and 2% translation, no flip.
AdamW uses constant learning rate 1e-4, weight decay 1e-4, batch 2, FP32, unweighted
cross-entropy, seed 42, maximum 10 total epochs and patience 3. Selection maximizes
validation macro-F1, then minimizes validation loss, retaining the earlier epoch
on an exact tie. There is no scheduler, weighting, oversampling or new augmentation.

Fresh baseline initialization must use cached `IMAGENET1K_V1` weights and a newly
seeded three-class head. The M3A one-epoch checkpoint is refused as a continuation
checkpoint. The historical `configs/baseline.yaml` is not consumed by this runner;
its old 20/5 placeholders do not govern the newly approved 10/3 protocol.
Threshold selection and internal-test/Shenzhen/Montgomery evaluation remain gated.

## One invocation, one additional epoch

`python -m src.training.baseline_session` starts the existing M3A hard supervisor.
The parent imports no Torch; the deadline includes worker imports, complete
development-image checksum verification, initialization, training, validation,
checkpoint writes and cleanup. Maximum duration is 3,600 seconds per invocation.
The worker `src.training.baseline` requires the supervisor's internal marker;
direct worker CLI launch is refused. The Python API used by fixtures has only
operation-boundary duration guards; future real execution must use the supervisor.

Every invocation requires a fresh output directory and log filename. It restores
a compatible prior M3B checkpoint or initializes a new experiment, executes one
complete training epoch and one validation pass, saves and verifies the new
boundary and best model, then returns. It never schedules another invocation.

`BaselineTrainer` subclasses the established `EpochTrainer`: the M3A training,
validation, data ordering, coverage checks and engine serialization are reused.
The original M1/M2/M3A runners and epoch engine are unchanged. AdamW persists
between batches and is restored between sessions. Zero workers and two CPU threads
are deterministic engineering settings bound to the experiment. Both loaders retain
all examples; the last singleton validation batch is kept. The audited join parses
held-out metadata but only assigned training/validation records enter hashing,
decoding, augmentation or optimization. No held-out image bytes are opened.

Each completed session increments the global epoch count by one. Stale epochs
increase only when a complete validation epoch fails the approved improvement
rule, including exact ties, and reset on improvement. They persist in checkpoints.
At stale count 3 or epoch 10 the experiment is finished and further training is
refused. If both conditions occur at epoch 10, the receipt reports early stopping;
the global cap still applies. Partial training or validation cannot finish/select
an epoch or produce a continuation checkpoint.

Receipt states distinguish:

| State | Meaning |
| --- | --- |
| `session_complete_experiment_active` | This session passed; a later separately authorized invocation may continue |
| `experiment_finished` | A successful session reached patience 3 or epoch 10 |
| `failed_or_interrupted` | No successful continuation claim; inspect phase, counters and failure reason |
| `experiment_already_finished` failure | A further invocation was refused before any optimizer update |

For an incompatible resume rejected before restoration, epoch/session counters
remain unknown (`null`) rather than inventing progress from an unread checkpoint.
Completed sample/step counters are persisted periodically; a forced timeout's
last receipt is explicitly a lower bound on work before termination.

## Immutable bindings and checkpoint safety

Scientific settings are immutable and checked against the approved protocol.
Source hashes, frozen manifest/split/summary hashes, exact partition membership,
initialization/weight hashes, Python/package/CUDA/device environment and driver
versions bind continuation to the same supported environment. Optimizer group
settings are checked independently of the configuration; changing the learning
rate, weight decay or optimizer defaults is rejected.

Session output path and duration are deliberately excluded from these immutable
bindings. The config-file hash is an audit field, while semantic experiment
settings are bound: changing a config copy's location/format or a session's output
directory/time limit does not invalidate an otherwise identical experiment.
Session GPU guard values may be made stricter, never relaxed beyond 80 C / 5 GiB.
Source, data, package, driver or device changes require investigation rather than
a silent claim of exact continuation. Cross-environment bitwise equality is not
claimed. A docs-only Git commit does not change the source binding.

Each successful output directory contains:

- `latest.pt`: current model/optimizer/disabled FP32 scaler, Python/NumPy/Torch
  CPU/all-CUDA/loader RNG states, epoch/step counters, full history, best model and
  metrics, experiment ID/configuration, stale count, terminal reason and a digest
  over the complete serialized continuation payload;
- `best.pt`: a separate CPU model-state export from the validation-selected epoch,
  selected metrics/epoch, immutable binding, experiment ID and model-state hash;
- `receipt.json`: unique run ID, session/experiment IDs, previous checkpoint hash,
  counts, steps/losses, history/metrics, selected epoch, stopping state, phase and
  completion status, runtimes/throughput, CPU/GPU telemetry and checkpoint hashes;
- a sibling `.log`: supervisor worker output, local-only.

Writes use same-directory staging and atomic replacement. The prior session's
files are read-only inputs; a failed new session cannot overwrite them. The engine
checkpoint is privately staged, extended with M3B controller state, payload-hashed
and atomically published. The best model is copied from the selected state, so it
survives later worse epochs. After writing, the controller restores and compares
all continuation state exactly and checks the best-model export; it does not run
another validation pass or optimizer update.

Public resume requires both `latest.pt` and its successful companion
`receipt.json`, with a matching checkpoint hash, experiment ID and completed epoch.
Keep/move the whole prior session directory together. A file produced by a failed
or interrupted session is not accepted merely because it can be deserialized.
Resume after partial work uses the last successful complete boundary, discarding
that partial epoch. There are no automatic retries after numeric/resource failures.
An explicitly authorized retry must use a new output directory and prior successful
checkpoint; never overwrite a failed receipt to turn it into success.

## GPU safeguards and CPU telemetry limitation

The existing GPU monitor samples temperature, utilization, device memory and
hardware thermal slowdown, with GPU UUIDs stripped from receipts. M3B stops at
80 C, active hardware thermal slowdown, over 5,120 MiB Torch reserved memory,
missing required GPU telemetry, OOM or non-finite logits/loss/gradients/parameters.
It also checks optimizer state after every optimizer update and before checkpoint
serialization. No automatic precision change or retry is used. Final monitoring
can invalidate session success. Polling cannot establish instantaneous maxima;
device-wide memory/utilization differ from per-process Torch memory. The deadline
kills the worker; termination/receipt cleanup may take a few additional seconds,
without extending training.

The new CPU adapter reads only already available local Windows WMI providers
`root\LibreHardwareMonitor` and `root\OpenHardwareMonitor`. It joins temperature
sensors to explicitly typed CPU hardware and accepts recognized Intel/AMD CPU
parent identifiers, rejecting GPU/board/anonymous values and invalid/null readings.
It never installs or starts a hardware-monitor service or driver and never writes
fan, clock or power settings. Provider values are best-effort observations; the
adapter cannot independently prove the hardware provider's sensor freshness.
Success-path parsing and provider loss are tested with mocks, not live CPU sensors.

**Actual local investigation:** both hardware-monitor namespaces and the ACPI
thermal-zone query were unreadable (`CimException`). The implemented adapter also
returned `no_readable_cpu_temperature_sensor`, zero CPU samples and
`temperature_available: false`. No CPU temperatures are fabricated. Receipts
explicitly state that GPU telemetry does not establish CPU or whole-laptop thermal
safety. There is no invented CPU stop threshold.

Microsoft documents that `Win32_TemperatureProbe.CurrentReading` is not populated
from SMBIOS, so it is not used as a live CPU thermometer.
[Microsoft class documentation](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-temperatureprobe).
ACPI thermal zones abstract regions containing potentially multiple devices;
unidentified zone readings must not be treated as processor package/core readings.
[Microsoft thermal architecture](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/design-guide).
The existing-provider approach follows LibreHardwareMonitor's WMI interface;
temperature values use degrees Celsius.
[WMI example](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/blob/master/LibreHardwareMonitor.Windows.Forms/TestScripts/basicwmi.py),
[sensor interface](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/blob/master/LibreHardwareMonitorLib/Hardware/ISensor.cs).

Before each future authorized session, use a hard stable surface, keep vents clear
with adequate ventilation, connect the appropriate charger, and watch for
unexpected heat, fan behavior, throttling or shutdowns. Stop and inspect unexpected
behavior even if the GPU sample is below 80 C. Reliable CPU observation remains a
limitation to address manually; installing monitoring software/drivers or changing
hardware settings is outside this implementation task.

## Fixture verification and commands

All M3B training verification uses synthetic images: six training and three
validation images, with nonexistent held-out fixture files. A Python-only fixture
mode refuses approved real metadata before reading any real image bytes. The
stochastic tiny model includes dropout plus Python/NumPy randomness and the real
training transform, exercising model/optimizer and all supported RNG sources.
CPU and optional actual CUDA tests compare two uninterrupted fixture epochs with
two separate sessions, changing output directory and session budget/config-file
audit hash. Batch order, losses, model state, optimizer state and final RNGs match
exactly. No M3B real-data execution is claimed.

Early-stop fixtures run actual synthetic optimizer updates and validation, then
control selection metrics to exercise exact ties: epoch 1 selected, epochs 2–4
stale counts 1/2/3, selected epoch 1 retained, and a fifth invocation refused.
Other fixtures cover a lower validation loss improving tied F1, a worse F1 failing
despite lower loss, an exact tie retaining the earlier epoch, improvement resetting
patience, 10 global epochs with checkpoint restoration, terminal refusal, changed
bindings, payload corruption, partial failures, atomic-write interruption, failed
receipt rejection, timeout/resource paths and CPU telemetry availability/loss.
Controlled fixture selection metrics are mocked controller evidence, not scientific
validation performance. Detailed observed evidence is recorded in
`docs/m3b-verification.json`.

Final local verification passed **165 tests, 2 optional CUDA skips** in
**96.33 seconds**. The separately enabled M3B CPU/CUDA equivalence fixture passed
**2 cases** in **11.14 seconds**. Ruff, all maintained-file formatting and dependency
checks passed. Existing M1/M2/M3A tests remain in the complete suite; the two default
skips are their optional M3A CUDA case and the new optional M3B CUDA case. Remote
CI is reported separately in the PR's checks and final delivery report.

The standalone demonstration completed four actual CPU fixture sessions with
controlled selection scores, retained best epoch 1, terminated at stale count 3,
and refused a fifth session with zero new samples/updates. Two existing fixture
sessions also match a two-thread uninterrupted reference exactly in order, losses,
model, optimizer and RNG state. Local receipts/checkpoints remain under ignored
`results/m3b-fixture-demonstration-verified/`.

Two standalone setup mistakes were corrected transparently: relative fixture
paths initially failed before training, and the uninterrupted reference initially
used eight default CPU threads instead of the approved two. Correcting paths and
thread count resolved them without changing the controller. The failed setup
receipt is retained under ignored `results/m3b-fixture-demonstration/`; its hash
and both corrections are included in the compact report. These are synthetic
verification issues, not actual baseline or hardware failures.

Use the unchanged environment; no dependency additions or upgrades are needed:

```powershell
.\.venv-m1\Scripts\python.exe -m pip check
.\.venv-m1\Scripts\python.exe -m pytest -q
.\.venv-m1\Scripts\ruff.exe check .
.\.venv-m1\Scripts\ruff.exe format --check src/training/pilot.py src/training/smoke.py src/training/epoch.py src/training/full_data.py src/training/preflight.py src/training/baseline.py src/training/baseline_session.py src/training/cpu_temperature.py src/data/provenance.py tests/test_m2.py tests/test_m3a.py tests/test_m3b.py
$env:M3B_TEST_CUDA_FIXTURES = '1'
.\.venv-m1\Scripts\python.exe -m pytest tests/test_m3b.py -k multisession_exact -q
Remove-Item Env:\M3B_TEST_CUDA_FIXTURES
```

The exact recommended **future session 1 command, not executed in this task**, is:

```powershell
.\.venv-m1\Scripts\python.exe -m src.training.baseline_session --config configs/m3b.yaml --output-dir results/m3b-baseline/session-01 --max-seconds 3600
```

Only after session 1 succeeds and the next session is separately authorized:

```powershell
.\.venv-m1\Scripts\python.exe -m src.training.baseline_session --config configs/m3b.yaml --resume results/m3b-baseline/session-01/latest.pt --output-dir results/m3b-baseline/session-02 --max-seconds 3600
```

There is no command loop or scheduled continuation. For relocated data append
`--image-root` pointing to the local release's `imgs` directory. Keep all session
artifacts under ignored `results/`; `.pt` and staging `.pt.tmp` are ignored globally.
Only sanitized lightweight fixture/verification metadata is committed. No private
paths, radiographs, GPU identifiers, cached weights or model checkpoints enter Git.

## Remaining gate before real session 1

Review and merge this implementation PR, then obtain explicit authorization for
the first real M3B session. Confirm unchanged source/environment/data, cached
ImageNet weights, a fresh session output/log name and adequate local storage.
The new per-step optimizer checks add work beyond M3A; their full-data runtime
has not been measured under this implementation-only authorization. The supervisor
must stop and report incomplete execution if a future epoch exceeds 60 minutes.
Address the CPU-temperature observation limitation and manual laptop precautions.
The M3A measured epoch supports planning, but fixture success does not establish
real M3B checkpoint performance or runtime. Do not reuse its checkpoint, choose
an external threshold, open held-out images or start a full 10-epoch sequence.
