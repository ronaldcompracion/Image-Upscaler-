# Local AI Image Enhancer

A Windows-first desktop application for local image restoration and AI
upscaling. Images stay on the computer. The only network request made by the
application is the explicit download of the selected model.

## Features

- PySide6 desktop UI with drag-and-drop and multi-image queue.
- Original and enhanced previews; PNG, JPEG, WEBP, and TIFF output.
- AI enhancement at 1x, 2x, or 4x with output naming that avoids overwrites.
- EXIF orientation correction, ICC profile pass-through, and alpha-channel
  preservation for formats that support transparency.
- Batch worker thread, progress, cancellation after the current inference,
  per-image errors, and retry of failed images.
- Per-job diagnostic IDs, terminal success/failure/cancel states, output
  decode-and-dimension verification, and recovery after inference failures.
- Hardware report for CPU, memory, NVIDIA/CUDA status, and free disk space.
- Tiled Real-ESRGAN inference, CUDA out-of-memory tile reduction, and CPU
  fallback.
- Rotating diagnostic logs and local model storage.

The application does not change the input files. A PNG/TIFF/WEBP output
preserves transparency; JPEG flattens transparency onto white.

## Requirements and installation

- Windows 10/11 and Python 3.10, 3.11, or 3.12.
- About 1 GB free disk space for the model and basic dependencies. PyTorch
  wheels need additional space.
- An NVIDIA GPU is optional. CUDA inference requires a compatible NVIDIA
  driver; the app checks PyTorch's actual CUDA availability and otherwise uses
  CPU inference.

From PowerShell in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
.\run.ps1
```

The installer creates `.venv`, installs PySide6 and image dependencies, installs
PyTorch 2.5.1 with CUDA 12.1 wheels when `nvidia-smi` is available, then installs
Real-ESRGAN dependencies and checks imports. To explicitly install CPU-only
PyTorch:

```powershell
.\install.ps1 -CpuOnly
```

`run.bat` is also provided for Windows users. On first launch, select **Download
AI model**. Once model setup has completed, inference can run offline. The
installer accesses PyPI and the PyTorch wheel index; the model download is a
separate, user-initiated connection to the official Real-ESRGAN GitHub release.

## First run and workflow

1. Start the app with `.\run.ps1` and download the Real-ESRGAN model from the
   model status row.
2. Add or drop images into the queue.
3. Choose a profile, output scale, processing mode, destination, format, and
   quality. The app saves outputs with the configured suffix and adds a number
   when a name already exists.
4. Start the batch. The UI remains responsive; cancellation takes effect after
   the current inference has returned.
5. Use **Open output folder** to inspect the results.

Model files are stored in
`%LOCALAPPDATA%\LocalAIImageEnhancer\models`; logs are stored in
`%LOCALAPPDATA%\LocalAIImageEnhancer\logs`.

## AI model and profiles

The integrated model is `RealESRGAN_x4plus.pth`, downloaded from the
`xinntao/Real-ESRGAN` GitHub release `v0.1.0`. It is a general-purpose x4
restoration/super-resolution model. The app runs the learned 4x model and uses
Lanczos only to produce the requested 1x or 2x output size. This is not a
resize-only enhancement path.

Profiles apply documented amounts of model-output blending with a faithful
Lanczos-resized input: Natural 72%, Photo Restoration 90%, Maximum Detail 100%,
Portrait 75%, Artwork 95%, Text & Documents 60%, and Custom uses the strength
slider. Text enhancement does not recover missing text. The model can invent
plausible fine detail, so inspect results where exact identity, lettering, or
pixel-level fidelity matters.

Face restoration, GFPGAN, CodeFormer, SwinIR, HAT, generative repair, and
DirectML/Apple acceleration are not integrated. Portrait is a conservative
general-model profile, not a face-restoration model. No unimplemented feature
is presented as active.

The model is fetched only from an HTTPS URL pinned in the source and the
download is checked against its published byte size. The GitHub release
metadata currently provides no SHA-256 digest; the app records the downloaded
SHA-256 locally to detect later file corruption, but this sidecar is not an
independent trusted checksum. The model is not bundled. Check the model
upstream's current license and terms for your use case, especially commercial
redistribution.

## Hardware and memory behavior

The app reports GPU hardware separately from CUDA support. CUDA is selected
only when PyTorch reports it available. Initial tile size is conservative
(128 px on CUDA and 96 px on CPU in balanced mode); an out-of-memory error
reduces tile size to 64 px and retries, then retries on CPU if CUDA still fails.
The UI displays a warning when this fallback occurs. Maximum-quality mode uses
larger initial tiles and may therefore trigger a fallback on GPUs with limited
free memory. The input size limit is 200 megapixels.
Before inference, the pipeline estimates working RAM for the chosen scale and
rejects a job if its estimate exceeds 80% of currently available memory.

## Tests and diagnostics

Run offline unit tests with the active environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use synthetic images and a fake engine for pipeline tests; they do
not download weights or claim to validate real-model quality. To verify model
inference, download the model and process a small sample image in the UI.
Diagnostics and full tracebacks are written to the rotating log in the local
application data folder. Worker and inference events include a per-job
identifier so a failed request can be distinguished from subsequent retries.

## Project layout

```text
main.py
app/
  engine.py          Real-ESRGAN backend and tiled/OOM inference
  hardware.py        Hardware and CUDA detection
  image_io.py        Input normalization and safe output writing
  model_registry.py  Curated model metadata and downloader
  pipeline.py        Profiles and end-to-end image processing
  ui.py              PySide6 desktop interface
  worker.py          Background processing and model-download workers
install.ps1
run.ps1
run.bat
requirements.txt
requirements-ai.txt
tests/
```
