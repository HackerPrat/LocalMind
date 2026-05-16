```
# LocalMind

> **Offline AI Trainer & Workbench** — one file, zero cloud, full control.

![LocalMind](LocalMind.png)

---

## What Is LocalMind?

LocalMind is a single-file Python application (`LocalMind.py`) that turns any machine into a self-contained AI training and inference workbench. Drop the file into a folder next to your model weights, run it, and it handles everything else — virtual environments, dependency resolution, llama.cpp compilation, and a pitch-black desktop GUI — without touching your system Python or requiring anything pre-installed beyond a base Python interpreter.

---

## Feature Overview

| Category | What LocalMind Does |
|---|---|
| **Bootstrap** | Auto-installs `uv`, manages isolated `ui` / `infer` / `train` venvs, compiles llama.cpp |
| **Model support** | GGUF, Safetensors, PyTorch `.bin/.pt/.pth/.ckpt`, ONNX, TFLite, CoreML, TF SavedModel, Keras `.h5`, HF model folders, PEFT adapter folders |
| **Document ingestion** | TXT, Markdown, PDF, DOCX, PPTX, XLSX, CSV, HTML, JSON, JSONL |
| **Image ingestion** | PNG, JPG, JPEG, WebP, BMP, TIFF, GIF (VLM captioning path) |
| **Training** | LoRA / QLoRA fine-tuning via PEFT + TRL, optional Unsloth acceleration, optional bitsandbytes 4-bit loading |
| **Quantization** | Q4_K_M, Q5_K_M, IQ4_XS+iMatrix, Q8_0, F16 copy — via `llama-quantize` |
| **iMatrix** | Build importance matrices from your own corpus with `llama-imatrix` |
| **Inference** | Real-time streaming chat backed by `llama-server` (OpenAI-compatible API) |
| **Context window** | Smart rolling-summary compaction — effectively infinite context in chat |
| **Export** | Adapter merge → HF model → GGUF conversion pipeline, one click |
| **Modes** | Simple mode (guided) and Advanced mode (every parameter exposed) |
| **Projects** | Named projects, each with its own source library, dataset, jobs, and exports |
| **OS support** | Windows, macOS (Intel + Apple Silicon), Linux (x86_64, ARM) |

---

## Requirements

| Hard requirement | Notes |
|---|---|
| **Python 3.10+** | Only the standard library is used at launch. LocalMind installs everything else itself. |
| **Internet (first run)** | Required to download `uv`, Python runtimes, pip packages, and llama.cpp binaries. Pass `--no-network` to operate fully offline after setup. |
| **~4 GB free disk** | For the full training stack (PyTorch, Transformers, etc.). Inference-only needs ~400 MB. |

Optional but beneficial:

- **NVIDIA GPU** — CUDA-accelerated training and inference; `nvidia-smi` is auto-detected.
- **AMD GPU** — ROCm path is auto-detected via `rocm-smi` or `/opt/rocm`.
- **Apple Silicon** — Metal acceleration used automatically by llama.cpp.

---

## Quick Start

```bash
# 1. Place LocalMind.py (and optionally LocalMind.png) in your working folder.
python LocalMind.py
```

On first launch LocalMind will:

1. Create a `.localmind/` state directory beside the script.
2. Download and install `uv` into an isolated bootstrap environment.
3. Create a `ui` venv (PySide6, requests, psutil, Pillow, packaging).
4. Create an `infer` venv (requests).
5. Detect and compile or download a llama.cpp backend matching your hardware.
6. Open the desktop GUI.

> **Tip — pre-install the full training stack at first launch:**
> ```bash
> python LocalMind.py --bootstrap-only --full-setup
> ```
> This installs the `train` venv (PyTorch, Transformers, PEFT, TRL, etc.) so the
> Training tab is immediately ready when the GUI opens.

---

## Directory Layout

After first run the working folder looks like this:

```
your-folder/
├── LocalMind.py              ← the application (never modified at runtime)
├── LocalMind.png             ← logo (optional, shown in GUI title bar)
├── imatrix.dat               ← optional pre-built importance matrix
└── .localmind/
    ├── manifest.json         ← environment and tool registry
    ├── logs/                 ← localmind.log, commands.log, bootstrap.log, fatal.log
    ├── envs/
    │   ├── ui/               ← PySide6 + GUI dependencies
    │   ├── infer/            ← lightweight inference dependencies
    │   └── train/            ← PyTorch training stack
    ├── tools/
    │   └── llama.cpp/        ← compiled or downloaded llama.cpp binaries
    ├── bootstrap/            ← uv bootstrap environment
    └── projects/
        └── <project-name>/
            ├── project.json      ← settings, chat history, active model
            ├── sources.sqlite3   ← versioned source library + FTS index
            ├── sources/          ← copies of ingested documents and images
            ├── datasets/         ← generated training JSONL
            ├── exports/          ← LoRA adapters, merged models, GGUFs
            └── jobs/             ← per-job scripts and artefacts
```

Everything LocalMind creates lives inside `.localmind/` and the project sub-directories. Your original model files and documents stay exactly where you put them.

---

## CLI Reference

```
python LocalMind.py [OPTIONS]
```

| Flag | What It Does |
|---|---|
| *(no flags)* | Normal launch — bootstrap as needed, then open the GUI. |
| `--advanced` | Open the GUI directly in Advanced mode. |
| `--project NAME` | Open or create a named project (default: `default`). |
| `--bootstrap-only` | Run the bootstrap check and exit without opening the GUI. |
| `--full-setup` | Used with `--bootstrap-only` — also installs the full training stack. |
| `--no-network` | Do not download or install anything; use already-installed envs only. |
| `--reset-envs` | Delete all managed venvs and rebuild from scratch on next launch. |
| `--python-version X.Y` | Request a specific Python version for managed venvs (default: `3.13`). Also readable from the `LOCALMIND_PYTHON` environment variable. |
| `--serve-local` | Start `llama-server` for the selected model without opening the GUI. |
| `--model PATH` | Model path for `--serve-local`. |

### Examples

```bash
# Standard GUI launch
python LocalMind.py

# Open the advanced UI for an existing named project
python LocalMind.py --advanced --project research-v2

# Bootstrap everything, then exit (useful in a CI/setup script)
python LocalMind.py --bootstrap-only --full-setup

# Use Python 3.12 for managed venvs instead of the default 3.13
python LocalMind.py --python-version 3.12

# Headless inference server (no GUI) — useful on a remote machine
python LocalMind.py --serve-local --model ./my-model.gguf

# Wipe and reinstall all managed venvs (useful after a major update)
python LocalMind.py --reset-envs
```

---

## The GUI — Panel by Panel

### Model Browser

- Lists every AI file or HF model folder found in the working directory.
- Shows format, file size, and detected metadata (architecture, quantization type, context length, chat template, etc.).
- GGUF metadata is parsed natively — no Python dependencies needed at scan time.
- Double-click any model to set it as the active inference target.

### Chat

A full streaming conversational interface backed by the embedded `llama-server`:

- Responses stream token-by-token.
- **Smart window compaction** — when the conversation grows long, LocalMind summarises older turns and injects the summary as a system message, keeping the context window within the model's limit without losing information. This gives an effectively unlimited conversation length for any model.
- The inference server starts and stops automatically; ports are allocated dynamically to avoid conflicts with other processes.
- Works with any GGUF that `llama-server` can load.

### Sources & Dataset

1. **Add sources** — browse or drag-and-drop documents and images into the project.
2. Each source is copied, SHA-256 hashed, versioned, chunked, and stored in a full-text-searchable SQLite database.
3. **Build dataset** — active chunks are assembled into a training JSONL file ready for LoRA fine-tuning.
4. Individual sources and chunks can be toggled active/inactive, edited, or deleted at any time without affecting the originals.

### Training *(requires the training stack — see `--full-setup`)*

**Simple mode** exposes only:

- Base model (HF Hub repo ID or local folder path)
- LoRA rank
- Number of epochs
- A single **Train** button

**Advanced mode** additionally exposes every tunable parameter:

| Parameter | Default |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj,v_proj` |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation steps | 4 |
| Max sequence length | 2048 |
| Load in 4-bit (bitsandbytes) | off |
| Sequence packing | off |
| Quantization preset | Fast Q4_K_M |
| Thread count | 4 |

Training runs in a background thread; live log output streams into the Jobs panel in real time.

### Export Pipeline

| Action | What It Produces |
|---|---|
| **Merge adapter** | Merges a PEFT adapter folder into its base model → full HF model folder |
| **Convert HF → GGUF** | Runs `convert_hf_to_gguf.py` from llama.cpp → `.f16.gguf` |
| **Quantize** | Runs `llama-quantize` with the chosen preset → quantized `.gguf` in `exports/` |
| **Build iMatrix** | Runs `llama-imatrix` against your dataset or corpus → `.imatrix.dat` |

The full round-trip is: **ingest documents → train LoRA → merge adapter → convert to GGUF → quantize → chat**.

### Jobs

- Every background job (training, quantization, conversion, iMatrix) appears in a live job list.
- Full stdout/stderr is streamed into the log view and written to `.localmind/logs/commands.log`.
- Jobs run concurrently where safe; the UI remains responsive throughout.

---

## Supported Model Formats

| Format | Detect | Train | Quantize | Chat |
|---|:---:|:---:|:---:|:---:|
| `.gguf` | ✅ | — | ✅ | ✅ |
| HF model folder (`config.json`) | ✅ | ✅ | via GGUF | — |
| PEFT adapter folder (`adapter_config.json`) | ✅ | — *(merge first)* | via GGUF | — |
| `.safetensors` | ✅ | ✅* | via GGUF | — |
| `.bin` / `.pt` / `.pth` / `.ckpt` | ✅ | ✅* | via GGUF | — |
| `.onnx` / `.tflite` / `.mlmodel` / `.pb` / `.h5` | ✅ metadata | — | — | — |

*\* When paired with a complete HF model folder (tokenizer + config).*

---

## Quantization Presets

| Preset | Type | iMatrix | Best For |
|---|---|:---:|---|
| Fast Q4_K_M | Q4_K_M | No | Everyday local chat — balanced size, speed, and quality |
| Quality Q5_K_M | Q5_K_M | No | Better retention than 4-bit with a moderate size increase |
| Compact IQ4_XS + iMatrix | IQ4_XS | Yes | Smallest footprint; best used with a pre-built importance matrix |
| Archive Q8_0 | Q8_0 | No | High-quality archival export with near-full precision |
| F16 Copy | COPY | No | Metadata/pruning pass — no tensor re-quantization |

---

## Managed Virtual Environments

LocalMind creates and maintains three isolated venvs. You never need to touch them.

| Env | Contents | When Created |
|---|---|---|
| `ui` | PySide6, requests, psutil, Pillow, packaging | Always, on first launch |
| `infer` | requests | Always, on first launch |
| `train` | torch, transformers, datasets, accelerate, peft, trl, sentencepiece, safetensors, gguf, pymupdf, docling + optional bitsandbytes, unsloth | On `--full-setup` or when Training tab is first used |

All envs are created via `uv` for maximum speed and reproducibility. The Python version for each env defaults to `3.13` and can be changed with `--python-version`.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `LOCALMIND_PYTHON` | Default Python version for managed venvs (e.g. `3.12`) |
| `LOCALMIND_UV` | Path to a specific `uv` executable to use instead of the auto-detected one |
| `LOCALMIND_CUDA_ROOT` | Path to CUDA toolkit root (also respects `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`) |

---

## llama.cpp Backend Selection

At bootstrap, LocalMind probes the system and selects the best available llama.cpp backend automatically:

| Hardware Detected | Backend Used |
|---|---|
| NVIDIA GPU (`nvidia-smi` present) | CUDA build |
| AMD GPU (`rocm-smi` or `/opt/rocm`) | ROCm build |
| Apple Silicon (macOS + ARM) | Metal build |
| Everything else | CPU build |

Pre-built binaries are downloaded when available. If no pre-built binary matches, LocalMind falls back to building from source inside `.localmind/tools/`.

---

## Logging

All activity is written to `.localmind/logs/`:

| File | Contents |
|---|---|
| `localmind.log` | General application events |
| `bootstrap.log` | Environment setup and dependency installation |
| `commands.log` | Every subprocess command and its full output |
| `fatal.log` | Unhandled exceptions with full tracebacks |

---

## Project Files Preserved at Runtime

The following files in the working directory are never modified, moved, or deleted by LocalMind:

```
LocalMind.png
LocalMind.py
prompt.txt
imatrix.dat
*.gguf  (your model files)
```

---

## Offline Operation

After the initial setup, LocalMind can run entirely without internet access:

```bash
python LocalMind.py --no-network
```

This prevents any download or install attempt. Existing managed venvs and llama.cpp binaries are used as-is.

---

## Troubleshooting

**The GUI does not open.**
Run `python LocalMind.py --bootstrap-only` and inspect the terminal output. The `ui` venv may have failed to install PySide6. Check `.localmind/logs/bootstrap.log` for details.

**Training fails with "Train environment is not installed".**
Run `python LocalMind.py --bootstrap-only --full-setup` to install the training stack, then relaunch normally.

**`llama-server` does not start.**
Check `.localmind/logs/commands.log`. The llama.cpp binaries may not match your OS/GPU. Delete `.localmind/tools/` and relaunch to trigger a fresh backend detection and download.

**Wrong Python version in a managed venv.**
Run `python LocalMind.py --reset-envs` to wipe all managed venvs. They will be rebuilt with the correct version on next launch.

**Port conflict on inference server.**
LocalMind scans ports `8765–8865` and picks the first free one automatically. If all ports in that range are occupied, it falls back to a kernel-assigned port. No manual configuration is needed.

---

## License

MIT License

Copyright (c) 2026 #@CK@

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
