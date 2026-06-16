# Agent Collaboration Log — AI Image Style Transformer

**Project:** AI Image Style Transformer  
**Repository:** https://github.com/rendraep/image-restyler  
**Agent:** Claude (claude-sonnet-4-6) via Claude Code CLI  

---

## Phase 1: Ideation and Planning

### Objective
Build a web application that uses AI to transform images. The initial concept was to replace text within an image using OCR and generative AI. The project later pivoted to a broader image style transformation tool (photo → cartoon, anime, oil painting, etc.).

### Key Prompts
> *"in this folder you have been build an app to change text of an image. can you tell me how to run it?"*

> *"i want to change my plan from change text into image changer. for example change a photo to become cartoon."*

### Tools Applied
- Claude Code CLI to explore the existing project structure (`ls`, `Read`)
- Identified existing stack: FastAPI backend, vanilla JS frontend, EasyOCR, Anthropic Claude API, Stable Diffusion inpainting

### Outcome
**Initial proposed stack (text replacement):**
| Component | Technology |
|---|---|
| OCR | EasyOCR |
| LLM | Anthropic Claude API |
| Image generation | Stable Diffusion inpainting |
| Backend | FastAPI |
| Frontend | HTML / CSS / JS |

**Revised stack (style transformation):**
| Component | Technology |
|---|---|
| LLM | Ollama (local, offline) |
| Image generation | Stable Diffusion XL img2img |
| SD Model | Lykon/DreamShaper XL (HuggingFace) |
| Backend | FastAPI |
| Frontend | HTML / CSS / JS |

The pivot was motivated by the desire for a more general-purpose tool and the availability of a local GPU (RTX 5070 Ti, 16GB VRAM).

---

## Phase 2: Architecture Design and Task Decomposition

### System Architecture

```
Browser (HTML/CSS/JS)
        │  POST /api/process  (multipart: image + prompt)
        ▼
FastAPI Backend (main.py)
        │
        ├─► llm_service.py  ──► Ollama API (localhost:11434)
        │       └─ Returns: sd_prompt, negative_prompt, strength, style_name
        │
        └─► sd_service.py   ──► HuggingFace Diffusers (SDXL img2img)
                └─ Returns: transformed image bytes (PNG)
```

### API Data Exchange Format

**Request:** `POST /api/process`  
```
Content-Type: multipart/form-data
- image: <file>   (JPG / PNG / WebP)
- prompt: <str>   (e.g. "make this a cartoon")
```

**Response:** `application/json`
```json
{
  "success": true,
  "result_image": "<base64-encoded PNG>",
  "parsed": {
    "style_name": "Cartoon",
    "sd_prompt": "cartoon style illustration, ...",
    "negative_prompt": "blurry, low quality, ...",
    "strength": 0.58
  }
}
```

### Task Decomposition
| Task | File | Description |
|---|---|---|
| LLM prompt parsing | `llm_service.py` | Ollama call → structured JSON |
| SDXL img2img inference | `sd_service.py` | Load model, resize, run pipeline |
| API routing | `main.py` | FastAPI endpoints, CORS, static serving |
| UI upload & preview | `frontend/app.js` | Drag & drop, file reader |
| UI progress & results | `frontend/app.js` | Step indicators, comparison view |
| Styling | `frontend/style.css` | Dark theme, two-column viewport layout |

---

## Phase 3: Code Generation and Implementation

### 3.1 Replacing Anthropic Claude API with Ollama

**Problem:** The original `llm_service.py` used the Anthropic API which required a paid API key.  
**Prompt:**
> *"how about we use ollama to get offline llm model?"*

**Agent action:** Rewrote `llm_service.py` to call the Ollama `/api/chat` REST endpoint using `httpx` (already in requirements), removing the `anthropic` dependency entirely.

**Key implementation:**
```python
response = httpx.post(
    f"{_OLLAMA_BASE_URL}/api/chat",
    json={
        "model": _OLLAMA_MODEL,
        "messages": [...],
        "stream": False,
        "think": False,          # disable chain-of-thought for speed
        "options": {"temperature": 0},
    },
    timeout=60.0,
)
```

---

### 3.2 Bottleneck: Non-existent Package `difflib2`

**Error:**
```
ERROR: Could not find a version that satisfies the requirement difflib2>=1.0.0
```

**Resolution:** The agent identified that the code actually imported Python's built-in `difflib` (standard library), so `difflib2` in `requirements.txt` was a mistake. Removed the line — no replacement needed.

---

### 3.3 Bottleneck: Slow Ollama Response (Thinking Model)

**Problem:** The LLM step took 20–30 seconds because `qwen3.5:9b` is a reasoning model that generates long `<think>...</think>` blocks before answering.

**Resolution:** Added `"think": False` to the Ollama API request to disable chain-of-thought mode. Response time dropped to 1–3 seconds. Also added a regex strip as a safety fallback:
```python
raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
```

---

### 3.4 Switching from SD 1.5 to SDXL

**Problem:** `runwayml/stable-diffusion-v1-5` produced weak style changes — images barely differed from the original.

**Agent analysis:** Identified two root causes:
1. No `.env` file existed → `USE_LOCAL_SD` was never `true` → app was silently using the PIL fallback (just brightness/contrast adjustments)
2. SD 1.5 is a general-purpose model, not optimized for dramatic style transfer

**Resolution:**
- Created `.env` with `USE_LOCAL_SD=true`
- Switched to `Lykon/dreamshaper-xl-1-0` (SDXL) for higher quality
- Upgraded pipeline from `StableDiffusionImg2ImgPipeline` to `StableDiffusionXLImg2ImgPipeline`
- Updated resize logic to SDXL's requirement of multiples of 64, at 1024px

```python
from diffusers import StableDiffusionXLImg2ImgPipeline

_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    _SD_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
_pipe.enable_model_cpu_offload()
```

---

### 3.5 Bottleneck: CUDA Out of Memory

**Error:**
```
CUDA out of memory. Tried to allocate 288.00 MiB.
Process 7208 has 6.51 GiB memory in use.  ← Ollama (qwen3.5:9b)
This process has 7.92 GiB memory in use.  ← SDXL
```

**Root cause:** Ollama loads `qwen3.5:9b` on the GPU (~6.5GB VRAM) and keeps it resident. SDXL requires ~7.5GB, totalling ~14GB — exceeding the RTX 5070 Ti's 15.46GB usable VRAM.

**Resolution:** Enabled `enable_model_cpu_offload()` on the SDXL pipeline. This moves model weights to RAM and streams them to GPU layer-by-layer during inference, reducing SDXL VRAM usage from ~7.5GB to ~3–4GB. Also added:
- `torch.cuda.empty_cache()` before each inference call
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `.env` to reduce memory fragmentation

---

### 3.6 Strength Tuning

**Problem:** Results either lost too much originality (too high strength) or showed no visible style change (too low strength).

**Iterative tuning process:**

| Iteration | Range | Outcome |
|---|---|---|
| Initial | 0.5 – 0.85 | Too dramatic, faces lost |
| Reduce 1 | 0.35 – 0.65 | Better, still losing detail |
| Reduce 2 | 0.25 – 0.50 | Good balance |
| Reduce 3 | 0.15 – 0.35 | Too subtle, style barely visible |
| Final | 0.40 – 0.65 | Best balance of style and originality |

The `strength` parameter is passed by the LLM as part of its JSON response, constrained by a hard clamp in code:
```python
parsed["strength"] = max(0.4, min(0.65, float(parsed["strength"])))
```

---

## Phase 4: Interface Encapsulation and Finalization

### 4.1 Frontend Framework Choice

The agent implemented the UI using **Vanilla HTML/CSS/JS** (no framework), served directly by FastAPI via `StaticFiles`. This avoids build tooling complexity while keeping the UI fully self-contained.

### 4.2 UI Features Implemented

| Feature | Implementation |
|---|---|
| Drag & drop upload | `dragover` / `drop` events on upload area |
| Image preview | `FileReader.readAsDataURL` |
| Animated progress steps | CSS `@keyframes pulse-ring` + JS class toggling |
| Side-by-side comparison | CSS Grid `1fr auto 1fr` |
| Download result | Dynamic `<a>` element with base64 data URL |

### 4.3 Bottleneck: Layout Overflow (Scrolling Page)

**Problem:** The stacked vertical layout caused the page to scroll, especially when results appeared below the input panel.

**Resolution:** Restructured to a two-column viewport-locked layout:
```css
html, body { height: 100%; overflow: hidden; }

.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.main {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;  /* input | results */
  min-height: 0;
}
```
Left column: upload + prompt (always visible). Right column: progress → results, appearing in place without pushing content.

### 4.4 Codebase Cleanup Before Publishing

**Prompt:**
> *"okay, i think it is finish. now i plan to upload the code to my github. can you check it again, remove unnecessary code, and add anything needed."*

**Agent actions:**
- Deleted `image_processor.py` — entirely unused after the pivot from text replacement to style transfer (contained EasyOCR logic, masking, OCR region detection)
- Removed `easyocr` and `numpy` from `requirements.txt` — leftover dependencies from the old pipeline
- Created `.gitignore` — excludes `.env`, `__pycache__`, `.vscode/`, venvs
- Created `backend/.env.example` — documents all required environment variables for new contributors

### 4.5 Documentation

The agent generated `README.md` covering setup instructions, configuration reference, and tech stack. This file (`AGENT_LOG.md`) was also generated by the agent to document the full collaboration workflow.

---

## Summary of Technical Bottlenecks Resolved

| # | Bottleneck | Resolution |
|---|---|---|
| 1 | `difflib2` package not found | Removed — Python's built-in `difflib` was already being used |
| 2 | Anthropic API key not available | Replaced with local Ollama via `httpx` |
| 3 | LLM response too slow (20–30s) | Disabled thinking mode with `"think": False` |
| 4 | PIL fallback instead of SD running | Missing `.env` file — created with `USE_LOCAL_SD=true` |
| 5 | Weak style transfer (SD 1.5) | Switched to DreamShaper XL (SDXL) |
| 6 | CUDA out of memory | Enabled `enable_model_cpu_offload()` on SDXL pipeline |
| 7 | Style too strong / loses originality | Iterative strength tuning, final range 0.40–0.65 |
| 8 | Page scrolling on results | Restructured to two-column `100vh` locked layout |
