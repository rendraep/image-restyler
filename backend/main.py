import base64
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from llm_service import parse_style_request
from sd_service import transform_image

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="AI Image Style Transformer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_image(
    image: UploadFile = File(..., description="Input image (JPG/PNG/WebP)"),
    prompt: str = Form(..., description="Style transformation description"),
):
    """
    Full pipeline:
      1. Ollama LLM parses the user's style request → sd_prompt, negative_prompt, strength
      2. Stable Diffusion img2img applies the style transformation
    """
    image_data = await image.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Step 1 — LLM prompt parsing
    try:
        parsed = parse_style_request(prompt)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"LLM parsing failed: {exc}")

    # Step 2 — Style transformation
    try:
        result_bytes = await transform_image(
            image_data=image_data,
            sd_prompt=parsed["sd_prompt"],
            negative_prompt=parsed["negative_prompt"],
            strength=parsed["strength"],
            style_name=parsed["style_name"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {exc}")

    return {
        "success": True,
        "result_image": base64.b64encode(result_bytes).decode(),
        "parsed": parsed,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
