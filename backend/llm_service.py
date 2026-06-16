import os
import json
import re
import httpx

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")


def parse_style_request(user_prompt: str) -> dict:
    """
    Use a local Ollama model to parse a natural-language style transformation request
    and return structured data for Stable Diffusion img2img.
    """
    system = (
        "You are a precise assistant that converts natural-language image style requests "
        "into Stable Diffusion prompts. Always respond with valid JSON only — no markdown, "
        "no explanation, just the JSON object."
    )

    user_message = f"""Convert the user's style transformation request into Stable Diffusion parameters.

User request: "{user_prompt}"

Return a JSON object with exactly these keys:
- "style_name": Short human-readable name for the style (e.g. "Cartoon", "Anime", "Oil Painting").
- "sd_prompt": Detailed positive Stable Diffusion prompt describing the target style. Include style keywords, medium, lighting, and quality boosters like "masterpiece, best quality, highly detailed".
- "negative_prompt": Negative prompt to avoid unwanted artifacts (e.g. "blurry, low quality, deformed, ugly, watermark").
- "strength": A float between 0.4 and 0.65 — how strongly to apply the transformation. Use 0.55-0.65 for dramatic styles like cartoon/anime, and 0.4-0.5 for subtle changes like color grading or soft illustration.

Example output for "make it look like a cartoon":
{{
  "style_name": "Cartoon",
  "sd_prompt": "cartoon style illustration, vibrant colors, bold outlines, cel shading, animated movie style, masterpiece, best quality, highly detailed",
  "negative_prompt": "realistic, photorealistic, blurry, low quality, deformed, ugly, watermark, grainy",
  "strength": 0.58
}}"""

    response = httpx.post(
        f"{_OLLAMA_BASE_URL}/api/chat",
        json={
            "model": _OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text[:400]}")

    raw = response.json()["message"]["content"].strip()

    # Strip <think> blocks (some models include reasoning traces)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)

    required_keys = {"style_name", "sd_prompt", "negative_prompt", "strength"}
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}")

    parsed["strength"] = max(0.4, min(0.65, float(parsed["strength"])))
    return parsed
