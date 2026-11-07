import base64
import io
import os
import random
import re
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request

# from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

load_dotenv()

# * GROQ MODEL FIRST

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1/")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# HF_API_KEY = os.getenv("HF_API_KEY", "")
# HF_IMAGE_MODEL = os.getenv(
#     "HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers"
# )

POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_MODEL = os.getenv("POLLINATION_MODEL", "black-forest-labs/flux.1-schnell")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

app = FastAPI(title="AI Image Generate Gateway")
# * Limit form using too much
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# * LOGIN CHECK: ask Supabase if the token from the app is real
async def require_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please log in first.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=500, detail="SUPABASE_URL or SUPABASE_KEY not configured."
        )

    token = authorization[len("Bearer ") :]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}"},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Could not reach the login server.")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=401, detail="Login expired. Please log in again."
        )
    return resp.json()


# * DATA CONTRACT ( ENCHANTED PROMPT )


# * AH NIS USER INPUT
# * ... AH NIS MEAN NEY THA (REQUIRED)
class EnhancePromptRequest(BaseModel):
    raw_prompt: str = Field(..., min_length=1, max_length=500)
    style: str = Field(..., min_length=1, max_length=50)


# * AH NIS AI OUTPUT
class EnhancePromptResponse(BaseModel):
    enhanced_prompt: str
    style: str
    model_used: str


# * AH NIS USER INPUT
class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)


# * AH NIS AI OUTPUT
class GenerateImageRespone(BaseModel):
    image_base64: str  # * url for image
    model_used: str


# * AH NIS ENDPOINT FOR CALLING GROQ
# * SET DEFUALT PROMPT FOR AI
ENHANCER_SYSTEM_PROMPT = (
    "You are a prompt enginner for a text-to-image generation sys~tem,"
    "Given a short image idea and a style, rewrite it into one detailed,"
    "Vivid image-generate prompt, 40-80 words, no explanation, no quotes."
)


def contains_unsafe_content(text: str) -> bool:
    lowerd = text.lower()
    return any(keyword in lowerd for keyword in UNSAFE_KEYWORD)


@app.post("/api/ai/enhance-prompt", response_model=EnhancePromptResponse)
@limiter.limit("5/minute")
async def enhance_prompt(
    request: Request, body: EnhancePromptRequest, user: dict = Depends(require_user)
):
    if not LLM_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="LLM_API_KEY not configured yet please input your API key",
        )
    if contains_unsafe_content(body.raw_prompt):
        raise HTTPException(
            status_code=400, detail="this prompt violates our contant policy."
        )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": ENHANCER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Image idea:{body.raw_prompt}\nStyle:{body.style}",
            },
        ],
        "temperature": 0.8,
        "max_tokens": 600,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
        )

    # * get groq why is it error
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Groq error:{resp.text}")

    data = resp.json()
    enhaned = data["choices"][0]["message"]["content"].strip()

    return EnhancePromptResponse(
        enhanced_prompt=enhaned, style=body.style, model_used=f"grop:{LLM_MODEL}"
    )


# * InferenceClient using it so that shotent code i guess
# hf_client = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)

# * Safty filtre before it get denger
UNSAFE_KEYWORD = {
    "nude",
    "naked",
    "nsfw",
    "porn",
    "pornographic",
    "sex",
    "sexual",
    "erotic",
    "nipple",
    "pussy",
    "vagina",
    "penis",
    "genital",
    "breast",
    "topless",
    "gore",
    "gory",
    "kill",
    "murder",
    "suicide",
    "selfharm",
    "child abuse",
    "csam",
    "terrorist",
    "bomb",
}

NSFW_LABEL = "nsfw"


@app.post("/api/ai/generate-image", response_model=GenerateImageRespone)
@limiter.limit("5/minute")
# * AH NIS BOS POLLINATION
async def generate_image(
    request: Request, body: GenerateImageRequest, user: dict = Depends(require_user)
):
    if contains_unsafe_content(body.prompt):
        raise HTTPException(
            status_code=400, detail="this prompt violates our contant policy."
        )
    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="POLLINATIONS API KEY not configured yet please input your API key",
        )

    encoded_prompt = quote(body.prompt)
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
    params = {
        "model": POLLINATIONS_MODEL,
        "width": 1024,
        "height": 1024,
        "private": "true",
        "nologo": "true",
        "safe": "true",
        "seed": random.randint(0, 2147483647),
    }

    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}

    async with httpx.AsyncClient(timeout=60.0) as clinet:
        resp = await clinet.get(url, params=params, headers=headers)

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Pollinations error: {resp.text}")

    image64_base = base64.b64encode(resp.content).decode("utf-8")

    return GenerateImageRespone(
        image_base64=image64_base, model_used=f"Pollination:{POLLINATIONS_MODEL}"
    )
