"""
generator.py
Initializes the LLM and embedding model from config, and exposes
async helpers for vision-based image description and table summarization.
"""

import os
import base64
import logging
import httpx
from dotenv import load_dotenv

from llama_index.llms.vllm import Vllm
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

from src.utils import load_llm_config

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_cfg = load_llm_config()
_env = os.getenv("APP_ENV")

if _env not in ("dev", "prod"):
    raise ValueError("APP_ENV must be 'dev' or 'prod'.")

_env_cfg = _cfg[_env]
logger.info("Initializing generator for APP_ENV=%s", _env)

# ─── Embedding model ─────────────────────────────────────────────────────────

embed_model = HuggingFaceEmbedding(model_name=_env_cfg["embedding_model"])
logger.info("Embedding model initialized: %s", _env_cfg["embedding_model"])

# ─── LLM ─────────────────────────────────────────────────────────────────────

if _env == "dev":
    llm = Groq(model=_env_cfg["llm_model"], api_key=GROQ_API_KEY)
else:
    vllm_cfg = _env_cfg.get("vllm", {})
    llm = Vllm(
        model=_env_cfg["llm_model"],
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        max_new_tokens=vllm_cfg.get("max_new_tokens", 512),
        vllm_kwargs={
            "swap_space": vllm_cfg.get("swap_space", 1),
            "gpu_memory_utilization": vllm_cfg.get("gpu_memory_utilization", 0.5),
        },
    )
logger.info("LLM initialized: %s", _env_cfg["llm_model"])

Settings.llm = llm
Settings.embed_model = embed_model


# ─── Vision: image description ───────────────────────────────────────────────

async def describe_image(image_path: str) -> str:
    """Call a vision-capable LLM to generate a semantic description of an image."""
    logger.debug("Describing image: %s", image_path)
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    vision_model = _cfg.get("vision_model", "meta-llama/llama-4-scout-17b-16e-instruct")
    max_tokens = _cfg.get("vision_max_tokens", 150)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": vision_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are analyzing an image from an equipment manual. "
                                "In 1-2 sentences, describe what this image shows "
                                "(e.g. wiring diagram, component location, warning label, "
                                "installation step, exploded-view diagram, etc.) and note "
                                "any key labels, part numbers, or values visible."
                            ),
                        },
                    ],
                }],
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        result = response.json()
        logger.debug("Image description generated for %s", image_path)
        return result["choices"][0]["message"]["content"].strip()


# ─── Table summarization ──────────────────────────────────────────────────────

async def summarize_table(markdown_table: str) -> str:
    """Use the text LLM to generate a semantic summary of a markdown table."""
    logger.debug("Summarizing table markdown (%s chars)", len(markdown_table))
    table_model = _env_cfg["llm_model"]
    max_tokens = _cfg.get("table_summary_max_tokens", 150)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": table_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing a table from an equipment manual. "
                            "Summarize what this table contains in 1-2 sentences: "
                            "what kind of data it holds (e.g. torque specs, part numbers, "
                            "wiring pin assignments, operating limits, error codes, etc.) "
                            "and the key column names or values. Be specific and concise."
                        ),
                    },
                    {"role": "user", "content": markdown_table},
                ],
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        result = response.json()
        logger.debug("Table summary generated")
        return result["choices"][0]["message"]["content"].strip()