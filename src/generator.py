"""
generator.py
Initializes the LLM and embedding model from config, and exposes
an async helper for table summarization.
"""

import logging
import os
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


# ─── Table summarization ──────────────────────────────────────────────────────

async def summarize_table(markdown_table: str) -> str:
    """
    Generate a semantic summary of a markdown table snippet.

    The caller is responsible for passing a representative sample
    (headers + a few rows) rather than the full table.
    """
    logger.debug("Summarizing table markdown (%s chars)", len(markdown_table))
    table_model = _env_cfg["llm_model"]
    max_tokens = _cfg.get("table_summary_max_tokens", 200)

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
                            "wiring pin assignments, operating limits, error codes) "
                            "and the key column names or values. Be specific and concise. "
                            "Do not include column names or sample values in your reply — "
                            "those will be appended separately."
                        ),
                    },
                    {"role": "user", "content": markdown_table},
                ],
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["message"]["content"].strip()
        logger.debug("Table summary generated (%s chars)", len(summary))
        return summary