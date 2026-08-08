"""Grounded question answering over the indexed Armenian legal corpus.

One retrieval + one LLM call per question — no agent loops (hackathon budget, D-constraints).
Every claim must carry a [Հոդված N] citation; when the retrieved articles don't cover the
question the model is required to say so rather than improvise.

Usage:
    uv run python rag.py "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը"
"""
import logging
import sys
import time
from collections import OrderedDict

from config import api_key, get, prompt

# Module logger, deliberately without a handler: the CLI configures one in __main__ and
# the Streamlit app attaches its own sink to show these lines in the debug panel.
log = logging.getLogger("openlaw.rag")

API_KEY = api_key()
BASE_URL = get("generation.base_url", env="OPENLAW_BASE_URL")
MODEL = get("generation.model", env="OPENLAW_MODEL")
TEMPERATURE = get("generation.temperature")
MAX_OUTPUT_TOKENS = get("generation.max_output_tokens")
TOP_K = get("retrieval.top_k")

SYSTEM = prompt("answer.system")
USER = prompt("answer.user")
REFUSAL_MARKER = prompt("answer.refusal_marker")


def client(base_url: str | None = None, api_key: str | None = None):
    from openai import OpenAI

    # Self-hosted vLLM is unauthenticated, but the OpenAI client rejects an empty
    # key — any non-empty placeholder satisfies it. `api_key` is per-call so a caller
    # switching to an on-prem endpoint can avoid sending it the OpenRouter key.
    return OpenAI(base_url=base_url or BASE_URL, api_key=api_key or API_KEY or "EMPTY")


def retrieve(
    question: str,
    k: int = TOP_K,
    mode: str | None = None,
    alpha: float | None = None,
    collection: str | None = None,
) -> list[dict]:
    """Hybrid search by default, then regroup parts back under their article.

    Chunks are sub-article sized (512-token model limit), so two parts of the same
    article can both hit. Presenting them as one [Հոդված N] block keeps the model
    from treating them as separate authorities.

    `collection` names an index version to query directly, bypassing the alias. It is
    a read-only override for one call — the alias itself is never touched.
    """
    from index import COLLECTION, HYBRID_ALPHA, SEARCH_MODE, connect, score_of, search

    target = collection or COLLECTION
    log.info(
        "retrieve: collection=%s mode=%s alpha=%s k=%s q=%r",
        target, mode or SEARCH_MODE, HYBRID_ALPHA if alpha is None else alpha, k, question,
    )
    t0 = time.perf_counter()
    client_ = connect()
    try:
        coll = client_.collections.get(target)
        objs = search(coll, question, mode=mode, alpha=alpha, k=k)
        log.info("retrieve: %d chunks in %.2fs", len(objs), time.perf_counter() - t0)
        grouped: OrderedDict[str, dict] = OrderedDict()
        for o in objs:
            p = o.properties
            art = grouped.setdefault(
                p["cite_id"],
                {
                    "cite_id": p["cite_id"],
                    "article": p["article"],
                    "title": p["title"],
                    "url": p["url"],
                    "repealed": p.get("repealed"),
                    "has_repealed_parts": p.get("has_repealed_parts"),
                    "texts": [],
                    "score": score_of(o),
                },
            )
            art["texts"].append(p["text"])
        out = list(grouped.values())
        log.info(
            "retrieve: %d chunks regrouped into %d articles: %s",
            len(objs), len(out), ", ".join(f"{a['cite_id']}({a['score']:.3f})" for a in out),
        )
        return out
    finally:
        client_.close()


def context_block(articles: list[dict]) -> str:
    block = prompt("context.block")
    out = []
    for a in articles:
        flags = ""
        if a.get("repealed"):
            flags = prompt("context.repealed_flag")
        elif a.get("has_repealed_parts"):
            flags = prompt("context.partly_repealed_flag")
        out.append(
            block.format(
                label=f"Հոդված {a['article']}" if a["article"] else a["title"],
                flags=flags,
                title=a["title"],
                body="\n".join(a["texts"]),
            )
        )
    return "\n\n".join(out)


def generate(
    question: str,
    articles: list[dict],
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> dict:
    """The LLM half of `answer`, over articles that were already retrieved.

    Split out so a caller can report retrieval and generation as separate stages
    (app.py's progress panel) without paying for retrieval twice.
    """
    model = model or MODEL
    user_msg = USER.format(question=question, context=context_block(articles))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    use_model = model or MODEL

    # Reasoning models emit a hidden reasoning trace first; if it eats the whole
    # budget the answer comes back empty. Retry once with room rather than
    # returning nothing to the user.
    budget = max_tokens or MAX_OUTPUT_TOKENS
    for attempt in range(2):
        log.info(
            "generate: model=%s base_url=%s max_tokens=%d temp=%s context=%d chars attempt=%d",
            model, base_url or BASE_URL, budget, TEMPERATURE, len(user_msg), attempt + 1,
        )
        t0 = time.perf_counter()
        resp = client(base_url, api_key).chat.completions.create(
            model=model, messages=messages, max_tokens=budget, temperature=TEMPERATURE
        )
        text = resp.choices[0].message.content
        u = resp.usage
        finish = resp.choices[0].finish_reason
        details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        log.info(
            "generate: finish=%s in %.2fs — prompt %d + completion %d/%d (%.0f%% of budget) "
            "= %d total; completion splits %d reasoning + %d answer (%.0f%% reasoning)",
            finish, time.perf_counter() - t0,
            u.prompt_tokens, u.completion_tokens, budget, 100 * u.completion_tokens / budget,
            u.total_tokens, reasoning, u.completion_tokens - reasoning,
            100 * reasoning / u.completion_tokens if u.completion_tokens else 0,
        )
        # finish=length means the model was cut off mid-sentence. It is NOT retried below:
        # a truncated answer is still non-empty, so the loop exits and the fragment is
        # returned. Surfacing it loudly is what makes that visible instead of silent.
        if finish == "length":
            log.warning(
                "generate: TRUNCATED — hit max_tokens=%d with %d reasoning + %d answer tokens. "
                "Raise generation.max_output_tokens (currently %d) to give the answer room.",
                budget, reasoning, u.completion_tokens - reasoning, MAX_OUTPUT_TOKENS,
            )
        if text and text.strip():
            break
        budget *= 2
        log.warning(
            "generate: EMPTY answer (finish=%s, %d reasoning tokens burned the whole budget) "
            "— retrying with max_tokens=%d",
            finish, reasoning, budget,
        )
    else:
        text = "Could not get an answer from the model. Please try again."

    return {
        "question": question,
        "model": model,
        "answer": text,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "reasoning_tokens": reasoning,
        "answer_tokens": u.completion_tokens - reasoning,
        "total_tokens": u.total_tokens,
        "finish_reason": finish,
        "max_tokens": budget,
        "truncated": finish == "length",
    }


def answer(
    question: str,
    k: int = TOP_K,
    mode: str | None = None,
    alpha: float | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    collection: str | None = None,
) -> dict:
    """Retrieve, then answer. Backend is a parameter so callers (eval.py) can
    compare models without mutating process environment."""
    articles = retrieve(question, k, mode=mode, alpha=alpha, collection=collection)
    result = generate(
        question, articles, model=model, base_url=base_url, max_tokens=max_tokens, api_key=api_key
    )
    return {**result, "retrieved": articles}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    # Stage/timing lines go to stderr so stdout stays just the answer and is pipeable.
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    r = answer(" ".join(sys.argv[1:]))
    print(f"\n{r['answer']}\n")
    print("— Sources —")
    for a in r["retrieved"]:
        print(f"  [{a['cite_id']}] {a['title'][:60]}  ({a['score']:.3f})  {a['url']}")
    print(
        f"\ntokens: {r['prompt_tokens']} in + {r['completion_tokens']} out "
        f"({r['reasoning_tokens']} reasoning) = {r['total_tokens']}  [{r['finish_reason']}]"
    )
