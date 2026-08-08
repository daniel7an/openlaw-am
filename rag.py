"""Grounded question answering over the indexed Armenian legal corpus.

One retrieval + one LLM call per question — no agent loops (hackathon budget, D-constraints).
Every claim must carry a [Հոդված N] citation; when the retrieved articles don't cover the
question the model is required to say so rather than improvise.

Usage:
    uv run python rag.py "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը"
"""
import sys
from collections import OrderedDict

from config import api_key, get, prompt

API_KEY = api_key()
BASE_URL = get("generation.base_url", env="OPENLAW_BASE_URL")
MODEL = get("generation.model", env="OPENLAW_MODEL")
TEMPERATURE = get("generation.temperature")
MAX_OUTPUT_TOKENS = get("generation.max_output_tokens")
TOP_K = get("retrieval.top_k")

SYSTEM = prompt("answer.system")
USER = prompt("answer.user")
REFUSAL_MARKER = prompt("answer.refusal_marker")


def client():
    from openai import OpenAI

    if not API_KEY:
        sys.exit("No API key. Put OPENROUTER_API_KEY=... in .env")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def retrieve(question: str, k: int = TOP_K, mode: str | None = None, alpha: float | None = None) -> list[dict]:
    """Hybrid search by default, then regroup parts back under their article.

    Chunks are sub-article sized (512-token model limit), so two parts of the same
    article can both hit. Presenting them as one [Հոդված N] block keeps the model
    from treating them as separate authorities.
    """
    from index import COLLECTION, connect, score_of, search

    client_ = connect()
    try:
        coll = client_.collections.get(COLLECTION)
        objs = search(coll, question, mode=mode, alpha=alpha, k=k)
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
        return list(grouped.values())
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


def answer(
    question: str,
    k: int = TOP_K,
    mode: str | None = None,
    alpha: float | None = None,
    model: str | None = None,
    articles: list[dict] | None = None,
) -> dict:
    """Generate a grounded answer. `model` / `articles` let eval reuse retrieval across models."""
    if articles is None:
        articles = retrieve(question, k, mode=mode, alpha=alpha)
    user_msg = USER.format(question=question, context=context_block(articles))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    use_model = model or MODEL

    # Reasoning models emit a hidden reasoning trace first; if it eats the whole
    # budget the answer comes back empty. Retry once with room rather than
    # returning nothing to the user.
    budget = MAX_OUTPUT_TOKENS
    for attempt in range(2):
        resp = client().chat.completions.create(
            model=use_model, messages=messages, max_tokens=budget, temperature=TEMPERATURE
        )
        text = resp.choices[0].message.content
        if text and text.strip():
            break
        budget *= 2
        print(
            f"  empty answer (finish_reason={resp.choices[0].finish_reason}, "
            f"reasoning burned the budget) — retrying with max_tokens={budget}",
            file=sys.stderr,
        )
    else:
        text = "Չհաջողվեց ստանալ պատասխան մոդելից: Փորձեք կրկին:"

    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "question": question,
        "answer": text,
        "retrieved": articles,
        "model": use_model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "total_tokens": usage.total_tokens,
        "finish_reason": resp.choices[0].finish_reason,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = answer(" ".join(sys.argv[1:]))
    print(f"\n{r['answer']}\n")
    print("— Աղբյուրներ —")
    for a in r["retrieved"]:
        print(f"  [{a['cite_id']}] {a['title'][:60]}  ({a['score']:.3f})  {a['url']}")
    print(
        f"\ntokens: {r['prompt_tokens']} in + {r['completion_tokens']} out "
        f"({r['reasoning_tokens']} reasoning) = {r['total_tokens']}  [{r['finish_reason']}]"
    )
