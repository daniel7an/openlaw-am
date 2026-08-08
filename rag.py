"""Grounded question answering over the indexed Armenian legal corpus.

One retrieval + one LLM call per question — no agent loops (hackathon budget, D-constraints).
Every claim must carry a [Հոդված N] citation; when the retrieved articles don't cover the
question the model is required to say so rather than improvise.

Usage:
    uv run python rag.py "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը"
"""
import os
import sys
from collections import OrderedDict

from dotenv import load_dotenv

load_dotenv()

# The team's .env carries the raw OpenRouter name; accept both spellings.
API_KEY = os.getenv("OPENLAW_API_KEY") or os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENLAW_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENLAW_MODEL", "deepseek/deepseek-v4-pro")

TOP_K = 8
MAX_OUTPUT_TOKENS = 800

SYSTEM = """Դու Հայաստանի Հանրապետության օրենսդրության վերաբերյալ տեղեկատվական օգնական ես:

ԿԱՆՈՆՆԵՐ՝
1. Պատասխանիր ՄԻԱՅՆ հայերենով:
2. Օգտագործիր ԲԱՑԱՌԱՊԵՍ ներքևում տրված հոդվածները: Մի՛ հենվիր սեփական հիշողությանդ վրա:
3. ՅՈՒՐԱՔԱՆՉՅՈՒՐ պնդում պետք է ուղեկցվի աղբյուրի հղումով՝ [Հոդված N] ձևաչափով:
4. Եթե տրված հոդվածները բավարար չեն հարցին պատասխանելու համար, ուղղակիորեն ասա՝
   «Տրամադրված հոդվածները բավարար չեն այս հարցին պատասխանելու համար», և նշիր, թե
   ՀՀ օրենսդրության որ բնագավառում կամ որ ակտում պետք է փնտրել պատասխանը:
   Այս դեպքում ՄԻ՛ հորինիր հոդվածի համարներ:
5. Եթե հոդվածը կամ դրա մասն ուժը կորցրել է, պարտադիր նշիր դա:
6. Մի՛ տուր իրավաբանական խորհրդատվություն. տուր տեղեկատվություն՝ հղումներով:
7. Պատասխանը թող լինի հակիրճ և կառուցվածքային:"""


def client():
    from openai import OpenAI

    if not API_KEY:
        sys.exit("No API key. Put OPENROUTER_API_KEY=... in .env")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Vector search, then regroup parts back under their article.

    Chunks are sub-article sized (512-token model limit), so two parts of the same
    article can both hit. Presenting them as one [Հոդված N] block keeps the model
    from treating them as separate authorities.
    """
    from index import COLLECTION, connect, encoder

    vec = encoder().encode([f"query: {question}"], normalize_embeddings=True)[0]
    client_ = connect()
    try:
        coll = client_.collections.get(COLLECTION)
        res = coll.query.near_vector(
            near_vector=list(map(float, vec)), limit=k, return_metadata=["distance"]
        )
        grouped: OrderedDict[str, dict] = OrderedDict()
        for o in res.objects:
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
                    "score": 1 - o.metadata.distance,
                },
            )
            art["texts"].append(p["text"])
        return list(grouped.values())
    finally:
        client_.close()


def context_block(articles: list[dict]) -> str:
    out = []
    for a in articles:
        label = f"Հոդված {a['article']}" if a["article"] else a["title"]
        flags = ""
        if a.get("repealed"):
            flags = " [ՈՒԺԸ ԿՈՐՑՐԵԼ Է]"
        elif a.get("has_repealed_parts"):
            flags = " [ՈՒՇԱԴՐՈՒԹՅՈՒՆ՝ հոդվածի որոշ մասեր ուժը կորցրել են]"
        body = "\n".join(a["texts"])
        out.append(f"=== [{label}]{flags} {a['title']}\n{body}")
    return "\n\n".join(out)


def answer(question: str, k: int = TOP_K) -> dict:
    articles = retrieve(question, k)
    prompt = (
        f"ՀԱՐՑ՝ {question}\n\n"
        f"ՀԱՍԱՆԵԼԻ ՀՈԴՎԱԾՆԵՐ՝\n\n{context_block(articles)}\n\n"
        f"Պատասխանիր հարցին՝ հենվելով բացառապես վերոնշյալ հոդվածների վրա, "
        f"յուրաքանչյուր պնդման կողքին նշելով [Հոդված N]:"
    )
    resp = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.1,
    )
    usage = resp.usage
    return {
        "question": question,
        "answer": resp.choices[0].message.content,
        "retrieved": articles,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = answer(" ".join(sys.argv[1:]))
    print(f"\n{r['answer']}\n")
    print("— Աղբյուրներ —")
    for a in r["retrieved"]:
        print(f"  [{a['cite_id']}] {a['title'][:60]}  ({a['score']:.3f})  {a['url']}")
    print(f"\ntokens: {r['prompt_tokens']} in + {r['completion_tokens']} out = {r['total_tokens']}")
