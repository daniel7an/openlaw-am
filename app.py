"""Streamlit demo UI for openlaw-am.

The point of the layout is auditability: the answer sits next to the exact articles
it was grounded in, each with a similarity score and a live arlis.am link, so a
claim can be checked against the source in one click.

Usage:
    uv run streamlit run app.py
"""
import logging
import re
import time
from collections import deque

import streamlit as st

import rag
from config import get

st.set_page_config(page_title="openlaw-am — RA legal assistant", page_icon="⚖️", layout="wide")

# The interface is English; the questions themselves stay Armenian because the corpus,
# the retrieval and the answers are Armenian.
EXAMPLES = [
    "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը",
    "Ի՞նչ դեպքերում կարող է գործատուն լուծել աշխատանքային պայմանագիրը",
    "Քանի՞ օր է տարեկան արձակուրդի նվազագույն տևողությունը",
    "Ինչպե՞ս գրանցել ամուսնություն Հայաստանում",  # Family Code, in no index -> should refuse
]


BACKENDS = get("generation.backends")


@st.cache_resource
def log_sink() -> deque:
    """Ring buffer fed by the root logger, rendered in the debug panel.

    cache_resource, not session_state: the handler must be attached exactly once per
    process, and the history has to survive Streamlit's rerun-on-every-interaction.
    Hooking the ROOT logger (not just openlaw.*) is deliberate — httpx's one-line-per-
    request output is what tells you whether the model endpoint was actually reached.
    """
    buf: deque = deque(maxlen=1000)

    class Sink(logging.Handler):
        def emit(self, record):
            try:
                buf.append(self.format(record))
            except Exception:  # a broken log line must never take down the page
                pass

    handler = Sink()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Weaviate's gRPC/batch chatter is per-object and would bury everything else.
    logging.getLogger("weaviate").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return buf


LOGS = log_sink()


@st.cache_resource(show_spinner="Loading the retrieval model…")
def warm():
    """Load the embedding model once, not on every rerun."""
    from index import encoder

    encoder()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def index_versions() -> tuple[list[dict], str | None]:
    """Index versions that exist in Weaviate right now, plus the alias target.

    Counts come from Weaviate (ground truth) and the per-document breakdown from the
    build manifest, so a version built outside this checkout still lists, just without
    its document mix.
    """
    import json
    from pathlib import Path

    from index import active_version, connect, versions

    path = Path(get("paths.index_manifest"))
    manifest = {e["version"]: e for e in json.loads(path.read_text())} if path.exists() else {}

    client = connect()
    try:
        out = []
        for v in versions(client):
            entry = manifest.get(v, {})
            out.append(
                {
                    "name": v,
                    "chunks": client.collections.get(v).aggregate.over_all(total_count=True).total_count,
                    "articles": entry.get("articles"),
                    "documents": entry.get("documents", {}),
                }
            )
        return out, active_version(client)
    finally:
        client.close()


@st.cache_data(ttl=60, show_spinner=False)
def reachable(base_url: str) -> bool:
    """Cheap liveness probe, so a dead SSH tunnel shows up before you click Ask
    rather than as a traceback in front of an audience."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3)
    except urllib.error.HTTPError:
        return True  # it answered; an auth error still means the endpoint is up
    except Exception:
        return False
    return True


def cite_links(answer: str, articles: list[dict]) -> str:
    """Turn [Հոդված N] into a link to that article on arlis.am."""
    url_by_article = {a["article"]: a["url"] for a in articles if a["article"]}

    def repl(m):
        num = m.group(1)
        url = url_by_article.get(num)
        return f"[[Հոդված {num}]]({url})" if url else m.group(0)

    return re.sub(r"\[Հոդված\s+([0-9]+(?:\.[0-9]+)?)\]", repl, answer)


def drain(container, seen: int) -> int:
    """Write the log lines produced since `seen` into `container`; return the new mark.

    Called between stages: the status panel is the only thing on screen while a query
    runs, so the lines a stage emitted belong next to that stage. The panel at the
    bottom of the page keeps the full history.
    """
    lines = list(LOGS)[seen:]
    if lines:
        container.code("\n".join(lines), language="log")
    return len(LOGS)


def log_panel() -> None:
    """Full log history. Called on the failure path too — once a query has errored the
    log is the only thing left on screen that explains why."""
    if not show_log:
        return
    st.divider()
    st.caption(f"Debug log — {len(LOGS)} lines, newest last")
    st.code("\n".join(LOGS) or "(nothing logged yet)", language="log", height=280)


st.caption(
    "An open, verifiable legal assistant grounded in ARLIS. "
    "Every statement comes with a link to the article it rests on. "
    "**This is not legal advice.**"
)

with st.sidebar:
    st.subheader("Model")
    # Default to whichever backend matches config/env, so the picker starts where the
    # CLI and eval.py would have started.
    default_model = get("generation.model", env="OPENLAW_MODEL")
    names = [b["name"] for b in BACKENDS]
    choice = st.selectbox(
        "Answering model",
        names,
        index=next((i for i, b in enumerate(BACKENDS) if b["model"] == default_model), 0),
        help="Retrieval, corpus and prompts are identical across backends — only the answering model changes.",
    )
    backend = BACKENDS[names.index(choice)]
    st.caption(backend["note"])
    max_out = st.slider(
        "Max output tokens", 500, 8000, rag.MAX_OUTPUT_TOKENS, 250,
        help="Caps reasoning + answer COMBINED, not just the answer. A reasoning model can "
             "spend most of this thinking before it writes a word — if answers cut off "
             "mid-word, this is the knob. On an empty answer the code retries once at 2×.",
    )
    if max_out < rag.MAX_OUTPUT_TOKENS:
        st.caption(f"⚠️ Below the configured default of {rag.MAX_OUTPUT_TOKENS:,} — expect truncation.")
    if not reachable(backend["base_url"]):
        st.error(f"`{backend['base_url']}` is not responding.")
        if backend.get("hint"):
            st.code(backend["hint"], language="bash")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Articles to retrieve (top-k)", 3, 15, rag.TOP_K)
    mode = st.radio(
        "Retrieval mode",
        ["hybrid", "vector", "bm25"],
        format_func={"hybrid": "Hybrid (BM25 + vector)", "vector": "Vector", "bm25": "BM25 (keywords)"}.get,
        help="Measured on 18 questions (hit@3): hybrid 72%, vector 67%, BM25 39%",
    )
    alpha = st.slider(
        "alpha (1 = vector, 0 = BM25)", 0.0, 1.0, get("retrieval.alpha"), 0.05,
        disabled=mode != "hybrid",
    )
    st.divider()
    st.subheader("Corpus")
    indexes, alias_target = index_versions()
    if not indexes:
        st.error("No index versions found in Weaviate. Build one: `uv run python index.py --name all_codes`")
        st.stop()

    prefix = get("weaviate.collection_prefix")
    wanted = get("weaviate.ui_default", env="OPENLAW_UI_COLLECTION") or alias_target
    index_names = [i["name"] for i in indexes]
    picked = st.selectbox(
        "Index version",
        index_names,
        index=index_names.index(wanted) if wanted in index_names else 0,
        format_func=lambda n: f"{n.removeprefix(prefix)}{' ★' if n == alias_target else ''}",
        help="Queried directly. Switching here does not repoint the `Article` alias — the CLI and eval.py are unaffected. ★ marks the current alias target.",
    )
    sel = indexes[index_names.index(picked)]
    docs = sel["documents"]

    st.markdown(
        f"- `{sel['name']}` — {sel['chunks']:,} chunks"
        + (f", {sel['articles']:,} articles" if sel["articles"] else "")
        + "\n"
        + f"- Text: ARLIS, in force as of 10.07.2026\n"
        + f"- Model: `{backend['model']}`\n"
        + f"- Embeddings: `{get('embedding.model', env='OPENLAW_EMBED_MODEL').split('/')[-1]}`"
    )
    if docs:
        with st.expander(f"{len(docs)} document(s) in this index"):
            st.markdown(
                "\n".join(f"- `{s}` — {n:,} chunks" for s, n in sorted(docs.items(), key=lambda kv: -kv[1]))
            )
    st.info(
        "The corpus contains only the Labor Code. "
        "For questions outside that scope the system is expected to refuse to answer."
        if set(docs) <= {"labor-code"}
        else "Questions outside the documents listed above are expected to be refused."
    )

    st.divider()
    st.subheader("Debug")
    show_log = st.toggle("Show log panel", value=True)
    if st.button("Clear log", use_container_width=True):
        LOGS.clear()

warm()

st.write("**Examples:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.q = ex

question = st.text_input(
    "Your question", value=st.session_state.get("q", ""), placeholder="e.g. Ի՞նչ է աշխատանքային պայմանագիրը"
)

# Keyed on the model, index and token budget too, so changing any of them re-answers the
# question on the spot — tuning a knob you then have to re-ask to see is not tuning.
pending = (question, backend["model"], sel["name"], max_out)

if st.button("Ask", type="primary") or (question and pending != st.session_state.get("asked")):
    if not question.strip():
        st.warning("Enter a question.")
        st.stop()
    st.session_state.asked = pending

    # Retrieval and generation are driven separately (rather than through rag.answer) so
    # each stage can report as it finishes — on a reasoning model the generation leg runs
    # ~25s, and a demo needs to show that it is working, not just spin.
    index_label = sel["name"].removeprefix(prefix)
    seen = len(LOGS)
    with st.status(f"Retrieving from {index_label}…", expanded=True) as status:
        try:
            st.write(f"**1/2 Retrieving** — `{index_label}`, {mode}"
                     + (f" α={alpha}" if mode == "hybrid" else "") + f", k={top_k}")
            t0 = time.perf_counter()
            articles = rag.retrieve(question, k=top_k, mode=mode, alpha=alpha, collection=sel["name"])
            t_retrieve = time.perf_counter() - t0
            st.write(
                f"Found **{len(articles)} articles** in {t_retrieve:.2f}s — "
                + ", ".join(f"`{a['cite_id']}`" for a in articles[:6])
                + (" …" if len(articles) > 6 else "")
            )
            seen = drain(st, seen)

            status.update(label=f"Generating with {backend['name']}…")
            st.write(
                f"**2/2 Generating** — `{backend['model']}` at `{backend['base_url']}`, "
                f"max_tokens={max_out:,}"
            )
            t0 = time.perf_counter()
            gen = rag.generate(
                question,
                articles,
                model=backend["model"],
                base_url=backend["base_url"],
                api_key=backend.get("api_key"),
                max_tokens=max_out,
            )
            t_generate = time.perf_counter() - t0
            st.write(
                f"Answered in {t_generate:.2f}s — finish `{gen['finish_reason']}`, "
                f"completion **{gen['completion_tokens']:,}/{gen['max_tokens']:,}** budget "
                f"({gen['reasoning_tokens']:,} reasoning + {gen['answer_tokens']:,} answer), "
                f"{gen['total_tokens']:,} total"
            )
            if gen["truncated"]:
                st.write(
                    f"⚠️ **Truncated** — hit the {gen['max_tokens']:,}-token cap. "
                    f"Reasoning took {gen['reasoning_tokens'] / max(gen['completion_tokens'], 1):.0%} of it."
                )
            seen = drain(st, seen)
        except Exception as e:
            status.update(label=f"{backend['name']} failed", state="error", expanded=True)
            st.write(f"❌ `{type(e).__name__}`: {e}")
            drain(st, seen)
            st.error(f"**{backend['name']}** failed: {e}")
            if backend.get("hint"):
                st.caption("If the endpoint is on the other side of a tunnel, bring it up with:")
                st.code(backend["hint"], language="bash")
            log_panel()
            st.stop()

        elapsed = t_retrieve + t_generate
        status.update(
            label=f"Done in {elapsed:.1f}s — {len(articles)} articles, {gen['total_tokens']:,} tokens",
            state="complete",
            expanded=False,
        )

    result = {**gen, "retrieved": articles}

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Answer")
        refused = rag.REFUSAL_MARKER in result["answer"]
        if refused:
            st.warning("The system refused to answer — the corpus has no sufficient basis.")
        if result["truncated"]:
            st.error(
                f"**Answer cut off** — generation hit the {result['max_tokens']:,}-token cap "
                f"({result['reasoning_tokens']:,} of those went to hidden reasoning, leaving only "
                f"{result['answer_tokens']:,} for the answer). Raise **Max output tokens** in the "
                f"sidebar — it re-answers as soon as you move it."
            )
        st.markdown(cite_links(result["answer"], result["retrieved"]))

    with right:
        st.subheader("Sources")
        st.caption("What the answer rests on — click to see the full text")
        for a in result["retrieved"]:
            flag = ""
            if a.get("repealed"):
                flag = " 🚫 repealed"
            elif a.get("has_repealed_parts"):
                flag = " ⚠️ partly repealed"
            label = f"Article {a['article']}" if a["article"] else a["title"]
            with st.expander(f"**{label}** — {a['title'][:44]} · `{a['score']:.3f}`{flag}"):
                st.markdown("\n\n".join(a["texts"]))
                st.markdown(f"[Open on arlis.am ↗]({a['url']})")

    cost = (
        result["prompt_tokens"] / 1e6 * backend["prompt_cost"]
        + result["completion_tokens"] / 1e6 * backend["completion_cost"]
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Articles", len(result["retrieved"]))
    m2.metric(
        "Tokens", f"{result['total_tokens']:,}",
        help=f"prompt {result['prompt_tokens']:,} · completion {result['completion_tokens']:,}"
             f"/{result['max_tokens']:,} budget "
             f"({result['reasoning_tokens']:,} reasoning + {result['answer_tokens']:,} answer)",
    )
    m3.metric("Retrieval", f"{t_retrieve:.2f}s", help=f"Weaviate {mode} search over {sel['name']}")
    m4.metric("Generation", f"{t_generate:.2f}s", help=backend["model"])
    m5.metric("Cost", f"${cost:.4f}" if cost else "self-hosted", help=backend["model"])

log_panel()
