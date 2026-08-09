"""Streamlit chat UI for openlaw-am — ChatGPT-style.

Left sidebar: chat history (LLM-named, one line each, empty chats hidden), new
chat, and a Settings modal. The model picker is a popover in the composer, next
to the input. The conversation anchors to the bottom and grows upward; user
prompts sit right, answers left — ChatGPT-style. Each assistant turn shows a
collapsible "Thinking" panel (retrieval trace + the model's hidden reasoning,
live), then streams the answer conversationally at a readable pace. Every
citation stays a clickable arlis.am link — auditability is still the point.

Usage:
    uv run streamlit run app.py
"""
import logging
import os
import re
import time
import uuid
from collections import deque

import streamlit as st

import rag
from config import get

st.set_page_config(page_title="openlaw-am — RA legal assistant", page_icon="⚖️", layout="wide")

# Streamlit ≥1.61 exposes no theme CSS variables, so the bubble palette is
# computed here from the resolved theme and injected as our own custom props.
_dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
st.markdown(
    "<style>:root {{ --bubble-user: {}; --bubble-assistant: {}; --bubble-hover: {}; }}</style>".format(
        "#262730" if _dark else "#e9ebf0",
        "#1a1c24" if _dark else "#f4f5f8",
        "rgba(255,255,255,.08)" if _dark else "rgba(0,0,0,.06)",
    ),
    unsafe_allow_html=True,
)

# Layout CSS, all in one place:
# - sidebar chat titles: one line, ellipsized — never wrap onto a second row;
# - conversation anchored to the bottom (grows upward, ChatGPT-style) — the
#   main block container is stretched to the viewport and content pushed down;
# - user prompts as right-aligned bubbles, assistant answers plain on the left.
st.markdown(
    """<style>
    section[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {
        overflow: hidden;
    }
    section[data-testid="stSidebar"] .stButton button p {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    /* Chats + Settings as a plain left-aligned list, not framed buttons. Only
       secondary buttons — ➕ New chat (primary) keeps its button look. */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        border: none; background: transparent; justify-content: flex-start;
    }
    /* the button's inner wrapper centers its fit-content label span — left it too */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] > div {
        justify-content: flex-start;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] p {
        text-align: left;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:focus:not(:active) {
        background: var(--bubble-hover); color: inherit; border: none;
    }
    /* Bottom-anchor the conversation. The scroll container is already a flex
       column of [content, spacer, sticky input]; an auto top margin pushes the
       content down while short and collapses to 0 once history overflows —
       scrolling stays intact and the input stays at the lowest point. No
       min-height: forcing one makes the sticky input's height permanent
       overflow, which is what pushed fresh messages under the prompt line. */
    section[data-testid="stAppScrollToBottomContainer"] > div[data-testid="stMainBlockContainer"] {
        margin-top: auto;
    }
    /* Streamlit puts a flex-growing spacer BETWEEN the content and the sticky
       input, so it absorbs the free space and margin-top:auto gets none. Send
       the spacer above the content (order:-1): short content then hugs the
       input, long content scrolls normally. Still no min-height anywhere —
       that's what once buried messages under the input. */
    section[data-testid="stAppScrollToBottomContainer"] > div[data-testid="stMainBlockContainer"] + div {
        order: -1;
    }
    /* No avatars; the query and the answer are both filled gray bubbles in two
       tones — user darker on the right, assistant lighter and left. */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        width: fit-content; max-width: 85%;
        margin-left: auto;
        padding: 0.75rem 1.25rem;
        background: var(--bubble-user);
        border-radius: 1.2rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        padding: 0.75rem 1.25rem;
        background: var(--bubble-assistant);
        border-radius: 1.2rem;
    }
    /* The loading/thinking status and sources render unframed inside the bubble. */
    div[data-testid="stChatMessage"] [data-testid="stExpander"] details {
        border: none; background: transparent;
    }
    /* Example-question buttons on the blank chat: equal boxes regardless of how
       many lines each question wraps to. */
    div[data-testid="stMainBlockContainer"] .stButton button {
        height: 4.5rem;
    }
    </style>""",
    unsafe_allow_html=True,
)

# The interface is English; the questions stay Armenian because the corpus,
# retrieval and answers are Armenian.
EXAMPLES = [
    "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը",
    "Ի՞նչ դեպքերում կարող է գործատուն լուծել աշխատանքային պայմանագիրը",
    "Քանի՞ օր է տարեկան արձակուրդի նվազագույն տևողությունը",
    "Ինչպե՞ս գրանցել ամուսնություն Հայաստանում",  # Family Code, in no index -> should refuse
]

BACKENDS = get("generation.backends")
STREAM_DELAY = 0.02   # s per streamed chunk — deliberate, readable pace
HISTORY_TURNS = 8     # prior messages sent to the model (4 Q/A pairs)


@st.cache_resource
def log_sink() -> deque:
    """Ring buffer fed by the root logger; rendered inside the Thinking panel.

    cache_resource, not session_state: the handler must attach exactly once per
    process and survive Streamlit's rerun-on-every-interaction.
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
    """Index versions in Weaviate right now, plus the alias target."""
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
    """Cheap liveness probe, so a dead SSH tunnel shows up before you ask."""
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
    """Turn [Տag, Հոդված N] (or legacy [Հոդված N]) into an arlis.am link."""
    by_tag_num = {(rag.doc_tag(a), a["article"]): a["url"] for a in articles if a["article"]}
    by_num = {a["article"]: a["url"] for a in articles if a["article"]}

    def repl(m):
        tag, num = m.group(1), m.group(2)
        url = by_tag_num.get((tag, num)) if tag else by_num.get(num)
        url = url or by_num.get(num)  # tag the model invented — fall back to number
        return f"[[{m.group(0)[1:-1]}]]({url})" if url else m.group(0)

    return re.sub(
        r"\[(?:([^,\[\]]{1,30}),\s*)?Հոդված\s+([0-9]+(?:\.[0-9]+)?)(?:\s*[,։:][^\]]*)?\]",
        repl, answer,
    )


def sources_expander(articles: list[dict]) -> None:
    with st.expander(f"Sources — {len(articles)} article(s)"):
        for a in articles:
            flag = ""
            if a.get("repealed"):
                flag = " — repealed"
            elif a.get("has_repealed_parts"):
                flag = " — partly repealed"
            label = f"{rag.cite_label(a)}" if a["article"] else a["title"]
            st.markdown(f"**{label}** — {a['title'][:60]} · `{a['score']:.3f}`{flag}")
            st.markdown(f"[Open on arlis.am ↗]({a['url']})")
            st.divider()


# ---------------------------------------------------------------- chat state
def new_chat() -> None:
    """Switch to an empty chat, reusing one if it exists.

    Empty chats are hidden from the sidebar and reused here, so mashing
    "New chat" can never litter the history with blank entries.
    """
    for cid, c in st.session_state.chats.items():
        if not c["messages"]:
            st.session_state.current = cid
            return
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "", "messages": []}
    st.session_state.current = cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()

# ---------------------------------------------------------------- settings
# Settings live in a centered modal (st.dialog), not the sidebar. Widgets keyed
# into session_state so the values survive the dialog closing; setdefault seeds
# them once per session.
for key, value in {
    "max_out": rag.MAX_OUTPUT_TOKENS,
    "top_k": rag.TOP_K,
    "mode": get("retrieval.mode"),
    "alpha": get("retrieval.alpha"),
    "show_log": False,
}.items():
    st.session_state.setdefault(key, value)


@st.dialog("Կարգավորումներ")
def settings() -> None:
    st.slider("Max output tokens", 500, 8000, step=250, key="max_out")
    st.slider("Articles to retrieve (top-k)", 3, 15, key="top_k")
    st.radio(
        "Retrieval",
        ["hybrid", "vector", "bm25"],
        key="mode",
        format_func={"hybrid": "Hybrid", "vector": "Vector", "bm25": "BM25"}.get,
        horizontal=True,
    )
    st.slider("alpha (1=vector, 0=BM25)", 0.0, 1.0, step=0.05, key="alpha",
              disabled=st.session_state.mode != "hybrid")
    st.toggle("Debug log", key="show_log")
    if st.session_state.show_log and st.button("Clear log", use_container_width=True):
        LOGS.clear()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## OpenLaw<sup>AM</sup>", unsafe_allow_html=True)
    st.caption("Բաց, ստուգելի իրավաբանական օգնական ՀՀ օրենսդրության համար։")

    # Only chats with messages are listed; the current empty one stays invisible
    # until its first question (see new_chat).
    named = [(cid, c) for cid, c in st.session_state.chats.items() if c["messages"]]
    if named:
        st.caption("**Chats**")
        for cid, chat_ in reversed(named):
            marker = "● " if cid == st.session_state.current else ""
            if st.button(f"{marker}{chat_['title']}", key=f"chat_{cid}", use_container_width=True):
                st.session_state.current = cid
                st.rerun()

    if st.button("Նոր զրուցարան", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.divider()
    if st.button("Կարգավորումներ", use_container_width=True):
        settings()

max_out, top_k = st.session_state.max_out, st.session_state.top_k
mode, alpha = st.session_state.mode, st.session_state.alpha
show_log = st.session_state.show_log

# The index version picker is gone from the UI: the app pins itself to
# weaviate.ui_default (or the alias target) and that's that. OPENLAW_UI_COLLECTION
# still overrides per deployment.
indexes, alias_target = index_versions()
if not indexes:
    st.error("No index versions in Weaviate. Build one: `uv run python index.py --name all_codes`")
    st.stop()
prefix = get("weaviate.collection_prefix")
wanted = get("weaviate.ui_default", env="OPENLAW_UI_COLLECTION") or alias_target
index_names = [i["name"] for i in indexes]
sel = indexes[index_names.index(wanted) if wanted in index_names else 0]

warm()
chat = st.session_state.chats[st.session_state.current]

# ---------------------------------------------------------------- history
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("thinking"):
                with st.expander(msg.get("thinking_label", "Thinking"), expanded=False):
                    st.markdown(msg["thinking"])
            st.markdown(msg["content"])
            if msg.get("sources"):
                sources_expander(msg["sources"])
            if msg.get("caption"):
                st.caption(msg["caption"])
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------- input
if not chat["messages"]:
    # Landing state: wordmark well above the examples (big bottom margin), the
    # examples in a narrower centered column, and a spacer holding them off the
    # prompt line. All in the same `not chat["messages"]` branch, so the first
    # question removes the whole group.
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:6rem;'>OpenLaw<sup>AM</sup></h1>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 3.2, 1])
    with mid:
        st.caption("Հարցերի օրինակներ")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.queued = ex
                st.rerun()
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

# The model picker lives in the composer, ChatGPT-style. st.bottom is the pinned
# container st.chat_input renders into; putting the picker there keeps the two
# glued together at the bottom instead of the picker scrolling away with the page.
names = [b["name"] for b in BACKENDS]
ui_default = get("generation.ui_default", env="OPENLAW_UI_MODEL", default="")
cli_default = get("generation.model", env="OPENLAW_MODEL")
default_idx = (
    names.index(ui_default)
    if ui_default in names
    else next((i for i, b in enumerate(BACKENDS) if b["model"] == cli_default), 0)
)


def backend_blurb(b: dict) -> str:
    """One line under each model option: liveness, price, what it is."""
    dot = "online" if reachable(b["base_url"]) else "offline"
    cost = b.get("prompt_cost")
    price = (
        "free · self-hosted" if cost == 0
        else f"${b['prompt_cost']:.2f} in / ${b['completion_cost']:.2f} out per 1M tokens" if cost
        else "hosted"
    )
    return " — ".join(filter(None, [f"{dot} · {price}", b.get("note")]))


# chat_input goes into the pinned bottom container first, the picker after it —
# so the model chip sits UNDER the prompt line.
question = st.chat_input("Տվեք Ձեր իրավաբանական հարցը")
if not question and st.session_state.get("queued"):
    question = st.session_state.pop("queued")

with st.bottom:
    current = st.session_state.get("model_name", names[default_idx])
    with st.popover(current):
        choice = st.radio(
            "Model", names, index=default_idx, key="model_name",
            captions=[backend_blurb(b) for b in BACKENDS],
        )
    backend = BACKENDS[names.index(choice)]
    if not reachable(backend["base_url"]):
        st.error(f"**{backend['name']}**: `{backend['base_url']}` is not responding.")
        if backend.get("hint"):
            st.code(backend["hint"], language="bash")
# api_key_env keeps secrets in .env — a literal api_key in config is only for
# placeholders like vLLM's "EMPTY".
api_key = backend.get("api_key") or (
    os.getenv(backend["api_key_env"]) if backend.get("api_key_env") else None
)

if question:
    chat["messages"].append({"role": "user", "content": question})
    if not chat["title"]:
        chat["title"] = question
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        thinking_notes: list[str] = []
        index_label = sel["name"].removeprefix(prefix)

        with st.status("Retrieving needed info…", expanded=True) as status:
            note_slot = st.empty()
            reason_slot = st.empty()
        answer_slot = st.empty()

        def think(line: str) -> None:
            thinking_notes.append(line)
            note_slot.markdown("\n".join(f"- {n}" for n in thinking_notes))

        try:
            think(f"Searching **{index_label}** ({mode}, k={top_k})…")
            t0 = time.perf_counter()
            articles = rag.retrieve(question, k=top_k, mode=mode, alpha=alpha, collection=sel["name"])
            t_retrieve = time.perf_counter() - t0
            think(
                f"Found **{len(articles)}** articles in {t_retrieve:.1f}s: "
                + ", ".join(f"`{rag.cite_label(a)}`" for a in articles[:6])
                + (" …" if len(articles) > 6 else "")
            )
            status.update(label="Formulating answer…")
            think(f"Reading the articles and writing an answer with `{backend['model']}`…")

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in chat["messages"][:-1]
                if m["role"] in ("user", "assistant")
            ][-HISTORY_TURNS:]

            streamed, reasoning, gen = "", "", None
            t0 = time.perf_counter()
            events = rag.generate_stream(
                question,
                articles,
                model=backend["model"],
                base_url=backend["base_url"],
                api_key=api_key,
                max_tokens=max_out,
                history=history,
            )
            for kind, payload in events:
                if kind == "reasoning":
                    reasoning += payload
                    status.update(label=f"Formulating answer… ({len(reasoning):,} characters of reasoning)")
                    reason_slot.markdown(f"*…{reasoning[-400:]}*")
                elif kind == "content":
                    if not streamed:
                        status.update(
                            label=f"Thought for {time.perf_counter() - t0:.0f}s · {len(articles)} sources",
                            state="complete",
                            expanded=False,
                        )
                    streamed += payload
                    answer_slot.markdown(cite_links(streamed, articles) + " ▌")
                    time.sleep(STREAM_DELAY)
                elif kind == "retry":
                    streamed = ""
                    think(f"Empty answer — retrying with a {payload:,}-token budget…")
                    status.update(label="Retrying…", state="running", expanded=True)
                elif kind == "done":
                    gen = payload
        except Exception as e:
            status.update(label=f"{backend['name']} failed", state="error", expanded=True)
            answer_slot.error(f"**{backend['name']}** failed: {e}")
            if backend.get("hint"):
                st.caption("If the endpoint is behind a tunnel, bring it up with:")
                st.code(backend["hint"], language="bash")
            st.stop()

        t_generate = time.perf_counter() - t0
        linked = cite_links(gen["answer"], articles)
        answer_slot.markdown(linked)
        if gen["truncated"]:
            st.warning(
                f"Answer cut off at the {gen['max_tokens']:,}-token cap "
                f"({gen['reasoning_tokens']:,} went to hidden reasoning). "
                f"Raise **Max output tokens** in Settings and re-ask."
            )
        sources_expander(articles)
        caption = (
            f"{backend['name']} · {gen['total_tokens']:,} tokens · "
            f"retrieval {t_retrieve:.1f}s · generation {t_generate:.1f}s"
        )
        st.caption(caption)

        thinking_text = "\n".join(f"- {n}" for n in thinking_notes)
        if reasoning:
            thinking_text += f"\n\n**Model reasoning:**\n\n*{reasoning}*"
        chat["messages"].append(
            {
                "role": "assistant",
                "content": linked,
                "thinking": thinking_text,
                "thinking_label": f"Thought for {t_generate:.0f}s · {len(articles)} sources",
                "sources": articles,
                "caption": caption,
            }
        )

    # Name the chat once, after the first answer — the raw question set at submit
    # time is only a fallback for when the naming call fails. The rerun repaints
    # the sidebar, which rendered before the title existed.
    if not chat.get("titled"):
        chat["titled"] = True
        chat["title"] = (
            rag.chat_title(
                question, model=backend["model"], base_url=backend["base_url"], api_key=api_key
            )
            or chat["title"]
        )
        st.rerun()

if show_log:
    st.divider()
    st.caption(f"Debug log — {len(LOGS)} lines, newest last")
    st.code("\n".join(LOGS) or "(nothing logged yet)", language="log", height=280)
