"""Streamlit demo UI for openlaw-am.

The point of the layout is auditability: the answer sits next to the exact articles
it was grounded in, each with a similarity score and a live arlis.am link, so a
claim can be checked against the source in one click.

Usage:
    uv run streamlit run app.py
"""
import re
import time

import streamlit as st

import rag
from config import get

st.set_page_config(page_title="openlaw-am — ՀՀ իրավական օգնական", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը",
    "Ի՞նչ դեպքերում կարող է գործատուն լուծել աշխատանքային պայմանագիրը",
    "Քանի՞ օր է տարեկան արձակուրդի նվազագույն տևողությունը",
    "Ինչպե՞ս գրանցել ամուսնություն Հայաստանում",  # out of corpus -> should refuse
]


@st.cache_resource(show_spinner="Բեռնվում է որոնման մոդելը…")
def warm():
    """Load the embedding model once, not on every rerun."""
    from index import encoder

    encoder()
    return True


def cite_links(answer: str, articles: list[dict]) -> str:
    """Turn [Հոդված N] into a link to that article on arlis.am."""
    url_by_article = {a["article"]: a["url"] for a in articles if a["article"]}

    def repl(m):
        num = m.group(1)
        url = url_by_article.get(num)
        return f"[[Հոդված {num}]]({url})" if url else m.group(0)

    return re.sub(r"\[Հոդված\s+([0-9]+(?:\.[0-9]+)?)\]", repl, answer)


st.title("⚖️ openlaw-am")
st.caption(
    "Բաց, ստուգելի իրավական օգնական՝ հիմնված ARLIS-ի վրա։ "
    "Յուրաքանչյուր պնդում ուղեկցվում է հոդվածի հղումով։ "
    "**Սա իրավաբանական խորհրդատվություն չէ։**"
)

with st.sidebar:
    st.subheader("Կարգավորումներ")
    top_k = st.slider("Որքա՞ն հոդված վերցնել (top-k)", 3, 15, rag.TOP_K)
    mode = st.radio(
        "Որոնման եղանակ",
        ["hybrid", "vector", "bm25"],
        format_func={"hybrid": "Հիբրիդ (BM25 + վեկտոր)", "vector": "Վեկտորային", "bm25": "BM25 (բանալի բառեր)"}.get,
        help="Չափված 18 հարցի վրա (hit@3)՝ հիբրիդ 72%, վեկտոր 67%, BM25 39%",
    )
    alpha = st.slider(
        "alpha (1 = վեկտոր, 0 = BM25)", 0.0, 1.0, get("retrieval.alpha"), 0.05,
        disabled=mode != "hybrid",
    )
    st.divider()
    st.subheader("Ընթացիկ կորպուս")
    st.markdown(
        f"- **ՀՀ Աշխատանքային օրենսգիրք** — 286 հոդված\n"
        f"- Տեքստը՝ ARLIS, ուժի մեջ 10.07.2026\n"
        f"- Մոդել՝ `{get('generation.model', env='OPENLAW_MODEL')}`\n"
        f"- Embeddings՝ `{get('embedding.model', env='OPENLAW_EMBED_MODEL').split('/')[-1]}`"
    )
    st.info(
        "Կորպուսում ներառված է միայն Աշխատանքային օրենսգիրքը։ "
        "Այլ ոլորտի հարցերին համակարգը պետք է հրաժարվի պատասխանել։"
    )

warm()

st.write("**Օրինակներ՝**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.q = ex

question = st.text_input(
    "Ձեր հարցը", value=st.session_state.get("q", ""), placeholder="Օր.՝ Ի՞նչ է աշխատանքային պայմանագիրը"
)

if st.button("Հարցնել", type="primary") or (question and question != st.session_state.get("asked")):
    if not question.strip():
        st.warning("Մուտքագրեք հարց։")
        st.stop()
    st.session_state.asked = question

    start = time.time()
    with st.spinner("Որոնում և պատասխանի կազմում…"):
        result = rag.answer(question, k=top_k, mode=mode, alpha=alpha)
    elapsed = time.time() - start

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Պատասխան")
        refused = rag.REFUSAL_MARKER in result["answer"]
        if refused:
            st.warning("Համակարգը հրաժարվեց պատասխանել՝ կորպուսում բավարար հիմք չկա։")
        st.markdown(cite_links(result["answer"], result["retrieved"]))

    with right:
        st.subheader("Աղբյուրներ")
        st.caption("Ինչի՞ վրա է հենվել պատասխանը — սեղմեք՝ ամբողջական տեքստը տեսնելու համար")
        for a in result["retrieved"]:
            flag = ""
            if a.get("repealed"):
                flag = " 🚫 ուժը կորցրել է"
            elif a.get("has_repealed_parts"):
                flag = " ⚠️ մասամբ ուժը կորցրել է"
            label = f"Հոդված {a['article']}" if a["article"] else a["title"]
            with st.expander(f"**{label}** — {a['title'][:44]} · `{a['score']:.3f}`{flag}"):
                st.markdown("\n\n".join(a["texts"]))
                st.markdown(f"[Բացել arlis.am-ում ↗]({a['url']})")

    cost = result["prompt_tokens"] / 1e6 * 0.435 + result["completion_tokens"] / 1e6 * 0.87
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Հոդված", len(result["retrieved"]))
    m2.metric("Tokens", f"{result['total_tokens']:,}")
    m3.metric("Ժամանակ", f"{elapsed:.1f}վ")
    m4.metric("Արժեք", f"${cost:.4f}")
