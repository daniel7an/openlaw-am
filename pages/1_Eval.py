"""Streamlit page: multi-model eval on pure Labor-Code QAs.

Compares citation metrics (deterministic) and answer correctness (Gemini judge).
Retrieval is shared across models; only generation differs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import eval as labor_eval
from config import get

st.set_page_config(page_title="Eval — openlaw-am", page_icon="📊", layout="wide")

METRIC_COLS = [
    "judge_mean_0_2",
    "judge_fully_correct",
    "judge_faithfulness",
    "citation_recall",
    "citation_precision",
    "exact_citation_match",
    "hallucinated_citation_rate",
    "hit@5",
    "latency_s",
    "tokens",
]

PCT_COLS = {
    "judge_fully_correct",
    "judge_faithfulness",
    "citation_recall",
    "citation_precision",
    "exact_citation_match",
    "hallucinated_citation_rate",
    "hit@5",
}


def _short(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _format_comparison(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("model")
    show = [c for c in METRIC_COLS if c in df.columns]
    return df[show]


def _render_comparison(payload: dict) -> None:
    meta = payload.get("meta", {})
    st.caption(
        f"n={meta.get('n_questions', '?')} pure labor QAs · "
        f"judge=`{meta.get('judge_model', labor_eval.JUDGE_MODEL)}` · "
        f"{meta.get('created_at', '')}"
    )

    cmp_rows = payload.get("comparison") or labor_eval.comparison_table(
        payload.get("results_by_model", {})
    )
    df = _format_comparison(cmp_rows)
    if df.empty:
        st.warning("Հարդյունքներ չկան։")
        return

    # Leaderboard metrics
    best_judge = df["judge_mean_0_2"].idxmax()
    best_cite = df["citation_recall"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Մոդելներ", len(df))
    c2.metric("Լավագույն judge", _short(str(best_judge)), f"{df.loc[best_judge, 'judge_mean_0_2']:.2f}/2")
    c3.metric("Լավագույն cite recall", _short(str(best_cite)), f"{df.loc[best_cite, 'citation_recall']:.0%}")
    c4.metric(
        "Նվազ. հալյուցինացիա",
        _short(str(df["hallucinated_citation_rate"].idxmin())),
        f"{df['hallucinated_citation_rate'].min():.0%}",
    )

    st.subheader("Համեմատական աղյուսակ")
    st.dataframe(
        df.style.format(
            {
                **{c: "{:.0%}" for c in PCT_COLS if c in df.columns},
                "judge_mean_0_2": "{:.2f}",
                "latency_s": "{:.1f}",
                "tokens": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Գծապատկերներ")
    chart_cols = [
        "judge_fully_correct",
        "citation_recall",
        "citation_precision",
        "exact_citation_match",
        "hallucinated_citation_rate",
    ]
    chart_df = df[chart_cols].copy()
    chart_df.index = [_short(i) for i in chart_df.index]
    st.bar_chart(chart_df, stack=False)

    judge_df = df[["judge_mean_0_2"]].copy()
    judge_df.index = [_short(i) for i in judge_df.index]
    st.caption("Judge mean (0–2 scale)")
    st.bar_chart(judge_df)

    # Per-question drilldown
    results = payload.get("results_by_model") or {}
    if not results:
        return

    st.subheader("Մանրամասներ ըստ հարցի")
    models = list(results.keys())
    # Align on question ids from the first model
    qids = [r["id"] for r in next(iter(results.values()))["rows"]]
    detail_rows = []
    for qid in qids:
        row = {"id": qid}
        for m in models:
            r = next((x for x in results[m]["rows"] if x["id"] == qid), None)
            if not r:
                continue
            row[f"{_short(m)} · judge"] = r["judge"].get("correctness")
            row[f"{_short(m)} · cite_r"] = r["citations"]["citation_recall"]
            row[f"{_short(m)} · exact"] = int(r["citations"]["exact_citation_match"])
        detail_rows.append(row)
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

    with st.expander("Պատասխաններ և judge rationale", expanded=False):
        qid = st.selectbox("Հարց", qids)
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            r = next(x for x in results[m]["rows"] if x["id"] == qid)
            with col:
                st.markdown(f"**`{_short(m)}`**")
                j = r["judge"]
                st.write(
                    f"judge={j.get('correctness')} · "
                    f"cite_recall={r['citations']['citation_recall']:.0%} · "
                    f"cited={r['citations']['cited_articles']} · "
                    f"gold={r['citations']['expected_articles']}"
                )
                if j.get("rationale"):
                    st.caption(j["rationale"])
                st.markdown(r["system_answer"] or "_(empty)_")


st.title("📊 Մոդելների գնահատում")
st.markdown(
    "Համեմատում է պատասխանող մոդելները **pure Labor-Code** հարցերի վրա "
    f"({len(labor_eval.labor_pure_questions())} հարց)՝ "
    "**citation metrics** (դետերմինիստիկ) և **answer correctness** (Gemini judge 0/1/2)։"
)

default_models = labor_eval.default_models()
with st.sidebar:
    st.subheader("Գնահատման կարգավորումներ")
    selected = st.multiselect(
        "Մոդելներ (OpenRouter)",
        options=default_models,
        default=default_models,
        help="Ցանկը՝ config.toml → [eval].models",
    )
    limit = st.slider("Քանի՞ հարց (limit)", 1, len(labor_eval.labor_pure_questions()), len(labor_eval.labor_pure_questions()))
    st.caption(
        f"Judge՝ `{labor_eval.JUDGE_MODEL}` · "
        f"embed՝ `{get('embedding.model', env='OPENLAW_EMBED_MODEL').split('/')[-1]}`"
    )
    st.divider()
    st.markdown(
        "**Citation recall** — ոսկե հոդվածը կա՞ որպես `[Հոդված N]`  \n"
        "**Citation precision** — մեջբերումներից քանիսն են ոսկե  \n"
        "**Judge 0/1/2** — պատասխանի ճշտություն vs gold summary"
    )

tab_run, tab_saved = st.tabs(["Վազեցնել", "Պահված արդյունքներ"])

with tab_run:
    run = st.button("Սկսել համեմատությունը", type="primary", disabled=not selected)
    if run:
        if not selected:
            st.error("Ընտրեք առնվազն մեկ մոդել։")
            st.stop()
        prog = st.progress(0.0, text="Սկսվում է…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            prog.progress(min(max(frac, 0.0), 1.0), text=msg)
            status.caption(msg)

        try:
            with st.spinner("Գնահատում… սա կարող է մի քանի րոպե տևել"):
                payload = labor_eval.compare_models(
                    models=selected, limit=limit, progress=on_progress, save=True
                )
            st.session_state["last_compare"] = payload
            prog.progress(1.0, text="Ավարտված է")
            if path := payload.get("meta", {}).get("path"):
                st.success(f"Պահվեց՝ `{path}`")
        except Exception as e:
            st.exception(e)
            st.stop()

    if "last_compare" in st.session_state:
        _render_comparison(st.session_state["last_compare"])
    else:
        st.info("Ընտրեք մոդելները և սեղմեք «Սկսել համեմատությունը».")

with tab_saved:
    saved = labor_eval.list_saved_comparisons()
    if not saved:
        st.info("Դեռ պահված համեմատություններ չկան (`results/labor_compare_*.json`).")
    else:
        choice = st.selectbox("Ֆայլ", saved, format_func=lambda p: p.name)
        if choice:
            payload = json.loads(Path(choice).read_text(encoding="utf-8"))
            _render_comparison(payload)
