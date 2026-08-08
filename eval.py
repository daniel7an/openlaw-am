"""Evaluate pure Labor-Code QAs: citation metrics + Gemini LLM-as-judge.

Filters eval/qa_dataset.json to questions whose expected_article_ids are all
labor-code-* (currently 9). For each: run rag.answer, score citations against
gold + retrieved context, then ask gemini-2.5-flash for a 0/1/2 correctness
verdict given question + gold summary + system answer.

Judge default is gemini-3.6-flash (gemini-2.5-flash is blocked for new API keys).

Usage:
    uv run python eval.py              # full pure-labor run → results/
    uv run python eval.py --limit 2    # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import get, prompt

load_dotenv()

QA_PATH = Path("eval/qa_dataset.json")
IRAVABAN_PATH = Path("eval/iravaban_qa.json")
OUT_DIR = Path("results")

# [Հոդված 83], [Հոդված 3.1], optional «րդ» / spaces.
# Also accepts a qualifier after the number — models routinely cite the *part* as
# well, e.g. [Հոդված 178, մաս 1]. Requiring "]" straight after the number scored
# those as zero citations, i.e. penalised the model for being more precise than
# the format asked for. Verified against real deepseek output on qa_003/021/023.
CITE_RE = re.compile(
    r"\[\s*Հոդված\s+(\d+(?:\.\d+)*)\s*(?:-?րդ)?\s*(?:[,։:][^\]]*)?\]",
    re.IGNORECASE,
)

# Secrets come from the environment. Everything else is a flag, so a run is fully
# described by the command line and reproducible from the results file.
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
JUDGE_SYSTEM = prompt("judge.system")
JUDGE_USER = prompt("judge.user")

DEFAULT_JUDGE = "gemini-3.6-flash"
JUDGE_BINARY_SYSTEM = prompt("judge_binary.system")
JUDGE_BINARY_USER = prompt("judge_binary.user")


def openai_judge(
    question: str,
    gold: str,
    system_answer: str,
    judge_model: str,
    base_url: str,
) -> dict:
    """Binary 1/0 correctness judge over any OpenAI-compatible endpoint.

    Used instead of the Gemini path because the Gemini free tier allows only
    20 requests/minute, which cannot sustain a multi-run sweep.
    """
    from rag import client

    messages = [
        {"role": "system", "content": JUDGE_BINARY_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_BINARY_USER.format(
                question=question, gold=gold, system_answer=system_answer
            ),
        },
    ]
    try:
        # deepseek-v4-pro is a reasoning model: it spends completion tokens on a
        # hidden trace before emitting anything, so a tight ceiling yields an empty
        # verdict. Retry with more room rather than scoring the row as unjudged.
        budget, text = 2000, ""
        for _ in range(3):
            resp = client(base_url).chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=budget,
                response_format={"type": "json_object"},
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                break
            budget *= 2
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        verdict = json.loads(text)
        score = int(verdict.get("score", -1))
        if score not in (0, 1):
            raise ValueError(f"bad score: {verdict}")
        return {
            "correctness": score,
            "faithfulness": None,
            "rationale": str(verdict.get("rationale", "")).strip(),
            "error": None,
        }
    except Exception as e:
        return {
            "correctness": None,
            "faithfulness": None,
            "rationale": str(e)[:300],
            "error": f"judge: {type(e).__name__}",
        }


def indexed_cite_ids() -> set[str]:
    """Every cite_id present in the current chunk set."""
    path = Path(get("paths.chunks"))
    return {
        json.loads(line)["cite_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def select_questions(scope: str, path: Path = QA_PATH) -> list[dict]:
    """Pick the question set to score.

    covered  — every gold article is in the index. The honest default: questions
               whose sources we simply do not hold measure corpus gaps, not the model.
    labor    — the original pure-labor subset (9), kept for comparison with earlier runs.
    all      — all 50, including ones no model can answer from this corpus.
    iravaban — benchmark 2: the merged iravaban.net set (eval/iravaban_qa.json),
               under the same coverage filter, so entries whose golds await review
               (repealed/renumbered articles) are held out automatically.
    """
    if scope == "iravaban":
        qs = json.loads(IRAVABAN_PATH.read_text(encoding="utf-8"))
        have = indexed_cite_ids()
        return [q for q in qs if q.get("expected_article_ids") and set(q["expected_article_ids"]) <= have]
    qs = json.loads(path.read_text(encoding="utf-8"))
    if scope == "all":
        return qs
    if scope == "labor":
        return [
            q for q in qs
            if (q.get("expected_article_ids") or [])
            and all(i.startswith("labor-code-") for i in q["expected_article_ids"])
        ]
    have = indexed_cite_ids()
    return [q for q in qs if set(q.get("expected_article_ids") or []) <= have and q.get("expected_article_ids")]


CITE_ID_RE = re.compile(r"^(?P<slug>.+?)-art-(?P<num>[0-9]+(?:[.\-][0-9]+)*)$")


def split_cite_id(cite_id: str) -> tuple[str, str | None]:
    """civil-code-art-1058 → ("civil-code", "1058"); law-165000 → ("law-165000", None).

    Was `labor-code-art-(.+)` only, which silently returned None for every other
    document — so once the corpus grew past the Labor Code, gold articles from the
    civil code / constitution produced an EMPTY expected set and scored 0% recall
    even when the model cited them correctly.
    """
    m = CITE_ID_RE.match(cite_id)
    return (m.group("slug"), m.group("num").replace("-", ".")) if m else (cite_id, None)


def article_num(cite_id: str) -> str | None:
    return split_cite_id(cite_id)[1]


def parse_cited_articles(answer: str) -> list[str]:
    return list(dict.fromkeys(CITE_RE.findall(answer or "")))


def citation_metrics(
    answer: str, expected_ids: list[str], retrieved: list[dict]
) -> dict:
    """Score [Հոդված N] citations against gold.

    The citation format carries no document, so across a 17-document corpus a bare
    "33" is ambiguous between constitution art. 33 and labor-code art. 33. We
    disambiguate through what was actually retrieved: a cited number counts for a
    gold (document, article) pair only if that pair was in the retrieved context.

    Gold entries with no article part (e.g. law-165000, a whole-act reference) are
    excluded from recall — they cannot be expressed in the [Հոդված N] format at all.
    """
    cited = parse_cited_articles(answer)
    cite_ids = [a.get("cite_id", "") for a in retrieved]
    retrieved_pairs = {split_cite_id(c) for c in cite_ids if c}
    retrieved_nums = {num for _, num in retrieved_pairs if num}

    expected_pairs = {split_cite_id(i) for i in expected_ids}
    doc_level_gold = sorted(slug for slug, num in expected_pairs if num is None)
    expected_pairs = {(s, n) for s, n in expected_pairs if n}
    expected_nums = sorted({n for _, n in expected_pairs})

    cit_set = set(cited)
    # A gold pair is covered when its number is cited AND that document supplied it.
    covered = {
        (slug, num) for slug, num in expected_pairs
        if num in cit_set and (slug, num) in retrieved_pairs
    }
    recall = len(covered) / len(expected_pairs) if expected_pairs else 0.0
    gold_nums = {n for _, n in covered}
    precision_vs_gold = len(gold_nums & cit_set) / len(cit_set) if cit_set else 0.0

    hallucinated = sorted(cit_set - retrieved_nums)
    supported = sorted(cit_set & retrieved_nums)
    halluc_rate = len(hallucinated) / len(cit_set) if cit_set else 0.0
    exp_set = expected_pairs

    return {
        "cited_articles": cited,
        "expected_articles": expected_nums,
        "doc_level_gold": doc_level_gold,
        "retrieved_articles": sorted(retrieved_nums),
        "citation_recall": recall,
        "citation_precision_vs_gold": precision_vs_gold,
        "supported_citations": supported,
        "hallucinated_citations": hallucinated,
        "hallucinated_citation_rate": halluc_rate,
        "exact_citation_match": bool(exp_set) and covered == exp_set,
    }


def retrieval_hits(retrieved: list[dict], expected_ids: list[str], k: int | None = None) -> dict:
    ranked = [a["cite_id"] for a in retrieved]
    exp = set(expected_ids)

    def hit_at(kk: int) -> bool:
        return bool(exp & set(ranked[:kk]))

    # After regrouping, rank is by first-seen chunk score order from rag.retrieve.
    # The run's own depth always gets a column, so a k=10/20 sweep reports hit@10/20.
    ks = sorted({1, 3, 5, 8} | ({k} if k else set()))
    out: dict = {"retrieved_cite_ids": ranked}
    out.update({f"hit@{kk}": hit_at(kk) for kk in ks})
    return out


def gemini_judge(question: str, gold: str, system_answer: str, judge_model: str = DEFAULT_JUDGE) -> dict:
    if not GEMINI_KEY:
        return {
            "correctness": None,
            "faithfulness": None,
            "rationale": "GEMINI_API_KEY missing",
            "error": "no_key",
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{judge_model}:generateContent"
    )
    user = JUDGE_USER.format(
        question=question, gold=gold, system_answer=system_answer
    )
    body = {
        "system_instruction": {"parts": [{"text": JUDGE_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    # Free tier allows 20 requests/minute; the API returns retryDelay on 429, so
    # honour it rather than burning the whole run on a transient quota bounce.
    for attempt in range(5):
        r = requests.post(url, params={"key": GEMINI_KEY}, json=body, timeout=90)
        if r.status_code != 429:
            break
        delay = 20.0
        m = re.search(r'"retryDelay":\s*"(\d+)s"', r.text)
        if m:
            delay = float(m.group(1))
        wait = delay + 2 * attempt
        print(f"    429 quota — waiting {wait:.0f}s (attempt {attempt + 1}/5)")
        time.sleep(wait)
    if r.status_code != 200:
        return {
            "correctness": None,
            "faithfulness": None,
            "rationale": r.text[:400],
            "error": f"http_{r.status_code}",
        }
    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        verdict = json.loads(text)
        c = int(verdict.get("correctness", -1))
        f = int(verdict.get("faithfulness", -1))
        if c not in (0, 1, 2) or f not in (0, 1):
            raise ValueError(f"bad scores: {verdict}")
        return {
            "correctness": c,
            "faithfulness": f,
            "rationale": str(verdict.get("rationale", "")).strip(),
            "error": None,
        }
    except Exception as e:
        return {
            "correctness": None,
            "faithfulness": None,
            "rationale": str(data)[:400],
            "error": f"parse: {e}",
        }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    hit_keys = sorted(
        {key for r in rows for key in r["retrieval"] if key.startswith("hit@")},
        key=lambda s: int(s.split("@")[1]),
    )
    ret = Counter()
    for r in rows:
        for k in hit_keys:
            ret[k] += int(bool(r["retrieval"].get(k)))

    cites = [r["citations"] for r in rows]
    judges = [r["judge"] for r in rows if r["judge"].get("correctness") is not None]

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    scores = [j["correctness"] for j in judges]
    # Binary judge scores 0/1; the older Gemini judge scored 0/1/2. Detect which so
    # "fully correct" means the top of whichever scale produced the file.
    top = 2 if any(sc == 2 for sc in scores) else 1
    faith = [j["faithfulness"] for j in judges if j.get("faithfulness") is not None]

    return {
        "n": len(rows),
        "judge_scale": top,
        "n_judged": len(judges),
        "retrieval": {k: ret[k] / n for k in hit_keys},
        "citations_per_answer_mean": avg([len(c["cited_articles"]) for c in cites]),
        "citation_recall_mean": avg([c["citation_recall"] for c in cites]),
        "citation_precision_vs_gold_mean": avg([c["citation_precision_vs_gold"] for c in cites]),
        "exact_citation_match_rate": avg([float(c["exact_citation_match"]) for c in cites]),
        "hallucinated_citation_rate_mean": avg([c["hallucinated_citation_rate"] for c in cites]),
        "judge_correctness_mean": avg([j["correctness"] for j in judges]),
        "judge_correctness_dist": dict(Counter(j["correctness"] for j in judges)),
        "judge_fully_correct_rate": avg([float(sc == top) for sc in scores]),
        "judge_faithfulness_rate": avg([float(f) for f in faith]),
        "latency_s_mean": avg([r["latency_s"] for r in rows]),
        "tokens_mean": avg([r["total_tokens"] for r in rows]),
        "by_difficulty": _by_difficulty(rows),
    }


def _by_difficulty(rows: list[dict]) -> dict:
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["difficulty"], []).append(r)
    out = {}
    for diff, rs in sorted(buckets.items()):
        judges = [x["judge"] for x in rs if x["judge"].get("correctness") is not None]
        out[diff] = {
            "n": len(rs),
            "hit@5": sum(x["retrieval"]["hit@5"] for x in rs) / len(rs),
            "citation_recall": sum(x["citations"]["citation_recall"] for x in rs) / len(rs),
            "exact_citation_match": sum(x["citations"]["exact_citation_match"] for x in rs) / len(rs),
            "judge_correctness_mean": (
                sum(j["correctness"] for j in judges) / len(judges) if judges else None
            ),
            "n_judged": len(judges),
        }
    return out


def print_report(summary: dict, rows: list[dict], args) -> None:
    print(f"\n=== eval (scope={args.scope}) ===")
    print(f"n={summary['n']}  model={args.model}  judge={args.judge_model}")
    print(f"retrieval  mode={args.search_mode or 'default'} alpha={args.alpha or 'default'} k={args.k}")
    r = summary["retrieval"]
    print("retrieval  " + "  ".join(f"{k}={v:.0%}" for k, v in r.items()))
    print(
        f"citations  recall={summary['citation_recall_mean']:.0%}  "
        f"precision_vs_gold={summary['citation_precision_vs_gold_mean']:.0%}  "
        f"exact_match={summary['exact_citation_match_rate']:.0%}  "
        f"halluc_rate={summary['hallucinated_citation_rate_mean']:.0%}"
    )
    print(
        f"judge      accuracy={summary['judge_fully_correct_rate']:.0%}  "
        f"(mean={summary['judge_correctness_mean']:.2f}/{summary['judge_scale']}, "
        f"n={summary['n_judged']}, dist={summary['judge_correctness_dist']})"
    )
    print(
        f"cost/lat   tokens≈{summary['tokens_mean']:.0f}/q  "
        f"latency≈{summary['latency_s_mean']:.1f}s/q"
    )
    print("\nper question:")
    for row in rows:
        j = row["judge"]
        c = row["citations"]
        print(
            f"  {row['id']} {row['difficulty']:<6} "
            f"hit5={int(row['retrieval']['hit@5'])} "
            f"cite_recall={c['citation_recall']:.0%} "
            f"exact={int(c['exact_citation_match'])} "
            f"judge={j.get('correctness')} "
            f"cited={c['cited_articles']} gold={c['expected_articles']}"
        )
        if j.get("rationale"):
            print(f"           └─ {j['rationale'][:160]}")


def rescore(
    path: Path,
    with_judge: bool = False,
    rejudge: bool = False,
    judge_model: str = DEFAULT_JUDGE,
    judge_backend: str = "openai",
    judge_base_url: str | None = None,
    judge_sleep: float = 4.0,
) -> None:
    """Recompute metrics from a stored run — no model calls.

    Answers are saved verbatim, so a scorer fix (e.g. the CITE_RE part-qualifier
    bug) can be applied to past runs instead of paying to regenerate them.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in payload["rows"]:
        retrieved = [{"cite_id": c} for c in row["retrieval"]["retrieved_cite_ids"]]
        row["citations"] = citation_metrics(
            row["system_answer"], row["expected_article_ids"], retrieved
        )
    if with_judge or rejudge:
        # Judge scores can be filled in (or replaced) after the fact — the answers
        # are stored, so a judge-prompt fix doesn't require regenerating them.
        # --with-judge fills only missing verdicts; --rejudge overwrites all.
        if judge_backend == "gemini" and not GEMINI_KEY:
            sys.exit("--with-judge on the gemini backend needs GEMINI_API_KEY in .env")
        todo = (
            payload["rows"]
            if rejudge
            else [r for r in payload["rows"] if r["judge"].get("correctness") is None]
        )
        print(f"judging {len(todo)} rows with {judge_model} ({judge_backend})…")
        for i, row in enumerate(todo, 1):
            if judge_backend == "gemini":
                row["judge"] = gemini_judge(
                    row["question_armenian"], row["gold_summary"], row["system_answer"], judge_model
                )
            else:
                row["judge"] = openai_judge(
                    row["question_armenian"], row["gold_summary"], row["system_answer"],
                    judge_model, judge_base_url,
                )
            print(f"  [{i}/{len(todo)}] {row['id']} -> {row['judge'].get('correctness')}")
            time.sleep(judge_sleep)
        payload["meta"]["judge_model"] = judge_model

    payload["summary"] = summarize(payload["rows"])
    payload["meta"]["rescored"] = True
    out = Path(path).with_name(Path(path).stem + "_rescored.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    s = payload["summary"]
    print(f"\n=== rescored: {Path(path).name} ===")
    print(f"model={payload['meta'].get('answer_model')}  n={s['n']}")
    r = s["retrieval"]
    print(f"retrieval  hit@1={r['hit@1']:.0%}  hit@3={r['hit@3']:.0%}  hit@5={r['hit@5']:.0%}")
    print(
        f"citations  recall={s['citation_recall_mean']:.0%}  "
        f"precision_vs_gold={s['citation_precision_vs_gold_mean']:.0%}  "
        f"exact_match={s['exact_citation_match_rate']:.0%}  "
        f"halluc_rate={s['hallucinated_citation_rate_mean']:.0%}"
    )
    if any(r["judge"].get("correctness") is not None for r in payload["rows"]):
        print(
            f"judge      mean={s['judge_correctness_mean']:.2f}/2  "
            f"fully_correct={s['judge_fully_correct_rate']:.0%}  "
            f"faithful={s['judge_faithfulness_rate']:.0%}  dist={s['judge_correctness_dist']}"
        )
    print(f"cost/lat   tokens~{s['tokens_mean']:.0f}/q  latency~{s['latency_s_mean']:.1f}s/q")
    print(f"wrote {out}")


def preflight() -> None:
    """Fail fast if retrieval is unavailable.

    Without this, a stopped Weaviate makes every question raise, each row scores 0,
    and the run still writes a results file that is indistinguishable from a real
    one. That has already happened once — an entire sweep of zeros looked valid.
    """
    from index import COLLECTION, connect

    try:
        client = connect()
    except Exception as e:
        sys.exit(f"Cannot reach Weaviate: {e}\nStart it with: docker compose up -d")
    try:
        n = client.collections.get(COLLECTION).aggregate.over_all(total_count=True).total_count
        if not n:
            sys.exit(f"Collection {COLLECTION!r} is empty — build an index first.")
        print(f"index {COLLECTION}: {n} objects")
    finally:
        client.close()


def run(args) -> Path:
    from rag import answer

    preflight()
    limit = args.limit
    questions = select_questions(args.scope)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        sys.exit("No pure labor-code questions found.")

    print(f"Evaluating {len(questions)} questions (scope={args.scope})…")
    rows = []
    for i, q in enumerate(questions, 1):
        q_hy = q["question_armenian"]
        print(f"\n[{i}/{len(questions)}] {q['id']} ({q['difficulty']})")
        t0 = time.perf_counter()
        try:
            result = answer(
                q_hy,
                k=args.k,
                mode=args.search_mode,
                alpha=args.alpha,
                model=args.model,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
            )
            err = None
        except Exception as e:
            result = {
                "answer": "",
                "retrieved": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            err = str(e)
            print(f"  ⚠️  rag failed: {e}")
        latency = time.perf_counter() - t0

        cites = citation_metrics(
            result["answer"], q["expected_article_ids"], result["retrieved"]
        )
        retrieval = retrieval_hits(result["retrieved"], q["expected_article_ids"], args.k)
        judge_fn = (
            gemini_judge
            if args.judge_backend == "gemini"
            else lambda qq, gg, aa, mm: openai_judge(qq, gg, aa, mm, args.judge_base_url)
        )
        judge = (
            judge_fn(q_hy, q["correct_answer_summary"], result["answer"], args.judge_model)
            if result["answer"]
            else {
                "correctness": 0,
                "faithfulness": 0,
                "rationale": f"empty answer ({err})",
                "error": err,
            }
        )

        row = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question_armenian": q_hy,
            "question_english": q.get("question_english"),
            "expected_article_ids": q["expected_article_ids"],
            "gold_summary": q["correct_answer_summary"],
            "system_answer": result["answer"],
            "retrieval": retrieval,
            "citations": cites,
            "judge": judge,
            "latency_s": round(latency, 3),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "error": err,
        }
        rows.append(row)
        print(
            f"  hit@5={retrieval['hit@5']}  cite_recall={cites['citation_recall']:.0%}  "
            f"judge={judge.get('correctness')}  {latency:.1f}s"
        )
        time.sleep(args.sleep)  # be kind to free-tier rate limits

    failed = [r for r in rows if r.get("error")]
    if failed:
        print(f"\n⚠️  {len(failed)}/{len(rows)} questions FAILED — metrics below are not trustworthy")
        print(f"    first error: {failed[0]['error'][:160]}")

    summary = summarize(rows)
    print_report(summary, rows, args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = re.sub(r"[^a-z0-9]+", "-", (args.model or "default").lower()).strip("-")
    out = Path(args.out) if args.out else OUT_DIR / f"{args.scope}_{tag}_{stamp}.json"
    payload = {
        "meta": {
            "created_at": stamp,
            "scope": args.scope,
            "n": len(rows),
            "answer_model": args.model,
            "base_url": args.base_url,
            "judge_model": args.judge_model,
            "judge_backend": args.judge_backend,
            "embed_model": get("embedding.model", env="OPENLAW_EMBED_MODEL"),
            "index": get("weaviate.alias", env="OPENLAW_COLLECTION"),
            "top_k": args.k,
            "search_mode": args.search_mode or get("retrieval.mode"),
            "alpha": args.alpha if args.alpha is not None else get("retrieval.alpha"),
            "max_tokens": args.max_tokens or get("generation.max_output_tokens"),
        },
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=get("generation.model"),
                   help="answering model id (default: config.toml generation.model)")
    p.add_argument("--base-url", default=get("generation.base_url"),
                   help="OpenAI-compatible endpoint (default: config.toml generation.base_url)")
    p.add_argument("--judge-model", default="deepseek/deepseek-v4-pro", help="judge model id")
    p.add_argument("--judge-backend", default="openai", choices=["openai", "gemini"],
                   help="openai = any OpenAI-compatible endpoint (default); gemini = AI Studio REST")
    p.add_argument("--judge-base-url", default=get("generation.base_url"),
                   help="endpoint for the openai judge backend")
    p.add_argument("--limit", type=int, default=None, help="only first N questions")
    p.add_argument("--scope", default="covered", choices=["covered", "labor", "all", "iravaban"],
                   help="covered=gold articles all indexed (default); labor=pure-labour 9; "
                        "all=50; iravaban=benchmark 2 (merged iravaban.net set)")
    p.add_argument("--k", type=int, default=get("retrieval.top_k"), help="articles retrieved")
    p.add_argument("--search-mode", default=None, choices=["hybrid", "vector", "bm25"])
    p.add_argument("--alpha", type=float, default=None, help="hybrid alpha, 1=vector 0=bm25")
    p.add_argument("--max-tokens", type=int, default=None, help="answer token ceiling")
    p.add_argument("--sleep", type=float, default=1.0, help="pause between questions")
    p.add_argument("--out", default=None, help="explicit results path")
    p.add_argument("--rescore", default=None,
                   help="recompute metrics for an existing results file (no answer-model calls)")
    p.add_argument("--judge-sleep", type=float, default=4.0,
                   help="seconds between judge calls (free tier allows 20/min)")
    p.add_argument("--with-judge", action="store_true",
                   help="with --rescore: fill in missing judge verdicts")
    p.add_argument("--rejudge", action="store_true",
                   help="with --rescore: re-judge ALL rows (e.g. after a judge-prompt fix)")
    parsed = p.parse_args()
    if parsed.rescore:
        rescore(
            Path(parsed.rescore),
            with_judge=parsed.with_judge,
            rejudge=parsed.rejudge,
            judge_model=parsed.judge_model,
            judge_backend=parsed.judge_backend,
            judge_base_url=parsed.judge_base_url,
            judge_sleep=parsed.judge_sleep,
        )
    else:
        run(parsed)
