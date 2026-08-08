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
    """
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


def article_num(cite_id: str) -> str | None:
    """labor-code-art-83 → 83; labor-code-art-3.1 → 3.1"""
    m = re.search(r"labor-code-art-(.+)$", cite_id)
    return m.group(1) if m else None


def parse_cited_articles(answer: str) -> list[str]:
    return list(dict.fromkeys(CITE_RE.findall(answer or "")))


def citation_metrics(answer: str, expected_ids: list[str], retrieved: list[dict]) -> dict:
    cited = parse_cited_articles(answer)
    expected_nums = [article_num(i) for i in expected_ids]
    expected_nums = [n for n in expected_nums if n]
    retrieved_nums = [a["article"] for a in retrieved if a.get("article")]

    exp_set, cit_set, ret_set = set(expected_nums), set(cited), set(retrieved_nums)

    # Gold citation coverage
    hit_expected = exp_set & cit_set
    recall = len(hit_expected) / len(exp_set) if exp_set else 0.0
    # Among cited, how many are expected
    precision_vs_gold = len(hit_expected) / len(cit_set) if cit_set else 0.0
    # Citations not present in retrieved context → hallucinated relative to grounding
    hallucinated = sorted(cit_set - ret_set)
    supported = sorted(cit_set & ret_set)
    halluc_rate = len(hallucinated) / len(cit_set) if cit_set else 0.0

    return {
        "cited_articles": cited,
        "expected_articles": expected_nums,
        "retrieved_articles": retrieved_nums,
        "citation_recall": recall,
        "citation_precision_vs_gold": precision_vs_gold,
        "supported_citations": supported,
        "hallucinated_citations": hallucinated,
        "hallucinated_citation_rate": halluc_rate,
        "exact_citation_match": exp_set <= cit_set and len(exp_set) > 0,
    }


def retrieval_hits(retrieved: list[dict], expected_ids: list[str]) -> dict:
    ranked = [a["cite_id"] for a in retrieved]
    exp = set(expected_ids)

    def hit_at(k: int) -> bool:
        return bool(exp & set(ranked[:k]))

    # After regrouping, rank is by first-seen chunk score order from rag.retrieve
    return {
        "retrieved_cite_ids": ranked,
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@8": hit_at(8),
    }


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
    r = requests.post(url, params={"key": GEMINI_KEY}, json=body, timeout=90)
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
    ret = Counter()
    for r in rows:
        for k in ("hit@1", "hit@3", "hit@5", "hit@8"):
            ret[k] += int(bool(r["retrieval"][k]))

    cites = [r["citations"] for r in rows]
    judges = [r["judge"] for r in rows if r["judge"].get("correctness") is not None]

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n": len(rows),
        "retrieval": {k: ret[k] / n for k in ("hit@1", "hit@3", "hit@5", "hit@8")},
        "citation_recall_mean": avg([c["citation_recall"] for c in cites]),
        "citation_precision_vs_gold_mean": avg([c["citation_precision_vs_gold"] for c in cites]),
        "exact_citation_match_rate": avg([float(c["exact_citation_match"]) for c in cites]),
        "hallucinated_citation_rate_mean": avg([c["hallucinated_citation_rate"] for c in cites]),
        "judge_correctness_mean": avg([j["correctness"] for j in judges]),
        "judge_correctness_dist": dict(Counter(j["correctness"] for j in judges)),
        "judge_fully_correct_rate": avg([float(j["correctness"] == 2) for j in judges]),
        "judge_faithfulness_rate": avg([float(j["faithfulness"]) for j in judges]),
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
        }
    return out


def print_report(summary: dict, rows: list[dict], args) -> None:
    print(f"\n=== eval (scope={args.scope}) ===")
    print(f"n={summary['n']}  model={args.model}  judge={args.judge_model}")
    print(f"retrieval  mode={args.search_mode or 'default'} alpha={args.alpha or 'default'} k={args.k}")
    r = summary["retrieval"]
    print(
        f"retrieval  hit@1={r['hit@1']:.0%}  hit@3={r['hit@3']:.0%}  "
        f"hit@5={r['hit@5']:.0%}  hit@8={r['hit@8']:.0%}"
    )
    print(
        f"citations  recall={summary['citation_recall_mean']:.0%}  "
        f"precision_vs_gold={summary['citation_precision_vs_gold_mean']:.0%}  "
        f"exact_match={summary['exact_citation_match_rate']:.0%}  "
        f"halluc_rate={summary['hallucinated_citation_rate_mean']:.0%}"
    )
    print(
        f"judge      mean={summary['judge_correctness_mean']:.2f}/2  "
        f"fully_correct={summary['judge_fully_correct_rate']:.0%}  "
        f"faithful={summary['judge_faithfulness_rate']:.0%}  "
        f"dist={summary['judge_correctness_dist']}"
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


def rescore(path: Path, with_judge: bool = False, judge_model: str = DEFAULT_JUDGE) -> None:
    """Recompute metrics from a stored run — no model calls.

    Answers are saved verbatim, so a scorer fix (e.g. the CITE_RE part-qualifier
    bug) can be applied to past runs instead of paying to regenerate them.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in payload["rows"]:
        retrieved = [{"article": a} for a in row["citations"]["retrieved_articles"]]
        row["citations"] = citation_metrics(
            row["system_answer"], row["expected_article_ids"], retrieved
        )
    if with_judge:
        # Judge scores can be filled in after the fact — the answers are stored, so a
        # run made before GEMINI_API_KEY existed doesn't have to be repeated.
        if not GEMINI_KEY:
            sys.exit("--with-judge needs GEMINI_API_KEY in .env")
        todo = [r for r in payload["rows"] if r["judge"].get("correctness") is None]
        print(f"judging {len(todo)} rows with {judge_model}…")
        for i, row in enumerate(todo, 1):
            row["judge"] = gemini_judge(
                row["question_armenian"], row["gold_summary"], row["system_answer"], judge_model
            )
            print(f"  [{i}/{len(todo)}] {row['id']} -> {row['judge'].get('correctness')}")
            time.sleep(1.0)
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


def run(args) -> Path:
    from rag import answer

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
        retrieval = retrieval_hits(result["retrieved"], q["expected_article_ids"])
        judge = (
            gemini_judge(q_hy, q["correct_answer_summary"], result["answer"], args.judge_model)
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
    p.add_argument("--judge-model", default=DEFAULT_JUDGE, help="Gemini judge model")
    p.add_argument("--limit", type=int, default=None, help="only first N questions")
    p.add_argument("--scope", default="covered", choices=["covered", "labor", "all"],
                   help="covered=gold articles all indexed (default); labor=pure-labour 9; all=50")
    p.add_argument("--k", type=int, default=get("retrieval.top_k"), help="articles retrieved")
    p.add_argument("--search-mode", default=None, choices=["hybrid", "vector", "bm25"])
    p.add_argument("--alpha", type=float, default=None, help="hybrid alpha, 1=vector 0=bm25")
    p.add_argument("--max-tokens", type=int, default=None, help="answer token ceiling")
    p.add_argument("--sleep", type=float, default=1.0, help="pause between questions")
    p.add_argument("--out", default=None, help="explicit results path")
    p.add_argument("--rescore", default=None,
                   help="recompute metrics for an existing results file (no answer-model calls)")
    p.add_argument("--with-judge", action="store_true",
                   help="with --rescore: fill in missing judge verdicts (needs GEMINI_API_KEY)")
    parsed = p.parse_args()
    if parsed.rescore:
        rescore(Path(parsed.rescore), parsed.with_judge, parsed.judge_model)
    else:
        run(parsed)
