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

from config import prompt

load_dotenv()

QA_PATH = Path("eval/qa_dataset.json")
OUT_DIR = Path("results")

# [Հոդված 83], [Հոդված 3.1], optional «րդ» / spaces
CITE_RE = re.compile(
    r"\[\s*Հոդված\s+(\d+(?:\.\d+)*)\s*(?:-?րդ)?\s*\]",
    re.IGNORECASE,
)

JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
JUDGE_SYSTEM = prompt("judge.system")
JUDGE_USER = prompt("judge.user")


def labor_pure_questions(path: Path = QA_PATH) -> list[dict]:
    qs = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in qs:
        ids = q.get("expected_article_ids") or []
        if ids and all(i.startswith("labor-code-") for i in ids):
            out.append(q)
    return out


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


def gemini_judge(question: str, gold: str, system_answer: str) -> dict:
    if not GEMINI_KEY:
        return {
            "correctness": None,
            "faithfulness": None,
            "rationale": "GEMINI_API_KEY missing",
            "error": "no_key",
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{JUDGE_MODEL}:generateContent"
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


def print_report(summary: dict, rows: list[dict]) -> None:
    print("\n=== Pure Labor-Code eval ===")
    print(f"n={summary['n']}  model={os.getenv('OPENLAW_MODEL')}  judge={JUDGE_MODEL}")
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


def run(limit: int | None = None) -> Path:
    from rag import answer

    questions = labor_pure_questions()
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        sys.exit("No pure labor-code questions found.")

    print(f"Evaluating {len(questions)} pure labor QAs…")
    rows = []
    for i, q in enumerate(questions, 1):
        q_hy = q["question_armenian"]
        print(f"\n[{i}/{len(questions)}] {q['id']} ({q['difficulty']})")
        t0 = time.perf_counter()
        try:
            result = answer(q_hy)
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
            gemini_judge(q_hy, q["correct_answer_summary"], result["answer"])
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
        time.sleep(1.0)  # be kind to free-tier rate limits

    summary = summarize(rows)
    print_report(summary, rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"labor_pure_eval_{stamp}.json"
    payload = {
        "meta": {
            "created_at": stamp,
            "scope": "pure_labor_code",
            "n": len(rows),
            "answer_model": os.getenv("OPENLAW_MODEL"),
            "judge_model": JUDGE_MODEL,
            "embed_model": os.getenv("OPENLAW_EMBED_MODEL"),
        },
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="only first N questions")
    args = p.parse_args()
    run(limit=args.limit)
