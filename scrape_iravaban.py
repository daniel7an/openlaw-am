"""Harvest question->citation benchmark candidates from iravaban.net.

The «Իրազեկում» (awareness) category (~350 pages x 14 posts) is a stream of
lawyer-prepared legal explainers whose headlines are citizen-style questions and
whose bodies cite exact statute articles («ՀՀ աշխատանքային օրենսգրքի 110-րդ
հոդվածի 1-ին մաս»). We extract (question, answer, citations), resolve citations
against data/chunks.jsonl ids, and emit benchmark candidates compatible with
eval/qa_dataset.json.

robots.txt: `Allow: /` for `*`, with Content-Signal ai-train=no — we use this
for retrieval EVALUATION (reference use), not model training. Crawl politely.

Usage:
    uv run python scrape_iravaban.py --listings              # all category pages -> data/iravaban/listings.jsonl
    uv run python scrape_iravaban.py --listings --pages 3    # first 3 pages only
    uv run python scrape_iravaban.py --articles              # fetch post HTML -> data/iravaban/raw/{id}.html
    uv run python scrape_iravaban.py --articles --limit 30   # first 30 uncached posts
    uv run python scrape_iravaban.py --extract               # -> data/iravaban/extracted.jsonl + eval/iravaban_candidates.jsonl

Cached: listings/articles skip work already on disk (--force re-fetches).
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
from parsel import Selector

BASE = "https://iravaban.net"
CATEGORY = f"{BASE}/category/newsfeed/awareness"
OUT = Path("data/iravaban")
RAW = OUT / "raw"
LISTINGS = OUT / "listings.jsonl"
EXTRACTED = OUT / "extracted.jsonl"
CANDIDATES = Path("eval/iravaban_candidates.jsonl")
CHUNKS = Path("data/chunks.jsonl")

DELAY = 1.0  # seconds between requests; same spirit as scrape.py's POLITE
HEADERS = {"User-Agent": "openlaw-am/0.1 (Hack Armenia 2026; open-source legal RAG; benchmark eval)"}
MIN_BYTES = 5000  # tiny responses are error pages, not posts

# Code-name genitives as they appear in citations, most specific first so
# «քրեական դատավարության օրենսգիրք» never resolves as criminal-code. Prefixes
# match the chunk-id scheme in data/chunks.jsonl; codes not yet in the corpus
# are still recorded (they tell us what to scrape from ARLIS next).
CODE_PATTERNS = [
    (r"քրեական\s+դատավարության\s+օրենսգրք", "criminal-procedure-code"),
    (r"քաղաքացիական\s+դատավարության\s+օրենսգրք", "civil-procedure-code"),
    (r"վարչական\s+իրավախախտումների\s+(?:վերաբերյալ\s+)?օրենսգրք", "admin-offenses-code"),
    (r"վարչական\s+դատավարության\s+օրենսգրք", "admin-procedure-code"),
    (r"աշխատանքային\s+օրենսգրք", "labor-code"),
    (r"քաղաքացիական\s+օրենսգրք", "civil-code"),
    (r"հարկային\s+օրենսգրք", "tax-code"),
    (r"քրեական\s+օրենսգրք", "criminal-code"),
    (r"ընտանեկան\s+օրենսգրք", "family-code"),
    (r"հողային\s+օրենսգրք", "land-code"),
    (r"ջրային\s+օրենսգրք", "water-code"),
    (r"Սահմանադրության", "constitution"),
    (r"«([^»]{3,120})»\s+(?:ՀՀ\s+)?օրենք", "law"),
]
CODE_RE = re.compile("|".join(f"(?P<c{i}>{p})" for i, (p, _) in enumerate(CODE_PATTERNS)))
# «110-րդ հոդված», «59.1-ին հոդված», optionally followed by «...ի 1-ին մաս»
ARTICLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(?:րդ|ին)\s+հոդված(?:ի\s+(\d+(?:\.\d+)?)\s*-\s*(?:րդ|ին)\s+մաս)?")
CITE_WINDOW = 200  # chars after a code mention in which article refs belong to it


def fetch(session, url):
    time.sleep(DELAY)
    r = session.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def crawl_listings(pages, force):
    """Walk category pagination, append post ids we haven't seen."""
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    if LISTINGS.exists() and not force:
        seen = {json.loads(l)["id"] for l in LISTINGS.open()}
    session = requests.Session()
    # discover total page count from page 1
    first = fetch(session, CATEGORY)
    nums = re.findall(r"/category/newsfeed/awareness/page/(\d+)", first)
    total = min(pages or 10**6, max((int(n) for n in nums), default=1))
    print(f"category pages: {total} (listed so far: {len(seen)})")
    with LISTINGS.open("a") as out:
        for page in range(1, total + 1):
            html = first if page == 1 else fetch(session, f"{CATEGORY}/page/{page}")
            ids = re.findall(r'href="https://iravaban\.net/(\d+)\.html"', html)
            new = [i for i in dict.fromkeys(ids) if i not in seen]
            for post_id in new:
                out.write(json.dumps({"id": post_id, "url": f"{BASE}/{post_id}.html", "page": page}) + "\n")
                seen.add(post_id)
            print(f"page {page}/{total}: +{len(new)} posts ({len(seen)} total)")


def crawl_articles(limit, force):
    """Fetch post HTML for every listed id not already cached."""
    RAW.mkdir(parents=True, exist_ok=True)
    todo = []
    for line in LISTINGS.open():
        row = json.loads(line)
        path = RAW / f"{row['id']}.html"
        if force or not path.exists() or path.stat().st_size < MIN_BYTES:
            todo.append(row)
    if limit:
        todo = todo[:limit]
    print(f"fetching {len(todo)} posts")
    session = requests.Session()
    for n, row in enumerate(todo, 1):
        try:
            html = fetch(session, row["url"])
        except requests.RequestException as e:
            print(f"{row['id']}: FAILED {e}")
            continue
        (RAW / f"{row['id']}.html").write_text(html)
        if n % 25 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} fetched")


def extract_citations(text):
    """All (code, article, part) triples: article refs within CITE_WINDOW chars
    after a code mention, stopping at the next code mention."""
    mentions = list(CODE_RE.finditer(text))
    cites = []
    for i, m in enumerate(mentions):
        idx = int(m.lastgroup[1:]) if m.lastgroup and m.lastgroup.startswith("c") else None
        code = CODE_PATTERNS[idx][1] if idx is not None else None
        if code == "law":
            law_name = next((g for g in m.groups() if g and "»" not in g and len(g) > 2), None)
            code = f"law:{law_name.strip()}" if law_name else "law"
        end = mentions[i + 1].start() if i + 1 < len(mentions) else len(text)
        window = text[m.end():min(m.end() + CITE_WINDOW, end)]
        for art in ARTICLE_RE.finditer(window):
            cites.append({"code": code, "article": art.group(1), "part": art.group(2)})
    # dedupe, keep order
    seen, out = set(), []
    for c in cites:
        key = (c["code"], c["article"], c["part"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def load_chunk_index():
    ids = {json.loads(l)["id"] for l in CHUNKS.open()}
    prefixes = {i.split("#")[0] for i in ids}
    return ids, prefixes


def parse_post(path):
    sel = Selector((RAW / path).read_text())
    title = (sel.css("h1::text").get() or sel.css("title::text").get() or "").strip()
    # the post's own date lives in the article header byline; .entry-date divs
    # elsewhere on the page are latest-news widgets showing today's date
    date = None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", " ".join(sel.css(".article-header .byline ::text").getall()))
    if m:
        date = m.group(1)
    body = " ".join(t.strip() for t in sel.css("section.entry-content ::text").getall() if t.strip())
    body = re.sub(r"\s+", " ", body)
    # drop the trailing law-firm disclaimer boilerplate
    body = re.split(r"Ծանուցում\s*[.:]", body)[0].strip()
    return title, date, body


def extract(min_date):
    ids, prefixes = load_chunk_index()
    OUT.mkdir(parents=True, exist_ok=True)
    CANDIDATES.parent.mkdir(exist_ok=True)
    n_posts = n_cited = n_cand = 0
    with EXTRACTED.open("w") as ex, CANDIDATES.open("w") as cand:
        for path in sorted(RAW.glob("*.html")):
            title, date, body = parse_post(path.name)
            if not title or not body:
                continue
            n_posts += 1
            cites = extract_citations(body)
            for c in cites:
                chunk_id = f"{c['code']}-art-{c['article']}"
                c["chunk_id"] = chunk_id
                c["resolved"] = chunk_id in prefixes
            row = {
                "id": f"irv_{path.stem}",
                "question_armenian": title,
                "answer_text": body,
                "source": "iravaban.net",
                "source_url": f"{BASE}/{path.stem}.html",
                "published": date,
                "citations": cites,
                "expected_article_ids": sorted({c["chunk_id"] for c in cites if c["resolved"]}),
                "auto_extracted": True,
            }
            ex.write(json.dumps(row, ensure_ascii=False) + "\n")
            if cites:
                n_cited += 1
            # benchmark candidate: question-style headline + >=1 citation in corpus
            if "՞" in title and row["expected_article_ids"] and (not min_date or (date or "") >= min_date):
                cand.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_cand += 1
    print(f"posts parsed: {n_posts}, with citations: {n_cited}, benchmark candidates: {n_cand}")
    print(f"-> {EXTRACTED}\n-> {CANDIDATES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listings", action="store_true")
    ap.add_argument("--articles", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--pages", type=int, help="cap listing pages")
    ap.add_argument("--limit", type=int, help="cap article fetches")
    ap.add_argument("--min-date", help="candidates: only posts on/after YYYY-MM-DD")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.listings:
        crawl_listings(args.pages, args.force)
    if args.articles:
        crawl_articles(args.limit, args.force)
    if args.extract:
        extract(args.min_date)
    if not (args.listings or args.articles or args.extract):
        ap.print_help()


if __name__ == "__main__":
    main()
