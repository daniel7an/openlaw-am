# openlaw-am — Plan & Progress

**Open, on-prem legal assistant for Armenian law — grounded in ARLIS, benchmarked in public.**

Hack Armenia 2026. This file is the single source of truth: plan, decisions, progress.
Claude = planner. Team = executors. Update checkboxes as you go.

---

## 1. Pitch skeleton (Evaluate / Reason / Generalize)

1. **Problem:** Legal understanding in Armenia is paywalled (KanonX, Orin AI) or unreliable.
   Published result (ar-lex_graph): naive RAG on Armenian law → **98% hallucinated citations**.
2. **We built:** open, auditable, on-prem-deployable legal Q&A over ARLIS — a citation for
   every claim, refusal when ungrounded.
3. **Evaluate:** 50-question public benchmark (seeded from ar-lex_graph, MIT) —
   retrieval hit@k, citation precision, hallucinated-citation rate, vs BM25 baseline.
4. **Reason:** hedges/refuses on amended-law and out-of-corpus questions.
5. **Generalize:** difficulty breakdown; API-model vs local-model gap, measured.
6. **Beat:** ar-lex_graph Graph-RAG = 50% citation acc / 24% halluc @ 88K tokens & ~120s/query.
   **Target: ≥50% citation accuracy at <10K tokens/query, plus a fully-local row.**

## 2. Hard constraints

- Code freeze **Sun 12:00**, demo **Sun 13:00** (hard 5-min pitch)
- Mentors on-site **Sat 15:00–18:00 only** → end-to-end slice must run by 15:00
- Compute: OpenRouter key → `deepseek/deepseek-v4-pro`, **$120 total** → single-pass RAG, no agent loops
- Open source, reproducible, real data, LLM must earn its place, code written during sprint

## 3. Decision log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Scope Phase 1 = **Labor Code only** (docid 176082) | 18/50 eval Qs touch it, **9 answerable from it alone**; already cached; vertical slice > breadth. Ceiling is real: easy 5/5, medium 4/7, **hard 0/6** — every hard labour Q is cross-instrument. Lifted by D10 |
| D10 | First post-slice step = **+ Constitution (75780) + Civil Code (172116)** | Best marginal return: labour 9→14, all-50 9→28, for two documents. Constitution alone appears in 5 of the 9 mixed labour Qs |
| D2 | Chunking = **one chunk per article** (`Հոդված N`) | Simplest established strategy for statutes; retrieval unit = citation unit = eval unit |
| D3 | Vector DB = **Weaviate** (Docker, BYO vectors) | Team choice; client-side embeddings keep the model swappable |
| D4 | Embeddings local, model = **TBD — reference coming from Tatul** | Placeholder: `intfloat/multilingual-e5-base`; one-line swap in .env |
| D5 | Generation via OpenAI-compatible client, `base_url` from .env | Same code path for DeepSeek (dev) and Ollama (on-prem demo) |
| D6 | Fetch fresh from ARLIS **`/hy/acts/{id}/latest`**; 2023 dump = metadata only | Dump is stale (max date 2023-04-12) |
| D8 | Scraper = **Scrapy** (`scrape.py`, two spiders) | Team choice; gives retries/throttle/robots for free |
| D9 | Always use the **`/latest`** suffix — never `print/act` or `DocumentView.aspx` | Those serve a **2023-05-21 snapshot**: 275 articles vs 286 in the live text. Measured, see §9 |
| D7 | Cite prior work openly (ar-lex_graph, arlis-db) | Credibility with mentors/jury; small ecosystem |

## 4. Current state (env — done)

- [x] Repo scaffold: `pyproject.toml`, `.env.example`, `.gitignore`, `data/corpus.json` (all 20 eval docids + slugs)
- [x] Deps installed (`uv sync`): sentence-transformers, weaviate-client v4, openai, rank-bm25
- [x] Docker up (Colima)
- [x] Labor Code cached: `data/raw/176082.html` (999KB, **286 articles**, current to 10.07.2026)
- [x] Eval set in repo: `eval/qa_dataset.json` (50 Qs, MIT, attributed)
- [x] `scrape.py` — Scrapy, cached, polite; act fetch + group discovery (see §9)
- [ ] `.env` created from `.env.example` with the team's OpenRouter key

## 5. Phase 1 — vertical slice on Labor Code (target: running by 15:00)

### Step 1.1 — `parse.py`: HTML → `data/chunks.jsonl` ✅ DONE
**Spec:**
- Article boundary in raw HTML: each article is a `<TABLE>` whose first cell is
  `<STRONG>Հոդված N.</STRONG>` and second cell `<STRONG>{title}</STRONG>`. Body = HTML until next such table.
  Split point: the `<TABLE` preceding each marker (use `rfind`). N may be decimal (`182.6`).
  ⚠️ **20 of 286 markers nest tags inside the `<STRONG>`** (`<A class=anch>`, `<IMG>`, a `⚖` case-law
  link). A naive `<STRONG>Հոդված N.</STRONG>` misses them (finds 266). Verified pattern:
  `<STRONG>(?:\s|&nbsp;|<(?!/STRONG)[^>]*>)*Հոդված\s+([0-9]+(?:\.[0-9]+)?)\s*\.` → 286/286, no dupes.
- Text cleanup: drop scripts/styles; `<BR>`, `</P>`, `</TD>` → newline; strip tags; `html.unescape`; collapse whitespace.
- Chunk record: `{id, cite_id, docid, slug, article, title, text, url}`
  - `id` = `labor-code-art-{N}` (matches eval's `expected_article_ids` exactly)
  - `cite_id` = same as id for codes; `law-{docid}` for doc-level laws (Phase 2)
  - `url` = `https://www.arlis.am/hy/acts/{docid}/latest` (per D9 — the DocumentView link
    the dataset uses shows the stale 2023 text, so it's the wrong thing to cite to a user)
- **Extract `<div id="act_body">` .. `<div id="act_sidebar">` first.** Otherwise the page header
  lands in the preamble and the sidebar metadata panel (status, amendment history) lands *inside*
  the last article — art. 266 came out at 15.3K chars instead of 211. Snap the slice to tag
  boundaries; both ids sit mid-tag. With the body scoped, only 1 article needs oversize splitting.
- Oversized articles (>6,000 chars): split into parts with ~500-char overlap; ids `...-art-N#p1`, `#p2`;
  same `cite_id`. Preamble before Հոդված 1 → one chunk, `cite_id = labor-code`.
- Repealed articles: ⚠️ **the assumption was wrong.** The consolidated text *deletes* repealed
  articles outright rather than keeping `ուժը կորցրել է` stubs — in the whole Labor Code only
  art. 112 is missing from the 1–266 sequence, and 0 articles carry repeal wording. What does
  survive is **19 articles with a repealed _part_** (`(մասն ուժը կորցրել է 03.05.23 ՀՕ-160-Ն)`),
  flagged `"has_repealed_parts": true`. That's the signal the refusal/amendment behaviour must use.
  Keep the article-level `"repealed"` check anyway for Phase-2 docs that may still carry stubs.

**Acceptance — all passing as of the first run:**
- [x] **288 chunks** = 286 articles + preamble + 1 oversize split (`art-195#p1/#p2`)
- [x] `labor-code-art-83` present, defines the employment contract (…համաձայնություն է աշխատողի և գործատուի միջև…)
- [x] no chunk >8K chars (longest 5,977); ids unique; zero HTML leakage into text
- [x] all **15/15** `labor-code-*` ids the eval set expects exist as `cite_id`s
- [x] spot-checked art. 3.3 (decimal numbering), 113, 266, preamble — boundaries and titles clean

### Step 1.2 — Weaviate up
**Spec:** `docker-compose.yml`: image `semitechnologies/weaviate` (pin latest 1.x), ports 8080 + 50051,
env `DEFAULT_VECTORIZER_MODULE=none`, `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`, volume `./data/weaviate`.
**Acceptance:** `curl localhost:8080/v1/.well-known/ready` → 200.

### Step 1.3 — `index.py`: embed + load
**Spec:**
- Model from `OPENLAW_EMBED_MODEL` (await Tatul's reference; placeholder e5-base).
  ⚠️ e5 models need prefixes: `"passage: "` at index time, `"query: "` at query time. Confirm whether
  the chosen model needs an instruction/prefix convention before indexing.
- Collection `Article`: BYO vectors (`vector_config` self-provided), HNSW + cosine;
  properties: cite_id, slug, article, title, text, url, repealed. Recreate collection on each run (idempotent).
- Batch-insert with client-side vectors; embed `title + "\n" + text`.
**Acceptance:** object count == chunk count; near-instant query response.

### Step 1.4 — retrieval smoke test (before wiring the LLM)
Query top-5 for 3 known eval questions, checking expected article present:
- qa_001 «Ի՞նչ է աշխատանքային պայմանագրի հասկացությունը» → `labor-code-art-83`
- a dismissal/termination question → its expected article
- one `hard` labor question from the set
**Acceptance:** 3/3 hit in top-5. If not: try query-prefix fix, then `hybrid` (BM25+vector) search — Weaviate has it built in; decide and log as D8.

### Step 1.5 — `rag.py`: grounded answer
**Spec:**
- Retrieve top-8 → prompt with numbered context blocks labeled `[Հոդված N]`.
- System prompt contract: answer in Armenian; **every claim cites `[Հոդված N]`**; only use provided
  articles; if context insufficient → say exactly that + suggest where to look; repealed article → must say repealed.
- Output max ~600 tokens. One call per question, no loops. Log token usage per query.
**Acceptance:** qa_001 answer cites Հոդված 83 and nothing hallucinated; an out-of-corpus question
(e.g. «Ինչպե՞ս գրանցել ամուսնություն») → clean refusal, no fake citations.

### Step 1.6 — 15:00 mentor window
Bring: running slice + this file. Ask **"Is this evaluation sound?"** — specifically:
citation-matching rule (exact article match vs partial credit), and whether LLM-judged
answer correctness is worth the tokens.

## 6. Phase 2+ (after mentors, Sat evening → Sun)

- [ ] **First, the moment the slice runs (D10): `uv run python scrape.py 75780 172116`** — Constitution +
      Civil Code, parse + re-index. Labour coverage 9→14 of 18, all-50 9→28. ~10 min. Do this before
      any other Phase-2 work.
- [ ] **Report mixed-corpus Qs as out-of-corpus, not wrong.** On a Labour-only (or partial) corpus the
      9 mixed labour Qs are unanswerable by construction; correct behaviour is answer-the-labour-part +
      refuse the rest. Score them as a separate `out_of_corpus` bucket — silently counting them as
      failures understates us and reads as sloppy evaluation. Raise at the 15:00 mentor window.
- [ ] **Eval harness** (`eval.py`): 50 Qs → retrieval hit@1/3/5, citation precision/recall,
      hallucinated-citation rate (cited ∉ retrieved), token count, latency; per-difficulty breakdown;
      BM25-only baseline row (rank-bm25 or Weaviate keyword mode). Results → `results/*.json` + markdown table.
      Labor-only first (20 Qs), full 50 after Phase-2 scrape.
- [ ] **Scale corpus**: `python scrape.py --phase 2` → all 20 docs; parse doc-level laws (cite_id = `law-{docid}`);
      re-index. Constitution: both docids share slug `constitution`, article ids `constitution-art-N`.
- [ ] **On-prem mode**: install Ollama, pull `gemma3` (4b first, test Armenian quality immediately —
      if broken, try 12b; if still broken, that's a finding, report it); swap base_url in .env; re-run eval → two-row table.
- [ ] **Benchmark extension**: +10–20 Qs on post-2023 amendments (ar-lex_graph's blind spot);
      5–10 hard Qs from datalex.am Cassation fact patterns (manual copy-paste, no scraping).
- [ ] **Spot-check gold answers** vs current law (dataset was built on 2023 texts).
- [ ] **UI**: CLI is acceptable; simple single-page web UI only if ahead of schedule Sun morning.
- [ ] **README**: reproduce in 3 commands; benchmark table; limitations & who-could-be-harmed section
      («informational, not legal advice» + known failure modes).
- [ ] If accessible: run 10 eval Qs through KanonX/Orin free tiers → head-to-head table.
- [ ] Pitch: 5 beats = problem → why-LLM → how-evaluated → numbers → where-it-breaks. Practice once, time it.

## 7. Risks / watchlist

- **Corpus is now 3 years newer than the benchmark.** We index the live text (286 art.); ar-lex_graph's
  gold answers were written against the 2023 text (275 art.). The 11 articles added since
  (3.3, 17.1, 18.1, 59.1, 95.1, 101.1, 102.1, 201.2–201.4, 207.1) plus amended wording mean some
  "wrong" answers may actually be right-and-current. Spot-check before reporting numbers — and note
  this is *our* angle: their blind spot is post-2023 law. Feeds the §6 amendment-questions task.
- **Embedding model TBD** is now the critical-path unknown — indexing blocked on D4 until reference arrives (placeholder OK for pipeline dev).
- Armenian quality of small local models — test at Phase start, not end.
- $120 budget: cap output tokens; cache retrieval results; don't idle-rerun eval (50 Qs × ~5K tok ≈ manageable, but only run full eval deliberately).
- Another legal-RAG team likely in the room → our moat: benchmark + on-prem + per-claim citations.
- ARLIS politeness: 1.5s delay, cache everything, never re-scrape what's on disk.

## 9. How ARLIS is structured (measured 2026-08-08)

**Unit of publication is the whole act, not the article.** One act = one HTML page containing
every article. Articles have no URLs of their own — we split them out locally in `parse.py`.

**URL scheme.** Everything hangs off one numeric `act id` namespace (`/hy/acts/{id}/...`).
Our corpus.json "docids" are act ids too, so no rekeying was needed.

| URL | Serves | Labor Code result |
|-----|--------|-------------------|
| `/hy/acts/{id}/latest` | **consolidated current text** ✅ | 999KB, 286 art., to 10.07.2026 |
| `/hy/acts/{id}` | one frozen version | 786KB, 275 art. |
| `/hy/acts/{id}/print/act` | legacy print view | 649KB, 275 art., **stuck at 2023-05-21** |
| `DocumentView.aspx?docid={id}` | legacy site | same stale text |

The Labor Code is act **176082** (= 51 = 227661; all three alias to the same latest text).
Act 45138 is the *1972* Labor Code, repealed 2005 — do not index it.

**Markup** is legacy HTML in all views (`<P>`, `<TABLE>`, `<STRONG>`, uppercase tags), so the
§1.1 parse spec applies to `/latest` unchanged apart from the nested-tag fix. Bonus signals present
in the text: per-article amendment notes `(83-րդ հոդվածը փոփ. 03.05.23 ՀՕ-160-Ն)` — useful for the
amended-law reasoning dimension — and `⚖` links to related case law.

**There is no subject-group taxonomy.** The UI has a "Դասակարգիչ" (Classifier) string, but it is a
dead translation key — no classifier endpoint exists. So *"all labour laws"* cannot be pulled as a
group; it has to be a curated list of act ids (which is what `data/corpus.json` is).

**Search API** (what `--search` uses), reverse-engineered from `/static/js/search.js`:
`GET /hy/search/page/{n}?{url-encoded JSON of filters}&order_by=` → `{status, message, html}`.
Returns 10 act cards per page; **403s without the `X-Requested-With: XMLHttpRequest` header**.
Useful filters: `simple_text`, `text_filter` (1 = title only, 2 = title+body), `status`
(1 = in force), `act_type` (1 Constitution, 3 constitutional law, 5 code, 7 law),
`exclude_changing_acts`, `include_old_versions`, date ranges.
Filtering matters: `աշխատան` in title = 306 pages raw → 16 acts with in-force + code/law +
exclude-amending. Even then it is keyword noise, not a legal domain (it catches "statistical
**works** programme"), which is why the curated list wins.

**Politeness:** `robots.txt` is 404 (no rules). We self-limit anyway: 1.5s delay, 1 concurrent
request, autothrottle, and everything cached on disk.

## 8. Attribution

- Benchmark seed: [ar-lex_graph](https://github.com/davitsargsyan0/ar-lex_graph) (MIT) — 50-Q Armenian legal QA dataset + published baselines
- Corpus discovery: arlis-db (OpenData Armenia) — metadata only; text fetched fresh from arlis.am
