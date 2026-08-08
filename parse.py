"""Split raw ARLIS act HTML into per-article chunks -> data/chunks.jsonl.

One chunk per article (PLAN D2): the retrieval unit, the citation unit and the
eval unit are all `Հոդված N`. Oversized articles are split into overlapping parts
that share a cite_id, so a citation still points at the article, not the part.

Usage:
    uv run python parse.py               # every act in data/raw/ that's in corpus.json
    uv run python parse.py 176082        # just this one
    uv run python parse.py --type code   # every code (constitution | code | law)
"""
import html
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from config import get

RAW = Path(get("paths.raw"))
CORPUS = Path(get("paths.corpus"))
OUT = Path(get("paths.chunks"))

# The embedding model truncates at 512 tokens, silently. Measured on the Labor Code:
# at article granularity 40/288 chunks blew the limit and 12% of the corpus was
# unreachable — including art. 85 and 113, both expected answers in the eval set.
# So chunks are sized in *tokens*, not chars (Armenian runs ~4.1 chars/token, and
# that ratio is too variable to fake with a char cap).
# 512 - 2 special - "passage: " - title (max 53 tok observed) leaves 400 for the body.
MODEL = get("embedding.model", env="OPENLAW_EMBED_MODEL")
MAX_TOKENS = get("chunking.max_tokens")
OVERLAP_TOKENS = get("chunking.overlap_tokens")
MODEL_LIMIT = get("chunking.model_limit")
RESERVE = get("chunking.reserve_tokens")


def body_budget(title: str) -> int:
    """Body tokens allowed for this article.

    What gets embedded is "passage: {title}\\n{text}", so the title competes with the
    body for the model's 512. Administrative-offence titles run to 140+ tokens, which
    is what pushed 4 chunks over the limit when the budget was a flat constant.
    """
    return max(64, min(MAX_TOKENS, MODEL_LIMIT - RESERVE - ntokens(title)))

SENTENCE = re.compile(r"(?<=[։:])\s+")  # Armenian full stop U+0589, and ASCII colon


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def ntokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])

# Article heading. 20 of the Labor Code's 286 headings nest <A class=anch>/<IMG>/a
# ⚖ case-law link inside the <STRONG>, so allow any inner tag before the word.
ARTICLE = re.compile(
    r"<STRONG>(?:\s|&nbsp;|<(?!/STRONG)[^>]*>)*Հոդված\s+([0-9]+(?:\.[0-9]+)?)\s*\.",
    re.I,
)
# Heading cell is `<STRONG>Հոդված N.</STRONG></P></TD><TD>...<STRONG>{title}</STRONG>`.
TITLE = re.compile(r"<TD[^>]*>(.*?)</TD>", re.I | re.S)
REPEALED = re.compile(r"^\(?\s*(հոդվածն?\s+)?ուժը\s+կորցրել\s+է", re.I)
# The consolidated text deletes fully-repealed articles outright (Labor Code: only
# art. 112 is gone), so the signal that actually survives is a repealed *part* of a
# live article — 19 of them. That's what the refusal behaviour has to key off.
REPEALED_PART = re.compile(r"մասն?\s+ուժը\s+կորցրել\s+է", re.I)
SCALES = "⚖"  # ⚖ marker on the per-article case-law link

DROP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
NEWLINE = re.compile(r"<BR[^>]*>|</P>|</TD>|</TR>|</DIV>", re.I)
TAGS = re.compile(r"<[^>]+>")


def clean(fragment: str) -> str:
    """HTML fragment -> readable plain text."""
    text = DROP.sub(" ", fragment)
    text = NEWLINE.sub("\n", text)
    text = TAGS.sub(" ", text)
    text = html.unescape(text).replace("\xa0", " ").replace(SCALES, " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def hard_split(text: str, budget: int = MAX_TOKENS) -> list[str]:
    """Last-resort token-window split for a single sentence that busts the budget.

    Tax and criminal-procedure articles contain sentences of 600+ tokens; without
    this they were emitted whole and silently truncated at embed time.
    """
    tk = tokenizer()
    ids = tk(text, add_special_tokens=False)["input_ids"]
    return [
        tk.decode(ids[i : i + budget], skip_special_tokens=True).strip()
        for i in range(0, len(ids), budget)
    ]


def units(text: str, budget: int = MAX_TOKENS) -> list[str]:
    """Paragraphs, falling back to sentences, then to a hard token window."""
    out = []
    for para in (p for p in text.split("\n") if p.strip()):
        if ntokens(para) <= budget:
            out.append(para)
            continue
        buf = ""
        for s in SENTENCE.split(para):
            if ntokens(s) > budget:  # one sentence over budget on its own
                if buf:
                    out.append(buf)
                    buf = ""
                out.extend(hard_split(s, budget))
            elif buf and ntokens(f"{buf} {s}") > budget:
                out.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            out.append(buf)
    return out


def split_long(text: str, budget: int = MAX_TOKENS) -> list[str]:
    """Pack paragraphs into <=MAX_TOKENS parts, overlapping by ~OVERLAP_TOKENS.

    Splits on natural boundaries rather than a token window, so no part starts
    mid-sentence — matters when the part is what the LLM ends up reading.
    """
    if ntokens(text) <= budget:
        return [text]

    parts, buf = [], []
    for unit in units(text, budget):
        if buf and ntokens("\n".join(buf + [unit])) > budget:
            parts.append("\n".join(buf))
            # Carry back trailing paragraphs as overlap so a rule split across the
            # boundary is still whole in one of the parts.
            tail, total = [], 0
            for prev in reversed(buf):
                if total + ntokens(prev) > OVERLAP_TOKENS:
                    break
                tail.insert(0, prev)
                total += ntokens(prev)
            buf = tail
        buf.append(unit)
    if buf:
        parts.append("\n".join(buf))
    return [p.strip() for p in parts if p.strip()]


def heading_title(fragment: str) -> str:
    """Second table cell of the heading = the article title."""
    cells = TITLE.findall(fragment)
    return clean(cells[1]).split("\n")[0].strip() if len(cells) > 1 else ""


def act_body(page: str) -> str:
    """Just the act text: `<div id="act_body">` .. `<div id="act_sidebar">`.

    Without this the page header lands in the preamble and the sidebar metadata
    panel (act number, status, amendment history) lands inside the final article.
    """
    start = page.find('id="act_body"')
    if start == -1:
        return page
    # Both ids sit mid-tag; snap to tag boundaries or the leftovers read as text.
    start = page.find(">", start) + 1
    end = page.find('id="act_sidebar"', start)
    return page[start : page.rfind("<", start, end) if end != -1 else len(page)]


def parse_act(docid: str, meta: dict) -> list[dict]:
    raw = act_body((RAW / f"{docid}.html").read_text(encoding="utf-8", errors="replace"))
    slug = meta["slug"]
    url = f"https://www.arlis.am/hy/acts/{docid}/latest"
    article_level = meta.get("article_level", True)

    marks = [(m.start(), m.group(1)) for m in ARTICLE.finditer(raw)]
    # Body runs from the <TABLE> that opens the heading to the next article's table.
    starts = [raw.rfind("<TABLE", 0, pos) for pos, _ in marks]
    starts = [s if s != -1 else pos for s, (pos, _) in zip(starts, marks)]

    chunks: list[dict] = []

    def add(cid: str, cite: str, article: str, title: str, text: str, repealed: bool):
        parts = split_long(text, body_budget(title))
        for i, part in enumerate(parts, 1):
            rec = {
                "id": cid if len(parts) == 1 else f"{cid}#p{i}",
                "cite_id": cite,
                "docid": docid,
                "slug": slug,
                "article": article,
                "title": title,
                "text": part,
                "url": url,
            }
            if repealed:
                rec["repealed"] = True
            if REPEALED_PART.search(part):
                rec["has_repealed_parts"] = True
            chunks.append(rec)

    # Everything before Հոդված 1 is the preamble.
    if starts:
        preamble = clean(raw[: starts[0]])
        if len(preamble) > 200:
            add(f"{slug}-preamble", slug, "", "Նախաբան", preamble, False)

    for i, (start, num) in enumerate(zip(starts, (n for _, n in marks))):
        end = starts[i + 1] if i + 1 < len(starts) else len(raw)
        fragment = raw[start:end]
        title = heading_title(fragment)
        text = clean(fragment)
        cid = f"{slug}-art-{num}"
        cite = cid if article_level else f"law-{docid}"
        repealed = bool(REPEALED.match(title) or REPEALED.search(text[:200]))
        add(cid, cite, num, title, text, repealed)

    return dedupe(chunks)


def dedupe(chunks: list[dict]) -> list[dict]:
    """Drop heading-only stubs and keep the fullest chunk per id.

    The criminal and administrative-offence codes repeat every article heading in a
    contents listing, which the article regex reads as a second occurrence. That
    produced 46 duplicate ids whose bodies were just the heading line — noise that
    would compete with the real article at retrieval time.
    """
    best: dict[str, dict] = {}
    for c in chunks:
        body = c["text"]
        for lead in (f"Հոդված {c['article']}.", c["title"]):
            if lead and body.startswith(lead):
                body = body[len(lead) :].lstrip()
        if not body.strip():  # heading with no article text
            continue
        keep = best.get(c["id"])
        if keep is None or len(c["text"]) > len(keep["text"]):
            best[c["id"]] = c
    return list(best.values())


def main() -> None:
    corpus = json.loads(CORPUS.read_text())
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--type" in sys.argv:
        wanted = sys.argv[sys.argv.index("--type") + 1]
        docids = [d for d, m in corpus.items() if m.get("type") == wanted]
        if not docids:
            sys.exit(f"no documents of type {wanted!r} in {CORPUS}")
        args = [a for a in args if a != wanted]
    else:
        docids = args or [d for d in corpus if (RAW / f"{d}.html").exists()]
    docids = args or docids

    all_chunks: list[dict] = []
    for docid in docids:
        if not (RAW / f"{docid}.html").exists():
            print(f"  {docid}: no raw HTML, run scrape.py first")
            continue
        chunks = parse_act(docid, corpus[docid])
        if not chunks:
            # e.g. 22512: repealed outright, /latest serves a one-line stub with no
            # articles and no preamble worth keeping.
            print(f"  {docid} ({corpus[docid]['slug']}): no content — skipped")
            continue
        articles = len({c["cite_id"] for c in chunks if c["article"]})
        repealed = sum(1 for c in chunks if c.get("repealed"))
        longest = max(len(c["text"]) for c in chunks)
        print(
            f"  {docid} ({corpus[docid]['slug']}): {len(chunks)} chunks, "
            f"{articles} articles, {repealed} repealed, longest {longest} chars"
        )
        all_chunks += chunks

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()
