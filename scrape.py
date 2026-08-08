"""Fetch legal acts from ARLIS as raw HTML into data/raw/{act}.html.

ARLIS has two generations of URL. The current one is act-id based and serves the
*consolidated latest* text; the legacy DocumentView docids serve a frozen snapshot
(the Labor Code docid 176082 is stuck at 2023-05-21). Always prefer act ids.

Usage:
    uv run python scrape.py                    # phase-1 acts (Labor Code)
    uv run python scrape.py --phase 2          # everything in corpus.json
    uv run python scrape.py 51 172116          # explicit ids
    uv run python scrape.py --force 51         # re-fetch even if cached
    uv run python scrape.py --search աշխատան   # discover acts -> data/search/<term>.json

Scrapy spiders. Cached: skips files that already exist. Polite to arlis.am:
1.5s delay, one request at a time, autothrottle on, retries on failure.
"""
import json
import re
import sys
import urllib.parse
from pathlib import Path

import scrapy
from scrapy.crawler import CrawlerProcess

RAW = Path("data/raw")
SEARCH_OUT = Path("data/search")
CORPUS = Path("data/corpus.json")
MIN_BYTES = 5000  # tiny responses are error/redirect pages, not acts

# Tried in order; first response over MIN_BYTES wins. `latest` = consolidated
# current text. The DocumentView fallback is for ids that only exist as legacy docids.
ENDPOINTS = [
    "https://www.arlis.am/hy/acts/{docid}/latest",
    "https://www.arlis.am/hy/acts/{docid}/print/act",
    "https://www.arlis.am/DocumentView.aspx?docid={docid}",
]

# Search-form vocabularies, read off the live form (see docs in PLAN.md §ARLIS).
ACT_TYPE = {"constitution": 1, "constitutional_law": 3, "code": 5, "law": 7}
STATUS = {"in_force": 1, "not_in_force": 2, "suspended": 3, "partly_in_force": 4}

POLITE = {
    "USER_AGENT": "openlaw-am/0.1 (Hack Armenia 2026; open-source legal RAG)",
    "ROBOTSTXT_OBEY": True,
    "DOWNLOAD_DELAY": 1.5,
    "CONCURRENT_REQUESTS": 1,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
    "AUTOTHROTTLE_ENABLED": True,
    "AUTOTHROTTLE_START_DELAY": 1.5,
    "AUTOTHROTTLE_MAX_DELAY": 15,
    "RETRY_TIMES": 3,
    "DOWNLOAD_TIMEOUT": 60,
    "LOG_LEVEL": "INFO",
    "TELNETCONSOLE_ENABLED": False,
}


class ArlisSpider(scrapy.Spider):
    """Download whole acts as raw HTML."""

    name = "arlis"
    custom_settings = POLITE

    def __init__(self, docids, force=False, **kw):
        super().__init__(**kw)
        self.docids = list(docids)
        self.force = force

    async def start(self):
        RAW.mkdir(parents=True, exist_ok=True)
        for docid in self.docids:
            out = RAW / f"{docid}.html"
            if out.exists() and out.stat().st_size > MIN_BYTES and not self.force:
                self.logger.info(f"{docid}: cached ({out.stat().st_size // 1024}KB)")
                continue
            yield self._request(docid, 0)

    def _request(self, docid: str, attempt: int):
        """Request endpoint #attempt for docid, or give up if we're out of them."""
        if attempt >= len(ENDPOINTS):
            self.logger.error(f"{docid}: FAILED on all endpoints")
            return None
        return scrapy.Request(
            ENDPOINTS[attempt].format(docid=docid),
            callback=self.parse,
            errback=self.on_error,
            cb_kwargs={"docid": docid, "attempt": attempt},
            meta={"docid": docid, "attempt": attempt},
            dont_filter=True,
        )

    def parse(self, response, docid: str, attempt: int):
        if len(response.body) <= MIN_BYTES:
            self.logger.warning(
                f"{docid}: too small ({len(response.body)}B) via {response.url}, trying fallback"
            )
            nxt = self._request(docid, attempt + 1)
            if nxt:
                yield nxt
            return

        (RAW / f"{docid}.html").write_bytes(response.body)
        articles = len(set(re.findall(r"Հոդված\s+(\d+(?:\.\d+)?)\s*\.", response.text)))
        self.logger.info(
            f"{docid}: {len(response.body) // 1024}KB, {articles} articles via {response.url}"
        )

    def on_error(self, failure):
        docid, attempt = failure.request.meta["docid"], failure.request.meta["attempt"]
        self.logger.warning(f"{docid}: {failure.value} via {failure.request.url}, trying fallback")
        return self._request(docid, attempt + 1)


class ArlisSearchSpider(scrapy.Spider):
    """Enumerate acts matching a search, via the site's own results API.

    The UI POSTs to /hy/search, then the page pulls results from
    GET /hy/search/page/{n}?<url-encoded JSON of the filters>&order_by=
    which returns {status, message, html}. That endpoint 403s without the
    XMLHttpRequest header.
    """

    name = "arlis_search"
    custom_settings = POLITE | {"DEFAULT_REQUEST_HEADERS": {"X-Requested-With": "XMLHttpRequest"}}

    CARD = re.compile(r'href="/hy/acts/(\d+)/latest"[^>]*>\s*<span[^>]*>\s*(.*?)\s*</span>', re.S)
    PAGE = re.compile(r'\{"text":"(\d+)","value":"\d+"\}')

    def __init__(self, term, max_pages=20, **kw):
        super().__init__(**kw)
        self.term = term
        self.max_pages = int(max_pages)
        self.found: dict[str, str] = {}
        # Title-only, in force, real legislation, no amending acts.
        self.params = {
            "simple_text": term,
            "text_filter": "1",
            "status": str(STATUS["in_force"]),
            "act_type": f"{ACT_TYPE['code']},{ACT_TYPE['law']}",
            "exclude_changing_acts": "1",
        }

    def _url(self, page: int) -> str:
        q = urllib.parse.quote(json.dumps(self.params, ensure_ascii=False))
        return f"https://www.arlis.am/hy/search/page/{page}?{q}&order_by="

    async def start(self):
        yield scrapy.Request(self._url(1), cb_kwargs={"page": 1}, dont_filter=True)

    def parse(self, response, page: int):
        html = json.loads(response.text)["html"]
        for act, title in self.CARD.findall(html):
            self.found.setdefault(act, re.sub(r"<[^>]+>", "", title).strip())

        pages = [int(p) for p in self.PAGE.findall(html)]
        last = max(pages) if pages else 1
        self.logger.info(f"page {page}/{last}: {len(self.found)} acts so far")

        if page == 1 and last > self.max_pages:
            self.logger.warning(f"{last} pages available, stopping at --max-pages {self.max_pages}")
        if page < min(last, self.max_pages):
            yield scrapy.Request(
                self._url(page + 1), cb_kwargs={"page": page + 1}, dont_filter=True
            )

    def closed(self, reason):
        SEARCH_OUT.mkdir(parents=True, exist_ok=True)
        out = SEARCH_OUT / f"{self.term}.json"
        out.write_text(json.dumps(self.found, ensure_ascii=False, indent=2))
        self.logger.info(f"wrote {len(self.found)} acts -> {out}")


def main() -> None:
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("-")]
    process = CrawlerProcess()

    if "--search" in argv:
        term = args[0] if args else "աշխատան"
        max_pages = args[1] if len(args) > 1 else 20
        process.crawl(ArlisSearchSpider, term=term, max_pages=max_pages)
    else:
        corpus = json.loads(CORPUS.read_text())
        phase = 2 if "--phase" in argv and "2" in args else 1
        explicit = [a for a in args if not (a == "2" and phase == 2)]
        docids = explicit or [d for d, m in corpus.items() if m["phase"] <= phase]
        process.crawl(ArlisSpider, docids=docids, force="--force" in argv)

    process.start()


if __name__ == "__main__":
    main()
