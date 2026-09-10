"""Tests for the web_fetch tool, against a small local HTTP server (same
pattern as test_llm.py's fake vLLM server -- no network access needed)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from unittest import mock

from intelligent_agents_chat.tools import webfetch

SHORT_HTML = b"""
<html><head><title>Short</title></head>
<body><article><h1>Hello</h1><p>This is a short article about testing.</p></article></body></html>
"""

_PARAGRAPH = "This is a long, repeated sentence about testing web content extraction. "


def _article_html(title: str, paragraph_count: int) -> bytes:
    paragraphs = "".join(
        f"<p>{_PARAGRAPH * 5} Paragraph number {i}.</p>" for i in range(paragraph_count)
    )
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><article><h1>{title}</h1>{paragraphs}</article></body></html>"
    ).encode()


# Extracted text sizes below are approximate but stable for this fixed input --
# see the module docstring's VERBATIM_CHAR_LIMIT/MAX_CONDENSE_INPUT_CHARS.
MEDIUM_HTML = _article_html("Medium", 15)  # ~5.7k chars extracted: condensed, not truncated
LONG_HTML = _article_html("Long", 80)  # ~30k chars extracted: condensed AND truncated

# A "later section" marker embedded far enough in that naive head-truncation
# (the first MAX_CONDENSE_INPUT_CHARS of ~30k) would miss it, but a
# word-overlap search should still find and window around it.
MARKER_TEXT = "Unique Later Section Marker"
MARKER_PARAGRAPH_INDEX = 60


def _marked_article_html() -> bytes:
    paragraphs = "".join(
        (
            f"<p>{MARKER_TEXT}. {_PARAGRAPH * 5} Paragraph number {i}.</p>"
            if i == MARKER_PARAGRAPH_INDEX
            else f"<p>{_PARAGRAPH * 5} Paragraph number {i}.</p>"
        )
        for i in range(80)
    )
    return (
        f"<html><head><title>Marked</title></head><body><article><h1>Marked</h1>"
        f"{paragraphs}</article></body></html>"
    ).encode()


MARKED_HTML = _marked_article_html()


# The exact real-world case this was built for: the model's search term
# ("roman period") doesn't literally appear on the page (which says "Roman
# era") -- only the word "roman" is shared. Padded with filler well past
# VERBATIM_CHAR_LIMIT so the search path actually runs instead of the whole
# (short) page just being returned verbatim regardless of search_terms.
def _roman_html() -> bytes:
    filler = "".join(f"<p>{_PARAGRAPH * 5} Filler {i}.</p>" for i in range(20))
    return (
        f"<html><head><title>History</title></head><body><article><h1>History</h1>"
        f"{filler}"
        f"<h2>Roman era</h2>"
        f"<p>During the Roman era, many roads were built across the provinces.</p>"
        f"<h2>Medieval period</h2>"
        f"<p>During medieval times, many castles were built across the land.</p>"
        f"</article></body></html>"
    ).encode()


ROMAN_HTML = _roman_html()


class _FakeWebHandler(BaseHTTPRequestHandler):
    routes = {
        "/short": (200, "text/html; charset=utf-8", SHORT_HTML),
        "/medium": (200, "text/html; charset=utf-8", MEDIUM_HTML),
        "/long": (200, "text/html; charset=utf-8", LONG_HTML),
        "/marked": (200, "text/html; charset=utf-8", MARKED_HTML),
        "/roman": (200, "text/html; charset=utf-8", ROMAN_HTML),
        "/not-html": (200, "application/octet-stream", b"binary-ish content"),
        "/missing": (404, "text/plain", b"not found"),
    }
    request_counts: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        _FakeWebHandler.request_counts[self.path] = (
            _FakeWebHandler.request_counts.get(self.path, 0) + 1
        )
        status, content_type, body = self.routes.get(self.path, (404, "text/plain", b""))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:  # noqa: A002
        pass  # keep test output quiet


class WebFetchToolTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeWebHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join()

    def setUp(self) -> None:
        # Process-wide mutable state shared across every test.
        webfetch._page_cache.clear()
        _FakeWebHandler.request_counts.clear()

    # -- basic fetch (no search_terms) -------------------------------------

    async def test_missing_url_is_a_reported_error(self) -> None:
        result = await webfetch.run({})
        self.assertTrue(result.startswith("Error:"))

    async def test_url_without_a_scheme_is_rejected(self) -> None:
        result = await webfetch.run({"url": "example.com/short"})
        self.assertTrue(result.startswith("Error:"))

    async def test_short_page_is_returned_verbatim(self) -> None:
        result = await webfetch.run({"url": f"{self.base_url}/short"})
        self.assertIn("short article about testing", result)

    async def test_missing_page_is_a_reported_http_error(self) -> None:
        result = await webfetch.run({"url": f"{self.base_url}/missing"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("404", result)

    async def test_non_html_content_type_is_rejected(self) -> None:
        result = await webfetch.run({"url": f"{self.base_url}/not-html"})
        self.assertTrue(result.startswith("Error:"))

    async def test_medium_page_is_condensed_via_the_subagent(self) -> None:
        fake_run = mock.AsyncMock(return_value="condensed summary")
        with mock.patch("intelligent_agents_chat.tools.webfetch.subagent.run", fake_run):
            result = await webfetch.run({"url": f"{self.base_url}/medium", "topic": "testing"})

        self.assertEqual(result, "condensed summary")
        fake_run.assert_awaited_once()
        task = fake_run.await_args.args[0]["task"]
        self.assertIn("testing", task)
        self.assertIn("Paragraph number 0", task)

    async def test_condense_task_instructs_the_subagent_to_keep_figures_exact(self) -> None:
        # Real case this addresses: the same table cell was summarized as a
        # different number on different tries (e.g. "3.662 Millionen" vs.
        # "3.878 Millionen") -- an LLM condensation pass is lossy for exact
        # figures unless explicitly told not to paraphrase them.
        fake_run = mock.AsyncMock(return_value="condensed summary")
        with mock.patch("intelligent_agents_chat.tools.webfetch.subagent.run", fake_run):
            await webfetch.run({"url": f"{self.base_url}/medium"})

        task = fake_run.await_args.args[0]["task"]
        self.assertIn("exactly as written", task)

    async def test_long_page_without_search_terms_condenses_from_the_start(self) -> None:
        fake_run = mock.AsyncMock(return_value="condensed summary")
        with mock.patch("intelligent_agents_chat.tools.webfetch.subagent.run", fake_run):
            result = await webfetch.run({"url": f"{self.base_url}/long"})

        task = fake_run.await_args.args[0]["task"]
        self.assertIn("Paragraph number 0", task)
        self.assertIn("page's start", task)
        self.assertIn("condensed summary", result)
        self.assertIn("page's start", result)

    async def test_repeated_fetch_of_the_same_url_does_not_redownload(self) -> None:
        await webfetch.run({"url": f"{self.base_url}/short"})
        await webfetch.run({"url": f"{self.base_url}/short"})

        self.assertEqual(_FakeWebHandler.request_counts.get("/short"), 1)

    # -- with search_terms --------------------------------------------------

    async def test_search_terms_missing_search_terms_is_ignored_not_an_error(self) -> None:
        # search_terms is optional -- an empty/absent list just means "read
        # from the start", same as not passing it at all.
        result = await webfetch.run({"url": f"{self.base_url}/short", "search_terms": []})
        self.assertIn("short article about testing", result)

    async def test_search_terms_accepts_a_bare_string_instead_of_a_list(self) -> None:
        result = await webfetch.run({"url": f"{self.base_url}/short", "search_terms": "testing"})
        self.assertIn("short article about testing", result)

    async def test_search_term_not_matching_any_word_is_reported_not_found(self) -> None:
        # /marked is long enough that the search path actually runs -- on a
        # page short enough to return verbatim, search_terms wouldn't matter
        # either way, which wouldn't test anything about search_terms itself.
        result = await webfetch.run(
            {"url": f"{self.base_url}/marked", "search_terms": ["quantum computing"]}
        )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("none of", result)

    async def test_search_finds_a_differently_worded_section_via_shared_words(self) -> None:
        # The real-world case this feature targets: "roman period" doesn't
        # appear verbatim -- the page says "Roman era". Only "roman" is
        # shared, and that alone should still be enough to match.
        result = await webfetch.run(
            {"url": f"{self.base_url}/roman", "search_terms": ["roman period"]}
        )
        self.assertFalse(result.startswith("Error:"))
        self.assertIn("Roman era", result)

    def test_search_words_keeps_decimal_numbers_intact(self) -> None:
        # Regression test for a real bug this was built to fix: "3.38" was
        # shredded by the period into "3" and "38", both too short to survive
        # the length filter -- silently turning a whole list of specific
        # figures (population counts, prices, ...) into no search signal at
        # all. A short *number* is specific enough to keep even though a
        # short common *word* ("the") still isn't.
        words = webfetch._search_words(["3.38 3.385 hello 1990 the"])
        self.assertEqual(words, {"3.38", "3.385", "hello", "1990"})

    async def test_cluster_radius_prefers_a_tight_cluster_over_scattered_matches(self) -> None:
        # Regression test for a real bug this was built to fix: scoring
        # with a large radius (the excerpt's whole window_size, as it used
        # to) rewards a region with more matches loosely scattered across a
        # huge span over a region where matches are actually clustered
        # together (e.g. a heading and the paragraphs right after it).
        # Five "loose" matches 600 chars apart (0, 600, ..., 2400), then a
        # "tight" cluster of three matches 8 chars apart, far away.
        buf = ["."] * 2410
        for position in range(0, 2401, 600):
            buf[position : position + 7] = list("keyword")
        text = "".join(buf) + ("." * 100) + "keyword keyword keyword end"

        wide_radius = webfetch._best_window(
            text, ["keyword"], window_size=30, context_before=0, cluster_radius=3000
        )
        narrow_radius = webfetch._best_window(
            text, ["keyword"], window_size=30, context_before=0, cluster_radius=50
        )

        self.assertNotIn("keyword keyword keyword", wide_radius or "")
        self.assertIn("keyword keyword keyword", narrow_radius or "")

    async def test_search_reaches_a_section_far_past_the_head_truncation_point(self) -> None:
        # A search_term that shares 3 of MARKER_TEXT's 4 words, not an exact
        # match -- and paragraph 60 is well past where naive head-truncation
        # (first ~20k of ~30k extracted chars) would stop.
        fake_run = mock.AsyncMock(return_value="condensed summary")
        with mock.patch("intelligent_agents_chat.tools.webfetch.subagent.run", fake_run):
            result = await webfetch.run(
                {
                    "url": f"{self.base_url}/marked",
                    "search_terms": ["Unique Later Section Beacon"],
                }
            )

        self.assertEqual(
            result,
            "condensed summary\n\n[Note: this is an excerpt around a match for the given search_terms, not the full page.]",
        )
        task = fake_run.await_args.args[0]["task"]
        self.assertIn(MARKER_TEXT, task)
        self.assertIn(f"Paragraph number {MARKER_PARAGRAPH_INDEX}", task)

    async def test_search_topic_alone_cannot_satisfy_a_search(self) -> None:
        # topic matching the page is not enough on its own -- search_terms
        # must actually share words with the page, or this must fail. (Made
        # up, deliberately not sharing any word with MARKED_HTML's fixed
        # boilerplate paragraph text.)
        result = await webfetch.run(
            {
                "url": f"{self.base_url}/marked",
                "search_terms": ["zzqwerty nonexistentterm blorpsnaffle"],
                "topic": MARKER_TEXT,
            }
        )
        self.assertTrue(result.startswith("Error:"))

    async def test_topic_is_excluded_from_the_search_vocabulary(self) -> None:
        # Regression test for the real bug this was built to fix: topic
        # often shares words with the whole page (e.g. a Wikipedia article's
        # own subject, mentioned hundreds of times) -- mixing it into the
        # same word pool as search_terms drowns out search_terms' far
        # rarer, more specific words with noise. topic must stay out of it.
        fake_best_window = mock.Mock(return_value="some excerpt")
        with mock.patch("intelligent_agents_chat.tools.webfetch._best_window", fake_best_window):
            await webfetch.run(
                {
                    # /marked is long enough that the search path (and thus
                    # _best_window) actually runs.
                    "url": f"{self.base_url}/marked",
                    "search_terms": ["testing"],
                    "topic": "some unrelated broad topic",
                }
            )

        self.assertEqual(fake_best_window.call_args.args[1], ["testing"])

    async def test_search_reuses_the_cache_from_a_prior_plain_fetch(self) -> None:
        await webfetch.run({"url": f"{self.base_url}/short"})
        await webfetch.run({"url": f"{self.base_url}/short", "search_terms": ["testing"]})

        self.assertEqual(_FakeWebHandler.request_counts.get("/short"), 1)

    # -- registry -----------------------------------------------------------

    def test_tool_schema_name_matches_the_registry_key(self) -> None:
        self.assertEqual(webfetch.TOOL.name, webfetch.TOOL.schema["function"]["name"])


if __name__ == "__main__":
    unittest.main()
