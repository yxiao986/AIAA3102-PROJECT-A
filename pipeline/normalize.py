"""Independently-switchable text normalization steps for Ticket 2.

Each step is a plain str -> str function so it can be tested in isolation. `normalize`
composes whichever steps are turned on, in a fixed order, and leaves the text completely
untouched if no step is requested (so the "no cleaning" variant matches the Ticket 1
baseline exactly).

Case is handled separately: TfidfVectorizer lowercases by default, so testing whether case
carries signal means disabling that at the vectorizer, not touching the text here. Use
`vectorizer_kwargs(preserve_case=True)` to get the matching TfidfVectorizer kwarg.
"""
import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # regional indicators (flag emoji)
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def strip_mentions(text: str) -> str:
    return MENTION_RE.sub(" ", text)


def strip_hashtag_symbol(text: str) -> str:
    """Drop the leading '#' but keep the word, e.g. '#earthquake' -> 'earthquake'."""
    return HASHTAG_RE.sub(r"\1", text)


def unescape_html(text: str) -> str:
    """Un-escape HTML entities, e.g. '&amp;' -> '&'."""
    return html.unescape(text)


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


STEP_FUNCS = {
    "strip_urls": strip_urls,
    "strip_mentions": strip_mentions,
    "strip_hashtag_symbol": strip_hashtag_symbol,
    "unescape_html": unescape_html,
    "strip_emoji": strip_emoji,
}

# Fixed application order: html entities first (so a decoded '&amp;' can't accidentally
# feed a later regex), then structural strips, then emoji last.
STEP_ORDER = ["unescape_html", "strip_urls", "strip_mentions", "strip_hashtag_symbol", "strip_emoji"]


def normalize(
    text: str,
    *,
    strip_urls: bool = False,
    strip_mentions: bool = False,
    strip_hashtag_symbol: bool = False,
    unescape_html: bool = False,
    strip_emoji: bool = False,
) -> str:
    """Apply the requested steps, in STEP_ORDER. No flags set -> text returned unchanged."""
    flags = {
        "strip_urls": strip_urls,
        "strip_mentions": strip_mentions,
        "strip_hashtag_symbol": strip_hashtag_symbol,
        "unescape_html": unescape_html,
        "strip_emoji": strip_emoji,
    }
    if not any(flags.values()):
        return text

    for step in STEP_ORDER:
        if flags[step]:
            text = STEP_FUNCS[step](text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_series(series, **flags):
    return series.map(lambda text: normalize(text, **flags))


def vectorizer_kwargs(preserve_case: bool = False) -> dict:
    """TfidfVectorizer kwargs for the case switch. preserve_case=True disables the
    vectorizer's default lowercasing so case-sensitive tokens ('URGENT' vs 'urgent')
    survive into the vocabulary."""
    return {"lowercase": False} if preserve_case else {}
