"""For each source that reaches no measurement, ask the PROVIDER, not our code.

R62 found sources whose every row sits inside the seal. The obvious next move
is "backfill them", and for two of them that turned out to be one word in a
config file -- `period: 2y` where the frame it claimed to match said 30y. VIX
went from 0 explorable rows to 6,791 and sec_filings from 0 to 50,055, at no
extra request (#298).

The rest are a different question, and it is a question about the SUPPLIER: is
there any free way to obtain a date before the seal? Guessing from the name of
a provider is how two of my own conclusions nearly went wrong, so each one is
asked directly and the answer is whatever comes back.

Nothing here writes to the database, and no key is ever printed -- only whether
one is present and how long it is.

    python scripts/diagnostics/can_a_sealed_source_be_backfilled_for_free.py
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from src.pipeline.sealed_period import SEAL_START  # noqa: E402

UA = {"User-Agent": "TradingBot/2.0",
      "Accept": "application/rss+xml, application/xml, text/xml, */*"}

#: How many feeds to ask. The answer has never differed between them and the
#: probe is a courtesy to other people's servers, not a crawl.
FEEDS_PROBED = 10


def say(source: str, reach: str, evidence: str) -> None:
    print(f"{source:<26}{reach:<26}{evidence}")


def _dates_in(text: str) -> list[dt.date]:
    stamps = (re.findall(r"<pubDate>([^<]+)</pubDate>", text)
              or re.findall(r"<updated>([^<]+)</updated>", text))
    found: list[dt.date] = []
    for stamp in stamps:
        for shape in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                found.append(dt.datetime.strptime(stamp.strip(), shape).date())
                break
            except ValueError:
                continue
        else:
            try:
                found.append(dt.date.fromisoformat(stamp[:10]))
            except ValueError:
                pass
    return found


def economic_calendar() -> None:
    """ForexFactory publishes one file per week. Are the older ones kept?"""
    base = "https://nfs.faireconomy.media"
    served = []
    for leaf in ("ff_calendar_thisweek.json", "ff_calendar_lastweek.json",
                 "ff_calendar_nextweek.json", "ff_calendar_2020-01-06.json"):
        try:
            response = requests.get(f"{base}/{leaf}", headers=UA, timeout=30)
        except Exception as error:  # noqa: BLE001 - a probe reports
            served.append(f"{leaf.split('_')[-1][:12]}={type(error).__name__}")
            continue
        served.append(f"{leaf.split('_')[-1][:12].replace('.json','')}="
                      f"{response.status_code}")
    say("economic_calendar", "this week only", "  ".join(served))


def fear_greed() -> None:
    """CNN's dataviz endpoint. Does any parameter widen its window?"""
    base = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    for suffix, label in ((f"", "no parameter"),
                          ("?start=2011-01-01", "?start=2011"),
                          ("/2011-01-01", "/2011 (path form)")):
        try:
            response = requests.get(base + suffix, headers=UA, timeout=30)
            if response.status_code != 200:
                say("fear_greed_data", label, f"HTTP {response.status_code}")
                continue
            points = response.json().get(
                "fear_and_greed_historical", {}).get("data", [])
            oldest = dt.datetime.fromtimestamp(
                points[0]["x"] / 1000, dt.UTC).date() if points else None
            say("fear_greed_data", label,
                f"{len(points)} points, oldest {oldest}")
        except Exception as error:  # noqa: BLE001
            say("fear_greed_data", label, f"{type(error).__name__}")


def google_news() -> None:
    url = ("https://news.google.com/rss/search?q=AAPL"
           "&hl=en-US&gl=US&ceid=US:en")
    try:
        found = _dates_in(requests.get(url, headers=UA, timeout=30).text)
        say("google_news", "search feed",
            f"{len(found)} items, oldest {min(found) if found else '?'}")
    except Exception as error:  # noqa: BLE001
        say("google_news", "search feed", type(error).__name__)


def reddit() -> None:
    try:
        found = _dates_in(requests.get(
            "https://www.reddit.com/r/stocks/.rss", headers=UA, timeout=30).text)
        say("sociological_sentiment", "subreddit feed",
            f"{len(found)} items, oldest {min(found) if found else '?'}")
    except Exception as error:  # noqa: BLE001
        say("sociological_sentiment", "subreddit feed", type(error).__name__)


def rss_news() -> None:
    """The 26 configured feeds. How far back does the deepest one reach?"""
    from src.config.unified_config_manager import UnifiedConfigManager

    feeds = (UnifiedConfigManager().get_config("knowledge_base")
             or {}).get("rss_feeds", [])
    oldest: list[dt.date] = []
    reached = 0
    for entry in feeds[:FEEDS_PROBED]:
        url = entry["url"] if isinstance(entry, dict) else entry
        try:
            found = _dates_in(requests.get(url, headers=UA, timeout=25).text)
        except Exception:  # noqa: BLE001
            continue
        if found:
            reached += 1
            oldest.append(min(found))
    say("rss_news", f"{reached}/{min(len(feeds), FEEDS_PROBED)} feeds answered",
        f"deepest reaches {min(oldest) if oldest else '?'} "
        f"of {len(feeds)} configured")


def newsapi() -> None:
    """Ask for a date before the seal and print what the provider says."""
    try:
        from src.core.security.secure_secrets_manager import SecretsManager
        SecretsManager()
    except Exception:  # noqa: BLE001 - the env may already be loaded
        pass
    key = os.getenv("NEWS_API_KEY")
    if not key:
        say("newsapi_articles", "no key in environment", "")
        return
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "Apple", "from": "2020-01-01", "to": "2020-01-07",
                    "pageSize": 5, "apiKey": key},
            headers=UA, timeout=45)
        body = response.json()
        say("newsapi_articles", f"HTTP {response.status_code} "
            f"({len(key)}-char key)",
            f"{body.get('code', '')}: {str(body.get('message', ''))[:80]}")
    except Exception as error:  # noqa: BLE001
        say("newsapi_articles", "request failed", type(error).__name__)


def main() -> int:
    seal = SEAL_START.date() if hasattr(SEAL_START, "date") else SEAL_START
    print(f"the seal starts {seal}. A source is BACKFILLABLE only if it "
          f"serves dates earlier than that, free.\n")
    print(f"{'source':<26}{'what was asked':<26}what came back")
    print("-" * 100)
    economic_calendar()
    fear_greed()
    google_news()
    reddit()
    rss_news()
    newsapi()
    print()
    print("Read the dates, not the HTTP codes: every one of these answers 200 "
          "for TODAY.\nA source that cannot serve a date before the seal "
          "cannot contribute to any\nmeasurement this project makes, however "
          "healthy its collector looks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
