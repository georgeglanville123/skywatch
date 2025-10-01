
import os
import json
import time
import random
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
from urllib.parse import urlparse
from string import Template

import httpx
import gspread
import trafilatura
import google.generativeai as genai
from flask import Flask, jsonify, request
from google.oauth2.service_account import Credentials
from google.auth import default as adc_default
from zoneinfo import ZoneInfo
from ipaddress import ip_address, ip_network
from email.utils import parsedate_to_datetime

# ────────────────────────── Config ──────────────────────────
TERM_GROUPS = [{
    "ai":  ["NTN", "Low Earth Orbit (LEO)", "Medium Earth Orbit (MEO)", "Geostationary Orbit (GEO)", "Multi-orbit", "High-altitude platform station (HAPS)", "Unmanned aerial vehicle (UAV)", "Satellite", "direct-to-device (D2D)", "terrestrial network", "Non-terrestrial network (NTN)", "direct-to-handset (D2H)", "direct-to-cell (D2H)", "constellation", "spectrum", "MSS"
],
    "act": ["partnership", "implementation", "pilot", "rollout"]
}]

# Refined list (100)
COMPANIES = [
    "AIS", "Altice", "America Movil", "AT&T", "Axiata", "Bell Canada", "Bharti Airtel",
    "Bouygues Telecom", "BT", "Charter Communications", "China Mobile", "China Telecom",
    "China Unicom", "Chunghwa Telecom", "CK Hutchison", "Comcast", "Cox Communications",
    "CYTA", "Deutsche Telekom", "du", "e&", "Elisa", "Entel",
    "Ethio Telecom", "Globe Telecom", "Iliad", "KDDI", "KPN", "KT", "LG Uplus",
    "Liberty Global", "Lumen Technologies", "MASMOVIL", "Maxis", "Millicom", "MTN",
    "NTT Docomo", "Omantel", "Ooredoo", "Orange", "PTCL", "PLDT", "Proximus",
    "Rakuten Mobile", "Reliance Jio", "Rogers Communications", "Safaricom", "stc",
    "Singtel", "SK Telecom", "SoftBank", "Spark New Zealand", "Sprint", "StarHub",
    "Sunrise Communications", "Swisscom", "Taiwan Mobile", "Tata Communications", "Tele2",
    "Telecom Argentina", "Telecom Egypt", "TIM", "Telefonica", "A1 Telekom Austria",
    "Telekom Malaysia", "Telenet", "Telenor", "Telia", "Telkom (South Africa)",
    "Telkom Indonesia", "Telstra", "Telus", "TPG Telecom", "Turkcell", "Veon", "Verizon",
    "Viettel", "Virgin Media O2", "Vodafone", "Wind Tre", "Zain", "T-Mobile US", "Vodacom",
    "Telkomsel", "Indosat Ooredoo Hutchison", "TIM Brasil", "Vivo", "Claro", "Mobily",
    "Airtel Africa"
]

BINARY_EXTS = (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx")
DOC_EXTS = BINARY_EXTS

MAX_RESULTS_PER_QUERY = 2
MAX_TOTAL_RESULTS     = 100
MAX_FETCH             = 25
CSE_TIMEOUT_SEC       = 30.0
FETCH_TIMEOUT_SEC     = 20.0
GEMINI_MODEL          = "gemini-1.5-flash"

# Time filtering knobs
TIME_MODE = os.getenv("TIME_MODE", "calendar")  # "calendar" or "rolling"
TIME_DAYS = max(1, int(os.getenv("TIME_DAYS", "7")))  # guard against 0/negatives
TIME_ZONE = "Europe/London"                     # used for calendar mode

# CSE pacing
CSE_BASE_DELAY_SEC = float(os.getenv("CSE_BASE_DELAY_SEC", "1.0"))
CSE_MAX_RETRIES    = int(os.getenv("CSE_MAX_RETRIES", "6"))
CSE_BACKOFF_FACTOR = float(os.getenv("CSE_BACKOFF_FACTOR", "1.8"))

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

AI_HINTS  = tuple(w.lower() for w in TERM_GROUPS[0]["ai"])
ACT_HINTS = tuple(w.lower() for w in TERM_GROUPS[0]["act"])

ACTION_VERBS = (
    "announce","launch","pilot","rollout","roll-out","implement",
    "deploy","deployment","go-live","trial","proof of concept","poc"
)
BAD_HINTS = (
    "job","hiring","career","vacancy","investor","ir","earnings",
    "q1","q2","q3","q4","whitepaper","webinar","rfp","tender",
    "brochure","datasheet"
)

# Extended to catch common patterns
PRESS_PATHS_SEGMENTS = {"press", "press-releases", "press-release", "news", "media", "newsroom"}

AGG_DOMAINS = {"news.google.com","news.yahoo.com","medium.com","substack.com","reddit.com"}

FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; telco-genai-bot/1.0)"}

# ────────────────────────── Helpers ──────────────────────────
def tail(v: str) -> str:
    return v[-6:] if v and v != "MISSING" else "MISSING"

def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def _extract_top_level_json(text: str) -> dict:
    """
    Robust-ish JSON object extractor:
    1) strip code fences
    2) try direct json.loads
    3) scan for first balanced {...} block and parse it
    """
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.I)
    s = re.sub(r'\s*```$', '', s)

    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = s.find('{', start + 1)
    return {}

def _has_any(text: str, terms) -> bool:
    """
    Boundary matcher that works for tokens like 'AT&T', 'e&', and short names.
    It treats boundaries as start/end or any non-alphanumeric char.
    """
    if not text:
        return False
    t = text.lower()
    for term in terms:
        if not term:
            continue
        pat = r'(?<![0-9a-z])' + re.escape(term.lower()) + r'(?![0-9a-z])'
        if re.search(pat, t):
            return True
    return False

def _host_matches_any(host: str, domains: set) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in domains)

def _is_probably_press_path(path: str) -> bool:
    path = (path or "").lower()
    segments = {seg for seg in path.strip("/").split("/") if seg}
    return any(seg in PRESS_PATHS_SEGMENTS for seg in segments)

def _etld_plus_one(host: str) -> str:
    # Lightweight eTLD+1 (not perfect for multi-part TLDs, but better than raw netloc)
    parts = host.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.lower()

def score_item(it: dict) -> int:
    title   = (it.get("title") or "")
    snippet = (it.get("snippet") or "")
    url     = (it.get("link")  or "")
    company = (it.get("company") or "")

    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    host = (parsed.netloc or "").lower()

    score = 0
    # Company match
    if _has_any(title, [company]):      score += 3
    elif _has_any(snippet, [company]):  score += 1

    # AI/action hints
    if _has_any(title, AI_HINTS):       score += 3
    if _has_any(title, ACT_HINTS):      score += 3
    if _has_any(snippet, AI_HINTS):     score += 1
    if _has_any(snippet, ACT_HINTS):    score += 1

    # Action verbs
    if _has_any(title, ACTION_VERBS):   score += 2
    if _has_any(snippet, ACTION_VERBS): score += 1

    if _is_probably_press_path(path):   score += 1

    # Domain-based tweaks (handle subdomains too)
    if _host_matches_any(host, AGG_DOMAINS): score -= 2

    # Company domain light boost (very fuzzy)
    company_key = re.sub(r'[^a-z0-9]', '', company.lower())
    host_key = re.sub(r'[^a-z0-9]', '', host.lower())
    if company_key and company_key in host_key:
        score += 1

    # Demotions
    if path.endswith(DOC_EXTS):                      score -= 3
    if _has_any(title, BAD_HINTS) or _has_any(snippet, BAD_HINTS): score -= 3

    return score

def cap_by_domain(items, per_domain=2):
    counts, out = {}, []
    for it in items:
        host = urlparse(it["link"]).netloc.lower()
        key = _etld_plus_one(host)
        if counts.get(key, 0) < per_domain:
            out.append(it)
            counts[key] = counts.get(key, 0) + 1
    return out

# ───────────────────── Query Builder ─────────────────────
def build_queries() -> List[Tuple[str, str]]:
    g = TERM_GROUPS[0]
    ai_clause  = "(" + " OR ".join(g["ai"])  + ")"
    act_clause = "(" + " OR ".join(g["act"]) + ")"
    return [
        (f'{ai_clause} AND {act_clause} AND "{company}"', company)
        for company in COMPANIES
    ]

# ───────────────────── Google CSE Search ─────────────────────
def _parse_retry_after(val: str | None) -> float | None:
    if not val:
        return None
    val = val.strip()
    if val.isdigit():
        try:
            return max(0.0, float(val))
        except Exception:
            return None
    try:
        dt = parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None

def google_cse_search(query: str, api_key: str, cx: str,
                      mode: str = TIME_MODE, days: int = TIME_DAYS) -> list:
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": MAX_RESULTS_PER_QUERY,
    }

    if mode == "calendar":
        ldn = ZoneInfo(TIME_ZONE)
        today_ldn = datetime.now(ldn).date()
        end = today_ldn - timedelta(days=1)               # exclude "today" for stability
        start = end - timedelta(days=days - 1)
        params["sort"] = f"date:r:{start:%Y%m%d}:{end:%Y%m%d}"
    else:
        params["dateRestrict"] = f"d{days}"

    delay = CSE_BASE_DELAY_SEC
    for attempt in range(1, CSE_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=CSE_TIMEOUT_SEC, headers=FETCH_HEADERS) as client:
                resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)

            if resp.status_code in (429, 503):
                ra = resp.headers.get("retry-after")
                wait = _parse_retry_after(ra)
                if wait is None:
                    wait = delay
                wait = min(wait, 15.0)
                logging.warning("CSE %s on attempt %d. Waiting %.2fs. URL=%s",
                                resp.status_code, attempt, wait, resp.request.url)
                time.sleep(wait)
                delay *= CSE_BACKOFF_FACTOR
                continue

            if resp.status_code == 403:
                try:
                    payload = resp.json()
                    reason = payload.get("error", {}).get("errors", [{}])[0].get("reason")
                except Exception:
                    reason = "forbidden"
                logging.error("CSE 403 (%s). Skipping query. URL=%s", reason, resp.request.url)
                return []

            resp.raise_for_status()
            return resp.json().get("items", []) or []

        except httpx.HTTPStatusError as e:
            logging.error("CSE HTTP error on attempt %d: %s", attempt, e)
            time.sleep(delay)
            delay = min(delay * CSE_BACKOFF_FACTOR, 15.0)
        except Exception as e:
            logging.error("CSE transport error on attempt %d: %s", attempt, e)
            time.sleep(delay)
            delay = min(delay * CSE_BACKOFF_FACTOR, 15.0)

    logging.error("CSE failed after %d attempts for query: %s", CSE_MAX_RETRIES, query)
    return []

# ───────────────────── Fetch & Extract ─────────────────────
UNSAFE_NETS = [
    ip_network("127.0.0.0/8"),
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("169.254.0.0/16"),  # link-local
    ip_network("::1/128"),
    ip_network("fc00::/7"),        # unique local
]

def _host_is_private_or_local(hostname: str) -> bool:
    host = (hostname or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        ip = ip_address(host)
        if ip.is_reserved or any(ip in n for n in UNSAFE_NETS):
            return True
    except ValueError:
        # Not a literal IP, leave to DNS; we don't resolve to check private nets here.
        return False
    return False

def _is_safe_public_http_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    if _host_is_private_or_local(p.hostname):
        return False
    return True

def _safe_get_with_redirects(client: httpx.Client, url: str, timeout: float, max_redirects: int = 5) -> httpx.Response:
    """
    Perform a GET without automatic redirects, validating every hop for SSRF.
    """
    if not _is_safe_public_http_url(url):
        raise ValueError("unsafe_url")

    current_url = url
    for _ in range(max_redirects + 1):
        r = client.get(current_url, timeout=timeout, follow_redirects=False, headers=FETCH_HEADERS)
        if r.is_redirect:
            loc = r.headers.get("location")
            if not loc:
                break
            next_url = httpx.URL(current_url).join(loc).human_repr()
            if not _is_safe_public_http_url(next_url):
                raise ValueError("unsafe_redirect")
            current_url = next_url
            continue
        return r

    return r  # either final or max redirects reached

def fetch_and_extract(url: str, client: httpx.Client,
                      timeout: float = FETCH_TIMEOUT_SEC) -> dict:
    parsed = urlparse(url)
    path_lower = (parsed.path or '').lower()

    # Skip obvious binaries by extension (works even with querystrings)
    if path_lower.endswith(DOC_EXTS):
        return {"url": url, "text": None, "title": None, "date": None, "error": "skipped_binary"}

    try:
        r = _safe_get_with_redirects(client, url, timeout=timeout)
        r.raise_for_status()
    except ValueError as ve:
        return {"url": url, "text": None, "title": None, "date": None, "error": str(ve)}
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"HTTP:{e}"}

    ctype = (r.headers.get("content-type") or "").lower()
    # If server returns a binary content-type, skip
    if any(bin_ct in ctype for bin_ct in ("application/pdf", "application/msword",
                                          "application/vnd.openxmlformats", "application/vnd.ms-excel",
                                          "application/vnd.ms-powerpoint")):
        return {"url": url, "text": None, "title": None, "date": None, "error": f"binary_content:{ctype}"}

    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"non_html:{ctype}"}

    extracted = trafilatura.extract(
        r.text,
        include_comments=False,
        include_tables=False,
        output_format="json",
        with_metadata=True,
        favor_precision=True
    )
    if not extracted:
        return {"url": url, "text": None, "title": None, "date": None, "error": "extract_failed"}

    try:
        data = json.loads(extracted)
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"parse_json:{e}"}

    return {"url": url, "text": data.get("text"), "title": data.get("title"),
            "date": data.get("date"), "error": None}

# ─────────────────── LLM Relevance Filter ───────────────────
def gemini_generate_with_backoff(model, prompt: str, config: dict, max_tries: int = 5):
    for attempt in range(1, max_tries + 1):
        try:
            return model.generate_content(prompt, generation_config=config)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_429_503 = (status in (429, 503)) or ("429" in str(e)) or ("503" in str(e))
            if is_429_503 and attempt < max_tries:
                wait = min((2 ** attempt) + random.random(), 15.0)
                logging.warning("%s from Gemini, retry %d/%d after %.1fs",
                                status or "rate-limit", attempt, max_tries, wait)
                time.sleep(wait)
                continue
            raise

def llm_relevance_filter(items: list, gemini_key: str,
                         model_name: str = GEMINI_MODEL
                        ) -> Tuple[List[dict], List[dict], List[dict]]:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)


    prompt_template = Template(
        "You are a strict filter for news about telecom operators (CSPs) "
        "using generative AI (GenAI/LLMs) for deployments, pilots, rollouts, or implementations.\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "partnering, piloting  or rolling out sattelite services\": \"<=20 words\"}\n\n"
        "Text to review (truncated):\n"
        "\"\"\"$article\"\"\""
    )

    relevant, irrelevant, errors = [], [], []
    logging.info("Sending %d items to Gemini for relevance", len(items))

    for it in items:
        text = (it.get("extracted_text") or "").strip()
        if not text:
            irrelevant.append({"url": it["link"], "reason": "no text"})
            continue

        prompt = prompt_template.substitute(article=text[:3000])
        try:
            resp = gemini_generate_with_backoff(
                model, prompt, {"temperature": 0, "max_output_tokens": 60}
            )
        except Exception as e:
            errors.append({"url": it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text", "").strip()
        data = _extract_top_level_json(raw)
        if isinstance(data, dict) and data.get("relevant") is True:
            it["relevance_reason"] = data.get("reason", "")
            relevant.append(it)
        else:
            reason = data.get("reason", "not relevant") if isinstance(data, dict) else "bad JSON"
            irrelevant.append({"url": it["link"], "reason": reason})

        time.sleep(0.3 + random.random() * 0.2)

    return relevant, irrelevant, errors

# ─────────────────── LLM Structured Extraction ───────────────────
def llm_extract_structured(items: list, gemini_key: str,
                           model_name: str = GEMINI_MODEL
                          ) -> Tuple[List[List[str]], List[dict]]:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    prompt_schema = Template(
        "Extract these fields from the article text:\n"
        "- company: primary telecom operator or vendor\n"
        "- technology: GenAI/LLM used\n"
        "- activity: deployment, pilot, rollout, etc.\n"
        "- summary: <=40 words\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "{\"company\":\"…\",\"technology\":\"…\",\"activity\":\"…\",\"summary\":\"…\"}\n\n"
        "Text:\n"
        "\"\"\"$article\"\"\""
    )

    rows, errors = [], []
    logging.info("Extracting structured data for %d items", len(items))

    for it in items:
        snippet = (it.get("extracted_text") or "")[:3500]
        prompt = prompt_schema.substitute(article=snippet)
        try:
            resp = gemini_generate_with_backoff(
                model, prompt, {"temperature": 0, "max_output_tokens": 120}
            )
        except Exception as e:
            errors.append({"url": it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text", "").strip()
        data = _extract_top_level_json(raw)
        if isinstance(data, dict) and all(k in data for k in ("company","technology","activity","summary")):
            rows.append([
                data["company"], data["technology"], data["activity"], data["summary"],
                it["link"], it.get("extracted_date") or today_iso()
            ])
        else:
            errors.append({"url": it["link"], "raw": raw})

        time.sleep(0.3 + random.random() * 0.2)

    return rows, errors

# ------------------ Google Sheets Setup (lazy) ------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1YRIXgBdft3PaJJrOLVjWudPTUXk8jfljNq7EkWtGtCY"
WORKSHEET_NAME = "NewsData"

def _get_gs_client():
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if sa_path:
            creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
        else:
            creds, _ = adc_default(scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        logging.exception("Failed to initialise Google Sheets client: %s", e)
        raise

def _open_sheet():
    client = _get_gs_client()
    return client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)

def _get_existing_urls(limit_rows: int = 500) -> set:
    """Fetch last ~limit_rows URLs to avoid duplicates across runs."""
    try:
        sheet = _open_sheet()
        # Column 5 is the URL column
        urls = sheet.col_values(5)
        last = urls[-limit_rows:] if len(urls) > limit_rows else urls
        return {u for u in last if u}
    except Exception as e:
        logging.warning("Could not fetch existing URLs: %s", e)
        return set()

def append_to_sheet(rows: list):
    """rows: list of [company, tech, activity, summary, url, date]"""
    sheet = _open_sheet()
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    logging.info("Appended %d rows to sheet '%s'", len(rows), WORKSHEET_NAME)

# ────────────────────────── Flask Routes ──────────────────────────
@app.get("/")
def health():
    return jsonify({"message": "Alive. POST /run to execute."})

@app.post("/run")
def run_pipeline():
    dry_run = request.args.get("dry_run", "false").lower() == "true"
    step = request.args.get("step", "all").lower()
    debug = request.args.get("debug", "false").lower() == "true"
    debug_payload = {}
    trace = {}  # url -> dict of details across steps

    cse_key = os.getenv("CSE_API_KEY", "MISSING")
    cse_cx = os.getenv("CSE_CX", "MISSING")
    gemini_key = os.getenv("GEMINI_API_KEY", "MISSING")
    if "MISSING" in (cse_key, cse_cx):
        return jsonify({"error": "CSE secrets missing"}), 500

    # 1) Build & Search
    queries = build_queries()
    all_results, seen = [], set()
    if step in ("all", "search", "fetch", "llm", "extract"):
        for q, company in random.sample(queries, k=len(queries)):  # randomize to spread throttle
            if len(seen) >= MAX_TOTAL_RESULTS:
                break
            items = google_cse_search(q, cse_key, cse_cx)
            for it in items:
                link = it.get("link")
                if link and link not in seen:
                    seen.add(link)
                    rec = {
                        "link": link,
                        "title": it.get("title", ""),
                        "snippet": it.get("snippet", ""),
                        "company": company
                    }
                    # Precompute score once
                    rec["_score"] = score_item(rec)
                    all_results.append(rec)
                    # Trace basics
                    trace.setdefault(link, {}).update({
                        "company": company,
                        "title": rec["title"],
                        "snippet": rec["snippet"],
                        "stage": "found",
                        "score": rec["_score"],
                    })
            time.sleep(CSE_BASE_DELAY_SEC + random.random() * 0.3)

    if debug:
        debug_payload["all_results"] = [
            {"company": it["company"], "title": it["title"], "link": it["link"]}
            for it in all_results
        ]

    # 1.5) Rank and pick the best to fetch (no duplicate scoring)
    ranked = sorted(
        all_results,
        key=lambda it: (it["_score"], min(len(it.get("title", "")), 120)),
        reverse=True
    )
    for idx, it in enumerate(ranked, start=1):
        url = it["link"]
        trace.setdefault(url, {}).update({
            "rank": idx
        })

    candidates = cap_by_domain(ranked, per_domain=2)[:MAX_FETCH]
    for it in candidates:
        it["_picked_for_fetch"] = True
        url = it["link"]
        trace.setdefault(url, {}).update({
            "picked_for_fetch": True,
            "stage": "candidate"
        })

    if debug:
        debug_payload["candidates"] = [it["link"] for it in candidates]

    # 2) Fetch
    fetched, fetch_errors = [], []
    if step in ("all", "fetch", "llm", "extract") and not dry_run:
        with httpx.Client(timeout=FETCH_TIMEOUT_SEC, headers=FETCH_HEADERS) as client:
            for it in candidates:
                res = fetch_and_extract(it["link"], client)
                if res["error"]:
                    fetch_errors.append(res)
                    trace.setdefault(res["url"], {}).update({
                        "stage": "fetch_error",
                        "fetch_error": res["error"]
                    })
                else:
                    it.update({
                        "extracted_text": res["text"],
                        "extracted_title": res["title"],
                        "extracted_date": res["date"],
                    })
                    fetched.append(it)
                    trace.setdefault(it["link"], {}).update({
                        "stage": "fetched",
                        "extracted_title": res["title"],
                        "extracted_date": res["date"]
                    })

    if debug:
        debug_payload["fetched_urls"] = [it["link"] for it in fetched]
        debug_payload["fetch_errors_detail"] = fetch_errors

    # 3) Relevance filter
    relevant, irrelevant, rel_errors = [], [], []
    if step in ("all", "llm", "extract") and not dry_run and fetched:
        if gemini_key == "MISSING":
            return jsonify({"error": "GEMINI_API_KEY missing"}), 500
        relevant, irrelevant, rel_errors = llm_relevance_filter(fetched, gemini_key)

    if debug:
        debug_payload["relevant_urls"] = [it["link"] for it in relevant]
        debug_payload["irrelevant_urls"] = [x.get("url") for x in irrelevant]

    # Trace LLM decisions
    for it in relevant:
        trace.setdefault(it["link"], {}).update({
            "stage": "llm_relevant",
            "llm_relevant": True,
            "llm_reason": it.get("relevance_reason", "")
        })
    for rec in irrelevant:
        url = rec.get("url")
        if url:
            trace.setdefault(url, {}).update({
                "stage": "llm_irrelevant",
                "llm_relevant": False,
                "llm_reason": rec.get("reason", "")
            })
    for err in rel_errors:
        url = err.get("url")
        if url:
            trace.setdefault(url, {}).update({
                "stage": "llm_error",
                "llm_error": err.get("error", "")
            })

    # 4) Structured extract
    structured, ext_errors = [], []
    if step in ("all", "extract") and not dry_run and relevant:
        structured, ext_errors = llm_extract_structured(relevant, gemini_key)

        extracted_urls = set(row[4] for row in structured)
        for url in extracted_urls:
            trace.setdefault(url, {}).update({"stage": "extracted"})
        for err in ext_errors:
            url = err.get("url")
            if url:
                trace.setdefault(url, {}).update({
                    "stage": "extract_error",
                    "extract_error": err.get("error") or "parse_failed"
                })

    if debug:
        debug_payload["structured_preview"] = structured

    # 5) Append to Google Sheet (dedupe across recent rows)
    written = 0
    if structured:
        try:
            existing = _get_existing_urls(limit_rows=500)
            to_write = [row for row in structured if row[4] not in existing]
            if to_write:
                logging.info("Writing %d new row(s) to Sheets. URLs: %s",
                             len(to_write), [row[4] for row in to_write])
                append_to_sheet(to_write)
                written = len(to_write)
            else:
                logging.info("All extracted URLs already present in the sheet.")
        except Exception as e:
            logging.exception("Failed to write to Google Sheet: %s", e)

    # Summary response
    resp = {
        "status": "ok",
        "date": today_iso(),
        "dry_run": dry_run,
        "step": step,
        "total_raw": len(all_results),
        "deduped": len(seen),
        "fetched": len(fetched),
        "fetch_errors": len(fetch_errors),
        "relevant": len(relevant),
        "irrelevant": len(irrelevant),
        "relevance_errors": len(rel_errors),
        "extract_rows": len(structured),
        "extract_errors": len(ext_errors),
        "written": written,
        "search_window": {"mode": TIME_MODE, "days": TIME_DAYS, "tz": TIME_ZONE},
    }
    if debug:
        resp["debug"] = debug_payload
        # Condensed per-URL audit trail
        resp["debug"]["trace"] = [
            {
                "url": url,
                **{k: v for k, v in info.items() if k in (
                    "company","title","snippet",
                    "score","rank","picked_for_fetch","stage",
                    "fetch_error","llm_relevant","llm_reason","llm_error",
                    "extracted_title","extracted_date","extract_error"
                )}
            }
            for url, info in trace.items()
        ]
        resp["secrets_tail"] = {
            "cse_key": tail(cse_key),
            "cse_cx": tail(cse_cx),
            "gemini": tail(gemini_key)
        }

    return jsonify(resp), 200

if __name__ == "__main__":
    # Keep Flask debug off in production; SSRF guard relies on our own checks.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
