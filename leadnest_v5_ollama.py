"""
LeadNest V5 — Ollama-connected business lead discovery
------------------------------------------------------
Install:
    py -m pip install flask ddgs

Requirements:
    1. Ollama running at http://127.0.0.1:11434
    2. Model available: ollama pull gemma3
    3. Run: py leadnest_v5_ollama.py
    4. Open: http://127.0.0.1:5000

Ollama generates focused search variations and can qualify result snippets.
DDGS supplies public web search results.

Important:
- No application-level total-lead limit: SQLite can keep saving unique results.
- Public search providers still impose their own indexing, pagination, rate and
  anti-bot limits. Therefore "unlimited internet search" cannot be guaranteed.
- Use only public business information and respect website terms and privacy laws.
"""

from flask import Flask, request, render_template_string, Response, redirect, url_for
from ddgs import DDGS
import sqlite3
import csv
import io
import json
import re
import threading
import urllib.request
import urllib.error
from datetime import datetime
from urllib.parse import urlparse

app = Flask(__name__)
DB_FILE = "leadnest_v5.db"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma3"
OLLAMA_TIMEOUT = 120

lock = threading.Lock()
state = {
    "busy": False,
    "message": "Ready",
    "last_error": "",
    "ollama": "Not checked"
}


def connect():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                snippet TEXT,
                query TEXT,
                source TEXT,
                ai_category TEXT DEFAULT '',
                ai_summary TEXT DEFAULT '',
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_state (
                query TEXT PRIMARY KEY,
                next_offset INTEGER DEFAULT 0,
                batches INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        # Upgrade databases created by earlier versions.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(leads)")}
        for name in ("ai_category", "ai_summary"):
            if name not in columns:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {name} TEXT DEFAULT ''")


def valid_url(value):
    try:
        p = urlparse(value or "")
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def ollama_chat(prompt, system=None, json_mode=False):
    """Call the local Ollama chat API. No cloud API key is used."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


def check_ollama():
    try:
        req = urllib.request.Request(OLLAMA_URL.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("models", [])]
        available = any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL + ":") for m in models)
        if available:
            return True, f"Connected · {OLLAMA_MODEL}"
        return False, f"Ollama running, but {OLLAMA_MODEL} is not installed. Run: ollama pull {OLLAMA_MODEL}"
    except Exception as exc:
        return False, f"Cannot connect to Ollama: {exc}"


def expand_search_query(query, use_ai=True):
    """Ask Gemma for related, distinct search phrases; fall back to original."""
    queries = [query]
    if not use_ai:
        return queries

    try:
        content = ollama_chat(
            "Create up to 5 distinct web-search query variations for finding "
            "public business pages and business contact/profile pages. "
            "Keep the same niche and location. Avoid duplicates. "
            "Return ONLY JSON in this format: "
            '{"queries":["query one","query two"]}\n'
            f"Original search: {query}",
            system="You generate precise search queries. Do not invent businesses.",
            json_mode=True
        )
        parsed = json.loads(content)
        for item in parsed.get("queries", []):
            item = re.sub(r"\s+", " ", str(item)).strip()
            if item and item.lower() not in {q.lower() for q in queries}:
                queries.append(item)
    except Exception:
        # Search still works if Ollama is offline or returns invalid JSON.
        pass
    return queries[:6]


def qualify_with_ollama(title, snippet, url):
    """Optional AI classification; search results are still saved if AI fails."""
    try:
        content = ollama_chat(
            "Classify this public web search result for a business-lead database. "
            "Return ONLY JSON: "
            '{"category":"business type or Unknown","summary":"one short factual sentence"}\n'
            f"Title: {title}\nSnippet: {snippet}\nURL: {url}",
            system="Use only facts visible in the supplied title, snippet and URL. "
                   "Do not guess phone numbers, emails, names or other missing facts.",
            json_mode=True
        )
        obj = json.loads(content)
        return str(obj.get("category", ""))[:120], str(obj.get("summary", ""))[:500]
    except Exception:
        return "", ""


def save_batch(query, results, source_name, ai_enrich=False):
    added = 0
    with connect() as conn:
        for item in results:
            url = (item.get("href") or item.get("url") or "").strip()
            if not valid_url(url):
                continue
            url = url.split("#", 1)[0]
            title = (item.get("title") or "").strip()
            snippet = (item.get("body") or item.get("snippet") or "").strip()

            category, summary = ("", "")
            if ai_enrich:
                category, summary = qualify_with_ollama(title, snippet, url)

            cur = conn.execute("""
                INSERT OR IGNORE INTO leads
                (url,title,snippet,query,source,ai_category,ai_summary,created_at)
                VALUES(?,?,?,?,?,?,?,?)
            """, (
                url, title[:1000], snippet[:3000], query, source_name,
                category, summary,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            if cur.rowcount:
                added += 1
        conn.commit()
    return added


def search_one_query(query, batch_size, region, timelimit, page):
    with DDGS() as ddgs:
        kwargs = {
            "query": query,
            "region": region or "in-en",
            "safesearch": "moderate",
            "timelimit": timelimit or None,
            "max_results": batch_size,
            "page": page
        }
        try:
            return list(ddgs.text(**kwargs))
        except TypeError:
            kwargs.pop("page", None)
            return list(ddgs.text(**kwargs))


def run_search(query, batch_size, region, timelimit, use_ai, ai_enrich):
    with lock:
        if state["busy"]:
            return False, "A search is already running. Wait for it to finish."
        state["busy"] = True
        state["message"] = "Searching public web sources..."
        state["last_error"] = ""

    try:
        old = get_query_state(query)
        offset = int(old.get("next_offset", 0))
        page = max(1, (offset // max(1, batch_size)) + 1)

        search_queries = expand_search_query(query, use_ai)
        combined = []
        errors = []

        for search_query in search_queries:
            try:
                results = search_one_query(
                    search_query, batch_size, region, timelimit, page
                )
                for item in results:
                    item["_leadnest_query"] = search_query
                combined.extend(results)
            except Exception as exc:
                errors.append(f"{search_query}: {type(exc).__name__}: {exc}")

        # Remove duplicate URLs inside this batch before saving.
        unique = {}
        for item in combined:
            url = (item.get("href") or item.get("url") or "").strip().split("#", 1)[0]
            if valid_url(url):
                unique.setdefault(url, item)

        added = 0
        for item in unique.values():
            source_query = item.get("_leadnest_query") or query
            added += save_batch(source_query, [item], "DDGS + Ollama" if use_ai else "DDGS", ai_enrich)

        with connect() as conn:
            conn.execute("""
                INSERT INTO query_state(query,next_offset,batches,updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(query) DO UPDATE SET
                    next_offset=excluded.next_offset,
                    batches=excluded.batches,
                    updated_at=excluded.updated_at
            """, (
                query, offset + len(unique), int(old.get("batches", 0)) + 1,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()

        msg = f"Checked {len(search_queries)} search phrases; received {len(unique)} unique results; saved {added} new leads."
        if not unique:
            msg = "No results returned. The public search provider may have no more results or may be rate-limiting."
        if errors:
            msg += " Provider notes: " + " | ".join(errors[:2])
        with lock:
            state["message"] = msg
        return True, msg

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        with lock:
            state["last_error"] = msg
            state["message"] = "Search failed. Check the search provider and Ollama connection."
        return False, msg
    finally:
        with lock:
            state["busy"] = False


def get_query_state(query):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM query_state WHERE query=?", (query,)
        ).fetchone()
        return dict(row) if row else {"query": query, "next_offset": 0, "batches": 0}


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "")
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        leads = conn.execute("SELECT * FROM leads ORDER BY id DESC LIMIT 500").fetchall()
        saved_queries = conn.execute(
            "SELECT * FROM query_state ORDER BY updated_at DESC LIMIT 30"
        ).fetchall()
    ok, ollama_status = check_ollama()
    with lock:
        state["ollama"] = ollama_status
        current = dict(state)
    return render_template_string(
        HTML, query=query, total=total, leads=leads,
        saved_queries=saved_queries, state=current, ollama_ok=ok
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    region = request.form.get("region", "in-en").strip()
    timelimit = request.form.get("timelimit", "").strip()
    use_ai = request.form.get("use_ai") == "on"
    ai_enrich = request.form.get("ai_enrich") == "on"
    try:
        batch_size = max(1, min(int(request.form.get("batch_size", "30")), 100))
    except ValueError:
        batch_size = 30
    if query:
        run_search(query, batch_size, region, timelimit, use_ai, ai_enrich)
    return redirect(url_for("index", q=query))


@app.route("/next", methods=["POST"])
def next_batch():
    query = request.form.get("query", "").strip()
    region = request.form.get("region", "in-en").strip()
    timelimit = request.form.get("timelimit", "").strip()
    use_ai = request.form.get("use_ai") == "on"
    ai_enrich = request.form.get("ai_enrich") == "on"
    try:
        batch_size = max(1, min(int(request.form.get("batch_size", "30")), 100))
    except ValueError:
        batch_size = 30
    if query:
        run_search(query, batch_size, region, timelimit, use_ai, ai_enrich)
    return redirect(url_for("index", q=query))


@app.route("/export.csv")
def export_csv():
    with connect() as conn:
        rows = conn.execute("""
            SELECT title,url,snippet,query,source,ai_category,ai_summary,created_at
            FROM leads ORDER BY id DESC
        """).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Business / Page Title", "URL", "Search Snippet", "Search Query",
        "Source", "AI Category", "AI Summary", "Saved At"
    ])
    for row in rows:
        writer.writerow(list(row))
    filename = "leadnest_leads_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.route("/clear", methods=["POST"])
def clear():
    with connect() as conn:
        conn.execute("DELETE FROM leads")
        conn.execute("DELETE FROM query_state")
        conn.commit()
    return redirect(url_for("index"))


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeadNest V5 — Ollama Lead Search</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0b1020;color:#edf2ff;font:14px Segoe UI,Arial,sans-serif}
.wrap{max-width:1450px;margin:auto;padding:24px}
.head{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}
.logo{font-size:26px;font-weight:800}.logo span{color:#a78bfa}
.pill{border:1px solid #293956;border-radius:25px;padding:8px 12px;color:#9aabc7}
h1{font-size:29px;margin:28px 0 8px}.muted{color:#9aabc7}
.panel{background:#121a2d;border:1px solid #293956;border-radius:15px;padding:20px;margin:18px 0}
label{display:block;margin:12px 0 7px;font-weight:650;color:#cbd5e1}
input,select{width:100%;padding:12px;border-radius:9px;border:1px solid #344766;background:#0b1427;color:white}
.grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px}
button,.btn{display:inline-block;padding:12px 17px;border:0;border-radius:9px;background:#8b5cf6;color:white;font-weight:700;cursor:pointer;text-decoration:none;margin:12px 8px 0 0}
.alt{background:#253653}.danger{background:#b93855}
.stats{display:flex;gap:12px;flex-wrap:wrap}
.stat{padding:14px 18px;border:1px solid #293956;border-radius:12px;background:#0b1427}
.stat b{font-size:23px;display:block}
.tablewrap{overflow:auto}table{width:100%;min-width:1050px;border-collapse:collapse}
th,td{padding:12px;text-align:left;border-bottom:1px solid #293956;vertical-align:top}
th{font-size:11px;color:#aebcda;text-transform:uppercase}
td a{color:#bba9ff;overflow-wrap:anywhere;text-decoration:none}
.notice{padding:12px;background:#1d2b43;border-radius:9px;margin-top:12px}
.status{padding:12px;border-radius:9px;margin-top:12px}
.good{background:#12372d;color:#a7f3d0}.bad{background:#3a2028;color:#fecdd3}
.warn{background:#2b2418;border-left:3px solid #f59e0b;padding:12px;margin-top:15px;color:#f5d9a6;line-height:1.6}
.checks{display:flex;gap:22px;flex-wrap:wrap;margin-top:12px}
.checks label{display:flex;align-items:center;gap:8px;margin:0;font-weight:500}
.checks input{width:auto}
@media(max-width:800px){.wrap{padding:13px}.grid{grid-template-columns:1fr}h1{font-size:23px}}
</style>
</head>
<body><div class="wrap">
<div class="head">
  <div class="logo">Lead<span>Nest</span> V5</div>
  <div class="pill">LOCAL OLLAMA · GEMMA3 · FREE SEARCH CONNECTOR</div>
</div>
<h1>Business lead discovery</h1>
<p class="muted">Gemma3 expands your search phrases and optionally classifies public search results. DDGS retrieves web results.</p>

<section class="panel">
  <div class="status {{'good' if ollama_ok else 'bad'}}"><b>Ollama:</b> {{state.ollama}}</div>
  <form method="post" action="/search">
    <label>Search query</label>
    <input name="query" required value="{{query}}" placeholder='Example: boutiques Instagram business Thanjavur'>
    <div class="grid">
      <div><label>Search region</label><select name="region">
        <option value="in-en">India — English</option>
        <option value="in-ta">India — Tamil</option>
        <option value="wt-wt">Worldwide</option>
        <option value="us-en">United States</option>
      </select></div>
      <div><label>Results per phrase</label><select name="batch_size">
        <option>10</option><option selected>30</option><option>50</option><option>100</option>
      </select></div>
      <div><label>Time filter</label><select name="timelimit">
        <option value="">Any time</option><option value="d">Past day</option>
        <option value="w">Past week</option><option value="m">Past month</option><option value="y">Past year</option>
      </select></div>
    </div>
    <div class="checks">
      <label><input type="checkbox" name="use_ai" checked> Use Gemma3 to expand search phrases</label>
      <label><input type="checkbox" name="ai_enrich"> Use Gemma3 to classify each result (slower)</label>
    </div>
    <button type="submit">⌕ Search web + AI</button>
  </form>

  <form method="post" action="/next" style="display:inline">
    <input type="hidden" name="query" value="{{query}}">
    <input type="hidden" name="region" value="in-en">
    <input type="hidden" name="batch_size" value="30">
    <input type="hidden" name="use_ai" value="on">
    <button class="alt" type="submit" {% if not query %}disabled{% endif %}>Fetch next batch</button>
  </form>
  <a class="btn alt" href="/export.csv">Export all CSV</a>
  <div class="notice">{{state.message}}{% if state.last_error %}<br><b>Error:</b> {{state.last_error}}{% endif %}</div>
  <div class="warn"><b>Search limit note:</b> Your database has no fixed total-lead cap. However, DDGS uses public search services, which can throttle requests, block automation, limit pagination or return only indexed pages. Gemma3 does not remove those provider limits. For more reliable large-scale discovery, connect a self-hosted SearXNG instance or another permitted search source.</div>
</section>

<div class="stats">
  <div class="stat"><b>{{total}}</b><span class="muted">Unique saved leads</span></div>
  <div class="stat"><b>{{leads|length}}</b><span class="muted">Rows displayed (latest 500)</span></div>
</div>

<section class="panel">
<h3>Saved results</h3>
<div class="tablewrap"><table>
<thead><tr><th>Title</th><th>URL</th><th>Snippet</th><th>Query</th><th>AI category</th><th>AI summary</th><th>Saved</th></tr></thead>
<tbody>
{% for r in leads %}
<tr>
<td>{{r.title}}</td><td><a href="{{r.url}}" target="_blank" rel="noopener">{{r.url}}</a></td>
<td>{{r.snippet}}</td><td>{{r.query}}</td><td>{{r.ai_category}}</td><td>{{r.ai_summary}}</td><td>{{r.created_at}}</td>
</tr>
{% else %}<tr><td colspan="7" class="muted">No results yet.</td></tr>{% endfor %}
</tbody></table></div>
<form method="post" action="/clear" onsubmit="return confirm('Delete all saved leads?')">
<button class="danger">Clear saved leads</button></form>
</section>

<section class="panel"><h3>Search history</h3>
<div class="tablewrap"><table><thead><tr><th>Query</th><th>Next offset</th><th>Batches</th><th>Updated</th></tr></thead>
<tbody>{% for q in saved_queries %}
<tr><td>{{q.query}}</td><td>{{q.next_offset}}</td><td>{{q.batches}}</td><td>{{q.updated_at}}</td></tr>
{% else %}<tr><td colspan="4" class="muted">No searches yet.</td></tr>{% endfor %}</tbody>
</table></div></section>
</div></body></html>
"""


if __name__ == "__main__":
    init_db()
    print("LeadNest V5 Ollama: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
