"""
LeadNest V4 — Free direct web search (single-file Flask app)

Install:
    py -m pip install flask ddgs

Run:
    py leadnest_v4.py

Open:
    http://127.0.0.1:5000

No Docker, no SearXNG server, no API key.

Honest limits:
- DDGS uses upstream public search services; it is not an independent web index.
- Search providers can throttle, challenge, change behavior, or return limited results.
- "Fetch next batch" requests more results where the backend supports it.
- There is no guaranteed unlimited number of indexed results.
- Search operators such as site: and inurl: are interpreted by the upstream
  search backend; support and accuracy vary.
"""

from flask import Flask, request, render_template_string, Response, redirect, url_for
from ddgs import DDGS
import sqlite3
import csv
import io
import re
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

app = Flask(__name__)
DB_FILE = "leadnest_v4.db"
lock = threading.Lock()
state = {"busy": False, "message": "Ready", "last_error": ""}

def connect():
    conn = sqlite3.connect(DB_FILE, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS leads(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            snippet TEXT,
            query TEXT,
            source TEXT,
            created_at TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS query_state(
            query TEXT PRIMARY KEY,
            next_offset INTEGER DEFAULT 0,
            batches INTEGER DEFAULT 0,
            updated_at TEXT
        )""")

def valid_url(value):
    try:
        p = urlparse(value or "")
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def get_state(query):
    with connect() as conn:
        row = conn.execute("SELECT * FROM query_state WHERE query=?", (query,)).fetchone()
        return dict(row) if row else {"query": query, "next_offset": 0, "batches": 0}

def save_batch(query, results, source_name):
    added = 0
    with connect() as conn:
        for item in results:
            url = (item.get("href") or item.get("url") or "").strip()
            if not valid_url(url):
                continue
            title = (item.get("title") or "").strip()
            snippet = (item.get("body") or item.get("snippet") or "").strip()
            cur = conn.execute("""
                INSERT OR IGNORE INTO leads(url,title,snippet,query,source,created_at)
                VALUES(?,?,?,?,?,?)
            """, (
                url.split("#", 1)[0], title[:1000], snippet[:3000],
                query, source_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            if cur.rowcount:
                added += 1
        conn.commit()
    return added

def run_search(query, batch_size, region, timelimit):
    with lock:
        if state["busy"]:
            return False, "A search is already running. Wait for it to finish."
        state["busy"] = True
        state["message"] = "Searching..."
        state["last_error"] = ""

    try:
        old = get_state(query)
        offset = int(old.get("next_offset", 0))
        # DDGS text search supports a page parameter in current versions.
        # Page-based continuation is best-effort and may vary by backend.
        page = max(1, (offset // max(1, batch_size)) + 1)

        with DDGS() as ddgs:
            kwargs = {
                "query": query,
                "region": region or "in-en",
                "safesearch": "off",
                "timelimit": timelimit or None,
                "max_results": batch_size,
                "page": page,
            }
            # Some library/backend versions do not support page. Fall back
            # to a first-page request and explain that continuation is limited.
            try:
                results = list(ddgs.text(**kwargs))
            except TypeError:
                kwargs.pop("page", None)
                results = list(ddgs.text(**kwargs))

        added = save_batch(query, results, "DDGS")
        with connect() as conn:
            conn.execute("""
                INSERT INTO query_state(query,next_offset,batches,updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(query) DO UPDATE SET
                    next_offset=excluded.next_offset,
                    batches=excluded.batches,
                    updated_at=excluded.updated_at
            """, (
                query,
                offset + len(results),
                int(old.get("batches", 0)) + 1,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
            conn.commit()

        msg = f"Received {len(results)} results; saved {added} new unique leads."
        if not results:
            msg = "The search provider returned no results for this batch."
        with lock:
            state["message"] = msg
        return True, msg

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        with lock:
            state["last_error"] = msg
            state["message"] = "Search failed. The upstream provider may be limiting requests."
        return False, msg
    finally:
        with lock:
            state["busy"] = False

@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "")
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        leads = conn.execute("SELECT * FROM leads ORDER BY id DESC LIMIT 500").fetchall()
        saved_queries = conn.execute("SELECT * FROM query_state ORDER BY updated_at DESC LIMIT 30").fetchall()
    with lock:
        current = dict(state)
    return render_template_string(
        HTML, query=query, total=total, leads=leads,
        saved_queries=saved_queries, state=current
    )

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    region = request.form.get("region", "in-en").strip()
    timelimit = request.form.get("timelimit", "").strip()
    try:
        batch_size = int(request.form.get("batch_size", "30"))
        batch_size = max(1, min(batch_size, 100))
    except ValueError:
        batch_size = 30

    if not query:
        return redirect(url_for("index"))
    ok, message = run_search(query, batch_size, region, timelimit)
    return redirect(url_for("index", q=query, notice=message))

@app.route("/next", methods=["POST"])
def next_batch():
    query = request.form.get("query", "").strip()
    region = request.form.get("region", "in-en").strip()
    timelimit = request.form.get("timelimit", "").strip()
    try:
        batch_size = max(1, min(int(request.form.get("batch_size", "30")), 100))
    except ValueError:
        batch_size = 30
    if query:
        run_search(query, batch_size, region, timelimit)
    return redirect(url_for("index", q=query))

@app.route("/export.csv")
def export_csv():
    with connect() as conn:
        rows = conn.execute(
            "SELECT title,url,snippet,query,source,created_at FROM leads ORDER BY id DESC"
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Business / Page Title", "URL", "Search Snippet", "Search Query", "Source", "Saved At"])
    for row in rows:
        writer.writerow(list(row))
    filename = "leadnest_v4_leads_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
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
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeadNest V4 — Free Search</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#edf2ff;font:14px Segoe UI,Arial,sans-serif}
.wrap{max-width:1400px;margin:auto;padding:24px}.head{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}
.logo{font-size:26px;font-weight:800}.logo span{color:#a78bfa}.pill{border:1px solid #293956;border-radius:25px;padding:8px 12px;color:#9aabc7}
h1{font-size:29px;margin:28px 0 8px}.muted{color:#9aabc7}.panel{background:#121a2d;border:1px solid #293956;border-radius:15px;padding:20px;margin:18px 0}
label{display:block;margin:12px 0 7px;font-weight:650;color:#cbd5e1}
input,select{width:100%;padding:12px;border-radius:9px;border:1px solid #344766;background:#0b1427;color:white}
.grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px}
button,.btn{display:inline-block;padding:12px 17px;border:0;border-radius:9px;background:#8b5cf6;color:white;font-weight:700;cursor:pointer;text-decoration:none;margin:12px 8px 0 0}
.alt{background:#253653}.danger{background:#b93855}.stats{display:flex;gap:12px;flex-wrap:wrap}
.stat{padding:14px 18px;border:1px solid #293956;border-radius:12px;background:#0b1427}.stat b{font-size:23px;display:block}
.tablewrap{overflow:auto}table{width:100%;min-width:850px;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #293956;vertical-align:top}th{font-size:11px;color:#aebcda;text-transform:uppercase}td a{color:#bba9ff;overflow-wrap:anywhere;text-decoration:none}
.notice{padding:12px;background:#1d2b43;border-radius:9px;margin-top:12px}.warn{background:#2b2418;border-left:3px solid #f59e0b;padding:12px;margin-top:15px;color:#f5d9a6;line-height:1.6}
@media(max-width:800px){.wrap{padding:13px}.grid{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><div class="wrap">
<div class="head"><div class="logo">Lead<span>Nest</span> V4</div><div class="pill">FREE DIRECT SEARCH · NO API KEY · NO DOCKER</div></div>
<h1>Business lead search</h1><p class="muted">Search public web results directly from Python using DDGS.</p>
<section class="panel"><form method="post" action="/search">
<label>Search query</label><input name="query" required value="{{query}}" placeholder='site:instagram.com "hair accessories" "Thanjavur"'>
<div class="grid">
<div><label>Search region</label><select name="region"><option value="in-en">India — English</option><option value="wt-wt">Worldwide</option><option value="in-ta">India — Tamil</option><option value="us-en">United States</option></select></div>
<div><label>Results per batch</label><select name="batch_size"><option>10</option><option selected>30</option><option>50</option><option>100</option></select></div>
<div><label>Time filter</label><select name="timelimit"><option value="">Any time</option><option value="d">Past day</option><option value="w">Past week</option><option value="m">Past month</option><option value="y">Past year</option></select></div>
</div>
<button type="submit">⌕ Search web</button></form>
<form method="post" action="/next" style="display:inline">
<input type="hidden" name="query" value="{{query}}"><input type="hidden" name="region" value="in-en"><input type="hidden" name="batch_size" value="30">
<button class="alt" type="submit" {% if not query %}disabled{% endif %}>Fetch next batch</button></form>
<a class="btn alt" href="/export.csv">Export all CSV</a>
<div class="notice">{{state.message}}{% if state.last_error %}<br><b>Error:</b> {{state.last_error}}{% endif %}</div>
<div class="warn"><b>Important:</b> This is free to install and needs no API key. It is not unlimited: upstream search services can throttle or block requests, and the DDGS backend may not provide deep pagination. Google-style operators are passed to the search provider, which may ignore some of them.</div>
</section>
<div class="stats"><div class="stat"><b>{{total}}</b><span class="muted">Unique saved pages</span></div><div class="stat"><b>{{leads|length}}</b><span class="muted">Rows displayed</span></div></div>
<section class="panel"><h3>Saved results</h3><div class="tablewrap"><table><thead><tr><th>Title</th><th>URL</th><th>Snippet</th><th>Query</th><th>Saved</th></tr></thead><tbody>
{% for r in leads %}<tr><td>{{r.title}}</td><td><a href="{{r.url}}" target="_blank" rel="noopener">{{r.url}}</a></td><td>{{r.snippet}}</td><td>{{r.query}}</td><td>{{r.created_at}}</td></tr>{% else %}<tr><td colspan="5" class="muted">No results yet.</td></tr>{% endfor %}
</tbody></table></div><form method="post" action="/clear" onsubmit="return confirm('Delete all leads?')"><button class="danger">Clear saved leads</button></form></section>
<section class="panel"><h3>Search batches</h3><div class="tablewrap"><table><thead><tr><th>Query</th><th>Next offset</th><th>Batches</th><th>Updated</th></tr></thead><tbody>
{% for q in saved_queries %}<tr><td>{{q.query}}</td><td>{{q.next_offset}}</td><td>{{q.batches}}</td><td>{{q.updated_at}}</td></tr>{% else %}<tr><td colspan="4" class="muted">No searches yet.</td></tr>{% endfor %}
</tbody></table></div></section>
</div></body></html>
"""

if __name__ == "__main__":
    init_db()
    print("LeadNest V4: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
