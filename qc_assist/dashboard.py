"""Operator dashboard: a local web page served from the device. No internet, no CDNs."""

from __future__ import annotations

import json
import time
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from .inspector import Inspector

LLM_TIMEOUT_S = 180


def create_app(inspector: Inspector) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    # "Add to Home screen" on a phone installs the dashboard as a full-screen app.
    @app.get("/manifest.webmanifest")
    def manifest():
        return jsonify(name="QC-Assist", short_name="QC-Assist", start_url="/", display="standalone",
                       background_color="#0f1418", theme_color="#0f1418",
                       icons=[{"src": f"/icon-{n}.png", "sizes": f"{n}x{n}", "type": "image/png"} for n in (192, 512)])

    @app.get("/icon-<int:size>.png")
    def icon(size):
        if size not in (192, 512):
            return Response(status=404)
        return Response(_icon_png(size), mimetype="image/png")

    @app.get("/stream")
    def stream():
        def gen():
            # Browsers render a multipart part only when the next boundary arrives, so the boundary
            # is sent right after each JPEG; otherwise the video lags one frame behind the stats.
            yield b"--frame\r\n"
            last = None
            while True:
                jpg = inspector.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n--frame\r\n"
                time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/frame")
    def frame():
        # The newest annotated frame plus its verdict in a header, so the page updates the image
        # and the text together (an MJPEG <img> rendered one frame behind the polled stats).
        f = inspector.wait_frame(int(request.args.get("after", 0)))
        if f is None:
            return Response(status=204)
        return Response(f[1], mimetype="image/jpeg",
                        headers={"X-Frame": json.dumps(f[2], ensure_ascii=True), "Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return jsonify(inspector.snapshot())

    @app.get("/api/summary")
    def summary():
        hours = float(request.args.get("hours", 8))
        return jsonify(inspector.log.summary(inspector.product, time.time() - hours * 3600))

    @app.post("/api/lang")
    def lang():
        inspector.lang = "hi" if request.json.get("lang") == "hi" else "en"
        return jsonify(lang=inspector.lang)

    @app.post("/api/report")
    def report():
        stats = inspector.log.summary(inspector.product, time.time() - 8 * 3600)
        t0 = time.perf_counter()
        res = inspector.advisor.submit("report", stats, lang=inspector.lang).result(timeout=LLM_TIMEOUT_S)
        return jsonify(text=res.text, source=res.source, note=res.note, seconds=round(time.perf_counter() - t0, 1),
                       stats=inspector.advisor.last_stats if res.source == "llama-executorch" else None)

    @app.post("/api/ask")
    def ask():
        question = (request.json.get("question") or "").strip()[:500]
        if not question:
            return jsonify(error="empty question"), 400
        stats = inspector.log.summary(inspector.product, time.time() - 8 * 3600)
        stats["recent_defects"] = [
            {k: r[k] for k in ("id", "severity", "region", "defect_type", "area_pct")}
            for r in inspector.log.recent(inspector.product, 50) if r["severity"] != "ok"
        ][:10]
        t0 = time.perf_counter()
        res = inspector.advisor.submit("answer", question, stats, lang=inspector.lang).result(timeout=LLM_TIMEOUT_S)
        return jsonify(text=res.text, source=res.source, note=res.note, seconds=round(time.perf_counter() - t0, 1),
                       stats=inspector.advisor.last_stats if res.source == "llama-executorch" else None)

    @app.post("/api/teach")
    def teach():
        body = request.json or {}
        try:
            if body.get("folder"):
                inspector.teach_from_folder(body["folder"])
            else:
                inspector.teach_from_camera(int(body.get("n", 20)))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True)

    return app


@lru_cache(maxsize=2)
def _icon_png(size: int) -> bytes:
    """App icon drawn at startup: dark tile, green check ring, "QC"."""
    img = np.full((size, size, 3), (24, 20, 15), np.uint8)  # BGR #0f1418
    c, r = size // 2, int(size * 0.36)
    cv2.circle(img, (c, c), r, (67, 160, 46), max(2, size // 24), cv2.LINE_AA)  # #2ea043
    scale, thick = size / 110, max(2, size // 30)
    (w, h), _ = cv2.getTextSize("QC", cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(img, "QC", (c - w // 2, c + h // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (243, 237, 230), thick, cv2.LINE_AA)
    return cv2.imencode(".png", img)[1].tobytes()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QC-Assist</title><link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#0f1418"><link rel="icon" href="/icon-192.png">
<style>
:root{--bg:#0f1418;--panel:#182027;--line:#26313a;--text:#e6edf3;--muted:#8b98a5;
--ok:#2ea043;--minor:#d29922;--major:#f0883e;--critical:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
header{display:flex;align-items:center;flex-wrap:wrap;gap:8px 16px;padding:12px 16px;border-bottom:1px solid var(--line)}
.pill{white-space:nowrap}
header h1{font-size:18px;margin:0}header .sub{color:var(--muted)}header .sp{flex:1}
.pill{padding:2px 10px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
.card{min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
#video{width:100%;border-radius:8px;background:#000;aspect-ratio:4/3;object-fit:contain}
.verdict{display:flex;gap:18px;align-items:baseline;margin-top:10px;flex-wrap:wrap}
.sev{font-size:28px;font-weight:700}.metric{color:var(--muted)}.metric b{color:var(--text)}
.sev-ok{color:var(--ok)}.sev-minor{color:var(--minor)}.sev-major{color:var(--major)}.sev-critical{color:var(--critical)}
.col{display:flex;flex-direction:column;gap:16px}
.adv{border-left:3px solid var(--line);padding:6px 10px;margin-bottom:8px;background:#11181e;border-radius:4px}
.adv .meta{color:var(--muted);font-size:12px}.adv p{margin:4px 0 0;white-space:pre-wrap}
.src{font-size:11px;padding:1px 6px;border-radius:4px;background:#26313a}
.src.llama-executorch{background:#1f6feb33;color:var(--accent)}.src.template{background:#d2992233;color:var(--minor)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center}
.grid div{background:#11181e;border-radius:6px;padding:8px}.grid b{display:block;font-size:20px}
button,input{font:inherit;border-radius:6px;border:1px solid var(--line);background:#11181e;color:var(--text);padding:7px 12px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button.primary{background:#1f6feb;border-color:#1f6feb}
.row{display:flex;gap:8px;flex-wrap:wrap}.row input{flex:1;min-width:0}
#answer{white-space:pre-wrap;margin-top:10px;color:var(--text)}.muted{color:var(--muted)}
#advs{max-height:340px;overflow:auto}
</style></head><body>
<header><h1>QC-Assist</h1><span class="sub" id="product"></span><span class="sp"></span>
<span class="pill" id="llm"></span><span class="pill">offline</span>
<button id="lang">EN / हिंदी</button></header>
<main>
<section class="col">
 <div class="card"><img id="video" alt="live inspection">
  <div class="verdict"><span class="sev" id="sev">-</span><span class="metric">type <b id="dtype">-</b></span>
   <span class="metric">score <b id="score">-</b></span>
   <span class="metric">vision <b id="ms">-</b> ms</span>
   <span class="metric">fps <b id="fps">-</b></span>
   <span class="metric" id="status"></span></div></div>
 <div class="card"><h2>Teach a new board</h2>
  <div class="row"><button class="primary" id="teach">Capture 20 good boards</button>
  <input id="folder" placeholder="...or folder of good images on this device"><button id="teachf">Train from folder</button></div>
  <div class="muted" id="teachst" style="margin-top:8px"></div></div>
</section>
<section class="col">
 <div class="card"><h2>Shift statistics (last 8 h)</h2>
  <div class="grid"><div><b id="n">0</b>inspected</div><div><b id="rate">0%</b>defect rate</div>
  <div><b id="crit" class="sev-critical">0</b>critical</div><div><b id="maj" class="sev-major">0</b>major</div></div></div>
 <div class="card"><h2>Llama advisory</h2><div id="advs" class="muted">No defects yet.</div></div>
 <div class="card"><h2>Ask the line</h2>
  <div class="row"><input id="q" placeholder="e.g. Where are most defects this shift?"><button id="ask">Ask</button>
  <button id="report">Shift report</button></div><div id="answer" class="muted"></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json()}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function tick(){
 try{const s=await (await fetch('/api/state')).json();
  $('product').textContent=s.product;$('llm').textContent=s.llama?'Llama 3.2 · ExecuTorch':'Llama not configured (fallback)';
  $('lang').textContent=s.lang==='hi'?'हिंदी':'English';
  $('teachst').textContent=s.teach||'';
  if(s.advisories.length)$('advs').innerHTML=s.advisories.map(a=>`<div class="adv"><div class="meta">#${a.id} · ${a.time} · <span class="sev-${a.severity}">${a.severity}</span> · ${esc(a.defect_type||'unclassified')} · ${esc(a.region)} <span class="src ${a.source}" title="${esc(a.note||'')}">${a.source}</span></div><p>${esc(a.text)}</p></div>`).join('');
 }catch(e){}
}
function verdict(s){const sev=s.severity||'-';$('sev').textContent=sev.toUpperCase();$('sev').className='sev sev-'+sev;
 $('dtype').textContent=s.severity&&s.severity!=='ok'?(s.defect_type||'unclassified'):'-';$('score').textContent=s.score??'-';
 $('ms').textContent=s.vision_ms??'-';$('fps').textContent=s.fps??'-';$('status').textContent=s.status||''}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function frames(){let id=0;const img=$('video');
 for(;;){try{const r=await fetch('/api/frame?after='+id,{cache:'no-store'});
  if(r.status!==200){await pause(200);continue}
  const st=JSON.parse(r.headers.get('X-Frame')),url=URL.createObjectURL(await r.blob()),old=img.src;
  await new Promise(done=>{img.onload=img.onerror=done;img.src=url});
  if(old.startsWith('blob:'))URL.revokeObjectURL(old);id=st.id;verdict(st)}catch(e){await pause(1000)}}}
async function stats(){try{const s=await (await fetch('/api/summary')).json();
 $('n').textContent=s.inspected;$('rate').textContent=s.defect_rate_pct+'%';$('crit').textContent=s.by_severity.critical;$('maj').textContent=s.by_severity.major}catch(e){}}
async function llm(url,body){$('answer').textContent='Llama is thinking...';
 const r=await post(url,body);const st=r.stats?` · ${r.stats.decode_tok_s} tok/s decode, ${r.stats.prefill_tok_s} tok/s prefill`:'';const why=r.note?` · Llama output rejected: ${r.note}`:'';$('answer').textContent=r.error||`${r.text}\\n\\n(${r.source}, ${r.seconds}s${st}${why})`}
$('ask').onclick=()=>{const q=$('q').value.trim();if(q)llm('/api/ask',{question:q})};
$('q').onkeydown=e=>{if(e.key==='Enter')$('ask').click()};
$('report').onclick=()=>llm('/api/report');
$('lang').onclick=async()=>{const s=await (await fetch('/api/state')).json();await post('/api/lang',{lang:s.lang==='hi'?'en':'hi'});tick()};
$('teach').onclick=()=>post('/api/teach',{n:20});
$('teachf').onclick=async()=>{const r=await post('/api/teach',{folder:$('folder').value});if(r.error)$('teachst').textContent=r.error};
setInterval(tick,500);setInterval(stats,3000);tick();stats();frames();
</script></body></html>"""
