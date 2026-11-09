#!/usr/bin/env python3
"""
Status page for the local ComfyUI, Comfy Cloud and the RunPod animate pod (stdlib only):  python3 dashboard.py [--port 8190] [--comfy http://127.0.0.1:8188]
Open http://127.0.0.1:8190 (#cloud / #runpod for those views). Local: RAM/VRAM, service, running job with step progress, queue,
history, interrupt/clear. Cloud (read-only): running/queued jobs and history. RunPod (read-only): the pod (up/booting/down, GPU,
$/hr, the balance, the last cold start) and, while it is up, its ComfyUI like the local view. Desktop notification when a job
finishes or fails. JSON for the KDE widgets: /api/state?target=local|cloud|runpod.

Step progress: ComfyUI sends progress events only to the client id that submitted the job, so this listens on a websocket
as targets.local.CLIENT_ID / comfy_api.CLIENT_ID (what `--local` and cloud `generate` submit under). Jobs submitted elsewhere
show as running/queued without step detail. Cloud jobs have no labels, so they are named by matching their prompt text
to a stage in pipelines/*/project.json.
"""
import argparse
import base64
import json
import os
import re
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config
sys.path.insert(0, str(ROOT / "pipeline"))
import comfy_api
import runpod_pod
from targets.local import CLIENT_ID as LOCAL_CLIENT_ID

SERVICE = config.COMFYUI_SERVICE


class Target:
    def __init__(self, name, base, client_id, headers=None, query=""):
        self.name, self.base, self.client_id, self.headers, self.query = name, base.rstrip("/"), client_id, headers or {}, query
        self.progress = {}   # prompt_id -> {"node", "value", "max"}
        self.started = {}    # prompt_id -> time.time() first seen running (local)
        self.detail = {}     # cloud job id -> prompt graph (cached)
        self.lock = threading.Lock()

    def call(self, path, data=None, timeout=5):
        req = urllib.request.Request(self.base + path, data=None if data is None else json.dumps(data).encode(),  # RunPod's proxy 403s urllib's agent
                                     headers={"Content-Type": "application/json", "User-Agent": "comfy-dashboard", **self.headers})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else {}


LOCAL = Target("local", config.LOCAL_COMFY_URL, LOCAL_CLIENT_ID)
CLOUD = Target("cloud", comfy_api.CLOUD_BASE, comfy_api.CLIENT_ID)  # headers/query filled in main() once the key is read
RUNPOD = Target("runpod", "", LOCAL_CLIENT_ID)  # the pod's ComfyUI (animate --target runpod submits as the local client id); base set while it is up


# --- minimal websocket client (text frames only) ---------------------------------------------------------------
def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _read_frame(sock):
    """One data frame as (fin, opcode, payload); a ping is answered with a masked pong and None returned."""
    b1, b2 = _recv_exact(sock, 2)
    op, n = b1 & 0x0F, b2 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    payload = _recv_exact(sock, n)
    if op == 9:  # ping -> pong (client frames must be masked)
        mask = os.urandom(4)
        sock.sendall(bytes([0x8A, 0x80 | len(payload)]) + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))
        return None
    return b1 & 0x80, op, payload


def _ws_connect(base, path):
    """Open a websocket to <base><path>: TCP (TLS for https), the upgrade request, and the 101 check. Returns the socket."""
    u = urlparse(base)
    tls = u.scheme == "https"
    sock = socket.create_connection((u.hostname, u.port or (443 if tls else 80)), timeout=10)
    if tls:
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        sock = ctx.wrap_socket(sock, server_hostname=u.hostname)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {u.netloc}\r\nUser-Agent: comfy-dashboard\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("closed during handshake")
        head += chunk
    if b" 101 " not in head.split(b"\r\n")[0]:
        raise ConnectionError(head.split(b"\r\n")[0])
    sock.settimeout(None)
    return sock


def ws_messages(base, path):
    """Yield decoded JSON messages from the websocket at <base><path> (ws/wss chosen by the base scheme)."""
    sock = _ws_connect(base, path)
    frag = b""
    while True:
        frame = _read_frame(sock)
        if frame is None:
            continue
        fin, op, payload = frame
        if op == 8:
            raise ConnectionError("closed by server")
        if op in (1, 0):
            frag += payload
            if fin:
                text, frag = frag, b""
                try:
                    yield json.loads(text)
                except ValueError:
                    pass


def _record_progress(t, msg) -> None:
    """One websocket message's progress/executing data into the target's progress map."""
    d = msg.get("data", {})
    with t.lock:
        if msg.get("type") == "progress" and "prompt_id" in d:
            t.progress[d["prompt_id"]] = {"node": d.get("node"), "value": d["value"], "max": d["max"]}
        elif msg.get("type") == "executing" and d.get("prompt_id"):
            if d.get("node") is None:
                t.progress.pop(d["prompt_id"], None)
            else:
                t.progress.setdefault(d["prompt_id"], {})["node"] = d["node"]


def ws_loop(t):
    while True:
        try:
            for msg in ws_messages(t.base, f"/ws?clientId={t.client_id}{t.query}"):
                _record_progress(t, msg)
        except Exception:
            time.sleep(3)


# --- labels ----------------------------------------------------------------------------------------------------
_index = {"t": 0, "stages": []}


def stage_index():
    """[(krea_scene prefix, "project stage N")] from pipelines/*/project.json, rescanned every 15 s."""
    if time.time() - _index["t"] > 15:
        rows = []
        for pj in config.PIPELINES_DIR.glob("*/project.json"):
            try:
                proj = json.loads(pj.read_text())
            except ValueError:
                continue
            name = re.sub(r"^\d{8}_", "", pj.parent.name)
            for i, s in enumerate(proj.get("stages", []), 1):
                ks = (s.get("krea_scene") or "").strip()
                if ks:
                    rows.append((ks[:120], f"{name} stage {s.get('id', i)}"))
        _index.update(t=time.time(), stages=rows)
    return _index["stages"]


def graph_label(graph, jid):
    """Name a cloud job from its prompt graph: which project stage, and whether it's a preview or a full run."""
    text = ((graph.get("30:6") or {}).get("inputs") or {}).get("text") or ""
    if any(k.startswith(("433:", "434:")) for k in graph):
        kind = "full"
    elif "30:963" in graph:
        kind = "krea2"
    else:
        kind = "preview"
    for prefix, label in stage_index():
        if text.startswith(prefix):
            return f"{label} ({kind})"
    return f"job {jid[:8]}" + (f" ({kind})" if text else "")


def node_title(graph, node_id):
    n = (graph or {}).get(str(node_id)) or {}
    return (n.get("_meta") or {}).get("title") or n.get("class_type") or (str(node_id) if node_id else "")


# --- state -----------------------------------------------------------------------------------------------------
def service_state():
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", SERVICE], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def local_label(item):
    extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
    return extra.get("label") or f"job {item[1][:8]}"


def local_state():
    return comfy_state(LOCAL, {"target": "local", "time": time.time(), "service": service_state(), "online": False})


def comfy_state(t, out):
    """A plain ComfyUI's machine, running job (with step progress), queue and history, into `out` (the local ComfyUI or the pod)."""
    try:
        stats, queue, hist = t.call("/system_stats"), t.call("/queue"), t.call("/history?max_items=10")
    except Exception as e:
        return {**out, "error": str(getattr(e, "reason", e))}
    sysd = stats["system"]
    out.update(online=True, version=sysd.get("comfyui_version"), ram_total=sysd["ram_total"], ram_free=sysd["ram_free"],
               devices=[{"name": d["name"], "vram_total": d["vram_total"], "vram_free": d["vram_free"]} for d in stats["devices"]])
    now = time.time()
    running = []
    with t.lock:
        for item in queue["queue_running"]:
            pid = item[1]
            t.started.setdefault(pid, now)
            p = t.progress.get(pid, {})
            running.append({"id": pid, "label": local_label(item), "elapsed": now - t.started[pid], "node": node_title(item[2], p.get("node")),
                            "value": p.get("value"), "max": p.get("max"), "stepped": pid in t.progress})
        live = {i[1] for i in queue["queue_running"]}
        for pid in [k for k in t.started if k not in live]:
            t.started.pop(pid)
    out["running"] = running
    out["pending"] = [{"id": i[1], "label": local_label(i)} for i in queue["queue_pending"]]
    rows = []
    for pid, h in hist.items():
        msgs = {m[0]: m[1].get("timestamp") for m in h["status"].get("messages", []) if isinstance(m[1], dict)}
        t0, t1 = msgs.get("execution_start"), msgs.get("execution_success") or msgs.get("execution_error")
        rows.append({"n": h["prompt"][0], "label": local_label(h["prompt"]), "status": h["status"].get("status_str"),
                     "duration": (t1 - t0) / 1000 if t0 and t1 else None})
    out["history"] = sorted(rows, key=lambda r: -r["n"])
    return out


def cloud_label(t, jid):
    if jid not in t.detail:
        try:
            t.detail[jid] = t.call(f"/api/jobs/{jid}", timeout=10).get("workflow", {}).get("prompt") or {}
        except Exception:
            return f"job {jid[:8]}", {}
    g = t.detail[jid]
    return graph_label(g, jid), g


_pod_cache = {"t": 0, "v": None, "bt": 0, "b": None}
_pod_lock = threading.Lock()


def runpod_state():
    """The pod (runpod_pod.status, cached 10 s) and the balance (60 s) so polling widgets don't hammer the RunPod API; while the pod's
    ComfyUI answers, its queue and progress like the local view."""
    with _pod_lock:
        now = time.time()
        if now - _pod_cache["t"] > 10:
            try:
                _pod_cache["v"] = runpod_pod.status()
            except Exception as e:  # noqa: BLE001
                _pod_cache["v"] = {"up": False, "error": str(e)}
            _pod_cache["t"] = now
        if now - _pod_cache["bt"] > 60:
            try:
                _pod_cache["b"] = runpod_pod.balance()
            except Exception:  # noqa: BLE001
                pass
            _pod_cache["bt"] = now
        pod, bal = _pod_cache["v"], _pod_cache["b"]
    out = {"target": "runpod", "time": time.time(), "online": False, "pod": pod, "balance": bal}
    if not pod.get("ready"):
        RUNPOD.base = ""
        return {**out, "error": pod.get("error") or ("pod booting" if pod.get("id") else "pod down")}
    RUNPOD.base = pod["url"]
    return comfy_state(RUNPOD, out)


_cloud_cache = {"t": 0, "v": None}
_cloud_lock = threading.Lock()


def cloud_state():
    """Cached for 2 s so several open pages/widgets don't each hit the cloud API."""
    with _cloud_lock:
        if time.time() - _cloud_cache["t"] > 2:
            _cloud_cache.update(v=_cloud_state(), t=time.time())
        return _cloud_cache["v"]


def _cloud_state():
    t = CLOUD
    out = {"target": "cloud", "time": time.time(), "online": False}
    try:
        jobs = t.call("/api/jobs?limit=12")["jobs"]
        live = t.call("/api/jobs?status=in_progress,pending&limit=25")["jobs"]
        version = t.call("/api/system_stats").get("system", {}).get("cloud_version")
    except Exception as e:
        return {**out, "error": str(getattr(e, "reason", e))}
    out.update(online=True, version=version)
    now = time.time()
    running, pending = [], []
    for j in sorted(live, key=lambda j: j["create_time"]):
        label, graph = cloud_label(t, j["id"])
        if j["status"] == "in_progress":
            with t.lock:
                p = t.progress.get(j["id"], {})
            start = (j.get("execution_start_time") or j["create_time"]) / 1000
            running.append({"id": j["id"], "label": label, "elapsed": now - start, "node": node_title(graph, p.get("node")),
                            "value": p.get("value"), "max": p.get("max"), "stepped": j["id"] in t.progress})
        else:
            pending.append({"id": j["id"], "label": label})
    out["running"], out["pending"] = running, pending
    rows = []
    for j in jobs:
        if j["status"] in ("in_progress", "pending"):
            continue
        t0, t1 = j.get("execution_start_time"), j.get("execution_end_time")
        rows.append({"n": j["create_time"], "label": cloud_label(t, j["id"])[0], "status": "success" if j["status"] == "completed" else j["status"],
                     "duration": (t1 - t0) / 1000 if t0 and t1 else None})
    out["history"] = rows[:10]
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ComfyUI Status</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--text:#1c2024;--mute:#646b75;--line:#e1e4e8;--ok:#1a7f4b;--bad:#c62828;--warn:#b26a00;--bar:#3b6fd4;--track:#e6e9ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14161a;--card:#1d2026;--text:#e6e8eb;--mute:#8d95a0;--line:#2b2f37;--ok:#4cc38a;--bad:#ff6b6b;--warn:#f2b04a;--bar:#6c9cff;--track:#2a2e36}}
:root[data-theme="dark"]{--bg:#14161a;--card:#1d2026;--text:#e6e8eb;--mute:#8d95a0;--line:#2b2f37;--ok:#4cc38a;--bad:#ff6b6b;--warn:#f2b04a;--bar:#6c9cff;--track:#2a2e36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:18px;margin:0;flex:1}
.pill{padding:3px 10px;border-radius:99px;font-size:13px;font-weight:600;border:1px solid currentColor}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.mute{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:0 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.bar{height:10px;border-radius:5px;background:var(--track);overflow:hidden}.bar>i{display:block;height:100%;background:var(--bar)}
.row{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-top:1px solid var(--line)}.row:first-child{border-top:0}
.big{font-size:17px;font-weight:600}
button{font:inherit;padding:5px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--text);cursor:pointer}
button:hover{border-color:var(--mute)}button.danger{color:var(--bad)}button.on{border-color:var(--bar);color:var(--bar);font-weight:600}
.num{font-variant-numeric:tabular-nums}
</style></head><body><main>
<header><h1>ComfyUI status</h1><span id="conn" class="pill mute">…</span>
<span><button id="t-local">Local</button> <button id="t-cloud">Cloud</button> <button id="t-runpod">RunPod</button></span>
<button id="notif">Enable notifications</button><button id="stop" class="danger">Interrupt</button><button id="clear" class="danger">Clear queue</button></header>
<div id="err"></div>
<div class="grid" id="health"></div>
<div class="card" id="run"></div>
<div class="card"><h2>Pending</h2><div id="pend"></div></div>
<div class="card"><h2>Recent</h2><div id="hist"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
let target=["#cloud","#runpod"].includes(location.hash)?location.hash.slice(1):"local";
const gb=b=>(b/2**30).toFixed(1)+" GB", dur=s=>s==null?"–":s>=60?Math.floor(s/60)+"m "+Math.round(s%60)+"s":s.toFixed(0)+"s";
const meter=(label,used,total)=>{const p=total?used/total*100:0;return `<div class="card"><h2>${label}</h2><div class="big num">${gb(used)} <span class="mute">/ ${gb(total)}</span></div><div class="bar"><i style="width:${p}%;${p>90?'background:var(--bad)':''}"></i></div></div>`};
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let last=null;
const name=()=>({cloud:"Comfy Cloud",runpod:"RunPod pod"})[target]||"ComfyUI";
// the RunPod view's pod card: state, GPU, $/hr, uptime, balance, the last cold start
const podCard=s=>{const p=s.pod||{},b=s.balance,c=p.last_start;return `<div class="card"><h2>RunPod pod</h2><div class="big">${p.id?esc(p.id)+" · "+(p.ready?'<span class="ok">up</span>':'<span class="warn">booting</span>'):'<span class="mute">down</span>'}</div>`+
  `<div class="mute">${esc(p.gpu||"")}${p.cost_per_hr!=null?" · $"+p.cost_per_hr.toFixed(2)+"/hr":""}${p.uptime!=null?" · up "+dur(p.uptime):""}${p.session_cost!=null?" · this session <b>$"+p.session_cost.toFixed(2)+"</b>":""}</div>`+
  (!p.id&&p.last_run?`<div class="mute">last run <b>$${p.last_run.cost.toFixed(2)}</b> · ${Math.round(p.last_run.minutes)} min on ${esc(p.last_run.gpu)}</div>`:"")+
  (b?`<div>balance <b class="num">$${b.balance.toFixed(2)}</b> <span class="mute">· spending $${b.spend_per_hr.toFixed(2)}/hr</span></div>`:"")+
  (c?`<div class="mute">last cold start: ready ${dur(c.ready_s)}${c.first_clip_s!=null?" · first clip "+dur(c.first_clip_s):""}${c.warm_clip_s!=null?" · next "+dur(c.warm_clip_s):""}</div>`:"")+`</div>`};
function setTarget(t){target=t;location.hash=t==="local"?"":t;last=null;
  ["local","cloud","runpod"].forEach(k=>$("t-"+k).className=t===k?"on":"");
  $("stop").style.display=$("clear").style.display=t==="local"?"":"none";tick()}
function notify(t,b){try{if(Notification.permission==="granted")new Notification(t,{body:b})}catch(e){}}
function render(s){
  const conn=$("conn");
  if(!s.online){conn.textContent=s.target==="runpod"?esc(s.error||"down"):"offline";conn.className=s.target==="runpod"?"pill mute":"pill bad";
    $("err").innerHTML=s.target==="runpod"?"":`<div class="card bad">${name()} is not responding (${esc(s.error||"")}).${s.service?" Service: <b>"+esc(s.service)+"</b>":""}</div>`;
    ["health","run","pend","hist"].forEach(i=>$(i).innerHTML="");if(s.target==="runpod")$("health").innerHTML=podCard(s);document.title="✖ "+name()+" offline";
    if(last&&last.online)notify(name()+" went offline",s.service?"Service: "+s.service:"");last=s;return}
  $("err").innerHTML="";conn.textContent="online";conn.className="pill ok";
  if(s.target==="cloud"){$("health").innerHTML=`<div class="card"><h2>Server</h2><div class="big">Comfy Cloud</div><div class="mute">${esc(s.version||"")}</div></div>`}
  else if(s.target==="runpod"){const d=s.devices[0]||{name:"GPU",vram_total:0,vram_free:0};
    $("health").innerHTML=podCard(s)+meter("VRAM",d.vram_total-d.vram_free,d.vram_total)+meter("RAM",s.ram_total-s.ram_free,s.ram_total)}
  else{const d=s.devices[0]||{name:"GPU",vram_total:0,vram_free:0};
    $("health").innerHTML=meter("VRAM",d.vram_total-d.vram_free,d.vram_total)+meter("RAM",s.ram_total-s.ram_free,s.ram_total)+
    `<div class="card"><h2>Server</h2><div class="big">v${esc(s.version||"?")}</div><div class="mute">${esc(d.name)}</div><div>service: <span class="${s.service==="active"?"ok":"warn"}">${esc(s.service)}</span>${s.service!=="active"?' <span class="mute">(running outside systemd)</span>':""}</div></div>`}
  if(s.running.length){$("run").innerHTML="<h2>Running</h2>"+s.running.map(r=>{
    const p=r.max?r.value/r.max*100:0;
    return `<div class="big">${esc(r.label)}</div><div class="mute">${esc(r.node||"starting…")} · ${dur(r.elapsed)}</div>`+
      (r.stepped?`<div class="bar" style="margin-top:8px"><i style="width:${p}%"></i></div><div class="num mute">${r.max?`step ${r.value}/${r.max}`:""}</div>`
                :`<div class="mute" style="margin-top:6px">no step detail (not submitted by the pipeline)</div>`)}).join("");
    document.title="▶ "+s.running[0].label+(s.running[0].max?` ${s.running[0].value}/${s.running[0].max}`:"")}
  else{$("run").innerHTML="<h2>Running</h2><div class='mute'>idle</div>";document.title="● "+name()+" idle"}
  $("pend").innerHTML=s.pending.length?s.pending.map((p,i)=>`<div class="row"><span>${i+1}. ${esc(p.label)}</span></div>`).join(""):"<div class='mute'>queue empty</div>";
  $("hist").innerHTML=s.history.map(h=>`<div class="row"><span>${esc(h.label)}</span><span class="num"><span class="${h.status==="success"?"ok":"bad"}">${esc(h.status||"?")}</span> <span class="mute">${dur(h.duration)}</span></span></div>`).join("")||"<div class='mute'>no history</div>";
  if(last&&last.online){
    const prev=new Set(last.history.map(h=>h.n));
    s.history.filter(h=>!prev.has(h.n)).forEach(h=>notify(h.status==="success"?"Job finished":"Job FAILED",h.label+" · "+dur(h.duration)));
    if(last.running.length&&!s.running.length&&!s.pending.length)notify("Queue empty","Nothing left to run")}
  last=s}
async function tick(){const t=target;try{const s=await (await fetch("/api/state?target="+t)).json();if(t===target)render(s)}catch(e){$("conn").textContent="dashboard lost";$("conn").className="pill bad"}}
async function post(p){await fetch(p,{method:"POST"});tick()}
$("t-local").onclick=()=>setTarget("local");$("t-cloud").onclick=()=>setTarget("cloud");$("t-runpod").onclick=()=>setTarget("runpod");
$("stop").onclick=()=>post("/api/interrupt");
$("clear").onclick=()=>confirm("Clear all pending jobs?")&&post("/api/clear");
$("notif").onclick=async()=>{try{await Notification.requestPermission()}catch(e){}$("notif").textContent=Notification.permission==="granted"?"Notifications on":"Notifications blocked"};
if(window.Notification&&Notification.permission==="granted")$("notif").textContent="Notifications on";
setTarget(target);setInterval(tick,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state":
            if "target=cloud" in u.query:
                target = "cloud"
            elif "target=runpod" in u.query:
                target = "runpod"
            else:
                target = "local"
            try:
                return self._send(200, json.dumps({"cloud": cloud_state, "runpod": runpod_state, "local": local_state}[target]()))
            except Exception as e:  # never let one bad payload take the page down
                return self._send(200, json.dumps({"target": target, "online": False, "error": f"dashboard error: {e}"}))
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._send(404, "{}")

    def do_POST(self):  # local only: the cloud view is read-only
        try:
            if self.path == "/api/interrupt":
                LOCAL.call("/interrupt", {})
            elif self.path == "/api/clear":
                LOCAL.call("/queue", {"clear": True})
            else:
                return self._send(404, "{}")
            self._send(200, "{}")
        except Exception as e:
            self._send(502, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        # keep stderr quiet: BaseHTTPRequestHandler logs every request there
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--comfy", default=LOCAL.base, help="local ComfyUI address")
    a = ap.parse_args()
    LOCAL.base = a.comfy.rstrip("/")
    if config.cloud_key_present():
        key = comfy_api.api_key()
        CLOUD.headers, CLOUD.query = {"X-API-Key": key}, f"&token={key}"
        threading.Thread(target=ws_loop, args=(CLOUD,), daemon=True).start()
    else:
        print("no COMFY_CLOUD_KEY: the cloud view is disabled", flush=True)
    threading.Thread(target=ws_loop, args=(LOCAL,), daemon=True).start()
    threading.Thread(target=ws_loop, args=(RUNPOD,), daemon=True).start()  # fails and retries every few seconds while the pod is down
    # loopback-only page; TLS has nothing to protect
    print(f"dashboard on http://127.0.0.1:{a.port}  (local ComfyUI at {LOCAL.base}, cloud view #cloud)", flush=True)  # NOSONAR
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()  # NOSONAR(S5332)


if __name__ == "__main__":
    main()
