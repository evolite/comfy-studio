"""A stand-in ComfyUI HTTP server: queues nothing, runs every prompt instantly, and serves a 1x1 PNG for each SaveImage output.
Enough for LocalTarget end to end (system_stats, object_info, upload, prompt, history, view, free) without a GPU."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082")


class FakeComfy:
    def __init__(self, classes=None):
        self.classes = set(classes or ())      # node classes object_info reports; empty = accept everything seen in a prompt
        self.prompts, self.uploads, self.freed, self.history = [], [], 0, {}
        self.fail_next = False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, data, ctype="application/json"):
                body = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/system_stats":
                    return self.send({"system": {}})
                if u.path == "/object_info":
                    classes = outer.classes or {n["class_type"] for p in outer.prompts for n in p.values()}
                    return self.send({c: {"input": {"required": {}}} for c in classes} or {"SaveImage": {"input": {"required": {}}}})
                if u.path == "/history":
                    return self.send(dict(list(outer.history.items())[-1:]))
                if u.path.startswith("/history/"):
                    pid = u.path.rsplit("/", 1)[1]
                    return self.send({pid: outer.history[pid]} if pid in outer.history else {})
                if u.path == "/view":
                    return self.send(PNG, "image/png")
                self.send_error(404)

            def do_POST(self):
                u = urlparse(self.path)
                data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if u.path == "/upload/image":
                    name = data.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
                    outer.uploads.append(name)
                    return self.send({"name": name})
                if u.path == "/free":
                    outer.freed += 1
                    return self.send({})
                if u.path == "/prompt":
                    graph = json.loads(data)["prompt"]
                    outer.prompts.append(graph)
                    pid = f"p{len(outer.prompts)}"
                    ok = not outer.fail_next
                    outer.fail_next = False
                    t = time.time() * 1000
                    outputs = {nid: {"images": [{"filename": f"{pid}_{nid}.png", "subfolder": "", "type": "output"}]}
                               for nid, n in graph.items() if n["class_type"].startswith("Save")}
                    outer.history[pid] = {"prompt": [len(outer.prompts), pid, graph], "outputs": outputs if ok else {},
                                          "status": {"status_str": "success" if ok else "error", "completed": ok,
                                                     "messages": [["execution_start", {"timestamp": t}], ["execution_success", {"timestamp": t + 2500}]]}}
                    return self.send({"prompt_id": pid})
                self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
