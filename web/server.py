import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from web.games import GAMES
from web.games.common import PlayerSpec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

SESSION = None


def _list_checkpoints(game_name: str):
    game_name_lower = game_name.lower()
    data_roots = [
        "data_az",
        "data_muzero",
        "data_mz",
        "data_stochastic_mz",
        "data_az_ctree",
        "data_az_ptree",
    ]
    game = GAMES.get(game_name)
    supported_algos = set()
    if game is not None and hasattr(game, "supported_policy_configs"):
        supported_algos = set(game.supported_policy_configs.keys())
    found = []
    for root in data_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "ckpt" not in dirpath:
                continue
            for name in filenames:
                if not name.endswith(".pth.tar"):
                    continue
                path = os.path.join(dirpath, name)
                if game_name_lower not in path.lower():
                    continue
                lower = path.lower()
                algo = None
                if "stochastic" in lower:
                    algo = "stochastic_muzero"
                elif "gumbel" in lower:
                    algo = "gumbel_muzero"
                elif "muzero" in lower or "mz" in lower:
                    algo = "muzero"
                elif "alphazero" in lower or "az" in lower:
                    algo = "alphazero"
                supported = algo in supported_algos
                found.append(
                    {
                        "path": path,
                        "algo": algo,
                        "supported": supported,
                    }
                )
    return sorted(found, key=lambda x: x["path"])


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_file(os.path.join(BASE_DIR, "index.html"), "text/html; charset=utf-8")
        if parsed.path == "/api/games":
            payload = [
                {"name": game.name, "label": game.label}
                for game in GAMES.values()
            ]
            return self._send_json({"games": payload})
        if parsed.path == "/api/checkpoints":
            params = parse_qs(parsed.query)
            game = params.get("game", [None])[0]
            if game not in GAMES:
                return self._send_json({"error": "unknown game"}, status=400)
            game_obj = GAMES.get(game)
            supported_algos = []
            if game_obj is not None and hasattr(game_obj, "supported_policy_configs"):
                supported_algos = sorted(game_obj.supported_policy_configs.keys())
            return self._send_json(
                {
                    "checkpoints": _list_checkpoints(game),
                    "supported_algos": supported_algos,
                }
            )
        if parsed.path == "/api/state":
            if SESSION is None:
                return self._send_json({"error": "no active session"}, status=400)
            return self._send_json({"state": SESSION.state()})
        if parsed.path.startswith("/static/"):
            rel = parsed.path[len("/static/"):]
            safe = os.path.normpath(rel)
            if safe.startswith(".."):
                return self._send_json({"error": "invalid path"}, status=400)
            target = os.path.join(STATIC_DIR, safe)
            if not os.path.isfile(target):
                return self._send_json({"error": "not found"}, status=404)
            if target.endswith(".js"):
                content_type = "application/javascript"
            elif target.endswith(".css"):
                content_type = "text/css"
            else:
                content_type = "application/octet-stream"
            return self._send_file(target, content_type)
        return self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        global SESSION
        parsed = urlparse(self.path)
        if parsed.path == "/api/session":
            body = self._read_json()
            game_name = body.get("game")
            if game_name not in GAMES:
                return self._send_json({"error": "unknown game"}, status=400)
            players = body.get("players") or {}
            auto_play = bool(body.get("auto_play", True))
            try:
                p1 = PlayerSpec(**(players.get("1") or {"player_type": "human"}))
                p2 = PlayerSpec(**(players.get("2") or {"player_type": "human"}))
            except TypeError as exc:
                return self._send_json({"error": f"invalid player spec: {exc}"}, status=400)
            SESSION = GAMES[game_name].new_session({1: p1, 2: p2}, auto_play=auto_play)
            return self._send_json({"state": SESSION.state()})
        if parsed.path == "/api/step":
            if SESSION is None:
                return self._send_json({"error": "no active session"}, status=400)
            body = self._read_json()
            action = body.get("action")
            if action is None:
                return self._send_json({"error": "missing action"}, status=400)
            try:
                SESSION.apply_human_action(int(action))
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=400)
            return self._send_json({"state": SESSION.state()})
        if parsed.path == "/api/auto":
            if SESSION is None:
                return self._send_json({"error": "no active session"}, status=400)
            SESSION.advance()
            return self._send_json({"state": SESSION.state()})
        return self._send_json({"error": "not found"}, status=404)


def main():
    port = int(os.environ.get("LIGHTZERO_WEB_PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"LightZero web server running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
