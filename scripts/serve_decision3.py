"""Optional local Decision 3.0 service. Run in a separate Transformers 5.17 environment."""

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vllm-sr/d3-flash")
    p.add_argument("--revision", required=True, help="Immutable Hugging Face commit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--warmup", action="store_true", help="Compile kernels before reporting ready")
    args = p.parse_args()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, device_map="auto", torch_dtype="auto"
    )
    runtime = model.runtime
    logging.info("Decision device=%s kernels=%s", runtime.device, json.dumps(runtime.kernels))
    if args.warmup:
        logging.info("Decision warmup started")
        seconds = runtime.warmup()
        logging.info("Decision warmup completed in %.3fs", seconds)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data):
            body = json.dumps(data).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logging.info("Client disconnected; completed result discarded")

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "unknown endpoint"})
            self.reply(200, dict(model=args.model, revision=args.revision, ready=True,
                                 device=str(runtime.device), kernels=runtime.kernels))

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.reply(404, {"error": "unknown endpoint"})
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32 << 20:
                return self.reply(413, {"error": "invalid request size"})
            try:
                payload = json.loads(self.rfile.read(size))
                if payload.pop("model", args.model) != args.model:
                    raise ValueError("Unexpected model")
                if set(payload) - {"state", "images", "questions"}:
                    raise ValueError("Unsupported request fields")
                started = time.monotonic()
                result = model.system_one(**payload)
                seconds = time.monotonic() - started
                result["meta"] = dict(model_id=args.model, revision=args.revision,
                                      system_one_seconds=seconds)
                logging.info("Decision completed in %.3fs", seconds)
                self.reply(200, result)
            except Exception as error:
                self.reply(422, dict(error=f"{type(error).__name__}: {error}"))

    logging.info("DECISION_READY model=%s revision=%s", args.model, args.revision)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
