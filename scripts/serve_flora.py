"""Local, resident Flora Outcome service. Official prompt/sampling contract, Transformers backend.

This is a single-GPU service; account for model memory before colocating with pi05/Decision.
Model scores do not execute actions or declare subtask completion.
"""

import argparse
import base64
import importlib.util
import io
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Flora-Rewarder")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--gpu-memory-gib", type=int, default=None,
                        help="Optional BF16 CPU offload; caps model placement, not activation memory")
    parser.add_argument("--int8", action="store_true", help="Experimental bitsandbytes int8 weights")
    args = parser.parse_args()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Flora service requires a CUDA GPU")
    root = snapshot_download(
        args.model, revision=args.revision, local_files_only=True,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "batch_utils/*.py", "README.md", "LICENSE*"],
    )
    spec = importlib.util.spec_from_file_location("flora_official_core", root + "/batch_utils/core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    processor = AutoProcessor.from_pretrained(root)
    placement = dict(device_map="cuda:0")
    if args.int8:
        from transformers import BitsAndBytesConfig

        if args.gpu_memory_gib is not None:
            parser.error("Choose int8 or BF16 CPU offload, not both")
        placement["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if args.gpu_memory_gib is not None:
        if args.gpu_memory_gib < 8:
            parser.error("GPU model placement budget must be at least 8 GiB")
        placement = dict(device_map="auto", max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "128GiB"})
    model = AutoModelForImageTextToText.from_pretrained(
        root, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **placement,
    ).eval()
    # This checkpoint's generation config stops at endoftext, while its chat tokenizer
    # terminates assistant turns with im_end. Honor both to avoid generating another turn.
    configured_eos = model.generation_config.eos_token_id
    stop_tokens = sorted({token for token in (
        *(configured_eos if isinstance(configured_eos, list) else [configured_eos]),
        processor.tokenizer.eos_token_id,
    ) if token is not None})

    def score(payload):
        if payload.get("model") != args.model or payload.get("revision") != args.revision:
            raise ValueError("Model/revision mismatch")
        if payload.get("quantization", "none") != ("int8" if args.int8 else "none"):
            raise ValueError("Quantization mismatch")
        if payload.get("mode") != "outcome" or type(payload.get("with_rationale")) is not bool:
            raise ValueError("Outcome with an explicit rationale flag is required")
        with_rationale = payload['with_rationale']
        if not isinstance(payload.get("task"), str) or not payload["task"].strip():
            raise ValueError("Task is required")
        frames = payload["frames"]
        if not 2 <= len(frames) <= 64 or len(frames) % 2:
            raise ValueError("Outcome requires 2..64 sampled frames, even count")
        images = []
        for frame in frames:
            with Image.open(io.BytesIO(base64.b64decode(frame, validate=True))) as image:
                if image.width * image.height > 4 * 256 * 256:
                    raise ValueError("Frame exceeds pixel budget")
                images.append(np.array(image.convert("RGB")))
        video = np.stack(images)
        metadata = payload["video_metadata"]
        if len(metadata["frames_indices"]) != len(frames):
            raise ValueError("Frame timestamps do not match video")
        if not 0 < metadata["fps"] <= 240 or not 0 < metadata["duration"] <= 120:
            raise ValueError("Invalid video clock or duration")
        indices = metadata["frames_indices"]
        if indices != sorted(indices) or indices[0] < 0 or indices[-1] >= metadata["total_num_frames"]:
            raise ValueError("Invalid sampled indices")
        messages = core.terminal_messages(payload["task"], with_rationale=with_rationale)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False, add_vision_id=False)
        torch.cuda.synchronize()
        started = time.monotonic()
        inputs = processor(text=[prompt], videos=[video], video_metadata=[metadata],
                           do_sample_frames=False,
                           size={"shortest_edge": 4096, "longest_edge": 29_360_128},
                           return_tensors="pt").to("cuda:0")
        tokens = inputs["input_ids"].shape[1]
        if tokens + core.terminal_max_tokens(with_rationale) > 16384:
            raise ValueError("Outcome input exceeds 16K token budget")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=core.terminal_max_tokens(with_rationale),
                                    eos_token_id=stop_tokens, pad_token_id=processor.tokenizer.pad_token_id)
        text = processor.batch_decode(output[:, tokens:], skip_special_tokens=True)[0].strip()
        value, rationale = core.parse_terminal(text, with_rationale=with_rationale)
        torch.cuda.synchronize()
        return dict(request_id=payload["request_id"], model=args.model, revision=args.revision,
                    score=value, raw_output=text, input_tokens=int(tokens), sampled_frames=len(frames),
                    seconds=time.monotonic() - started, backend="transformers", with_rationale=with_rationale,
                    rationale=rationale, gpu_memory_gib=args.gpu_memory_gib,
                    quantization="int8" if args.int8 else "none")

    if args.warmup:
        buffer = io.BytesIO()
        Image.new("RGB", (512, 256), (100, 100, 100)).save(buffer, format="PNG")
        frame = base64.b64encode(buffer.getvalue()).decode()
        logging.info("Flora warmup started")
        for count in (2, 32):
            score(dict(request_id="warmup", model=args.model, revision=args.revision, mode="outcome",
                       quantization="int8" if args.int8 else "none",
                       with_rationale=False, task="Pick up the cup.", frames=[frame] * count,
                       video_metadata=dict(fps=10, total_num_frames=count, frames_indices=list(range(count)),
                                           duration=count / 10, video_backend="recorded_rgb")))
        logging.info("Flora warmup completed")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logging.info("Caller disconnected; no robot state is changed")

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "Unknown endpoint"})
            self.reply(200, dict(ready=True, model=args.model, revision=args.revision, mode="outcome",
                                 device=str(model.device), rationale_modes=[False, True], backend="transformers",
                                 stop_token_ids=stop_tokens, gpu_memory_gib=args.gpu_memory_gib,
                                 quantization="int8" if args.int8 else "none",
                                 device_map={k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}))

        def do_POST(self):
            if self.path != "/v1/outcome":
                return self.reply(404, {"error": "Unknown endpoint"})
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32 << 20:
                return self.reply(413, {"error": "Request exceeds size budget"})
            try:
                result = score(json.loads(self.rfile.read(size)))
                self.reply(200, result)
            except Exception as error:
                logging.exception("Flora scoring failed")
                self.reply(422, dict(error=f"{type(error).__name__}: {error}"))

    logging.info("FLORA_READY model=%s revision=%s device=%s", args.model, args.revision, model.device)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
