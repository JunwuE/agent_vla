"""Measure fixed historical requests in the Decision environment; never execute robot actions.

Requests are a JSON list of {id, payload}, where payload contains the same state,
images and questions sent to /v1/systemone. Keep the file unchanged between runs.
Model loading and per-request warmups are excluded from the reported warm latency.
The run stage includes model-side image processing, GPU execution and score extraction;
it is deliberately not described as pure GPU forward time.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="vllm-sr/d3-flash")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if args.output.exists():
        parser.error("output already exists")

    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    raw = args.requests.read_bytes()
    requests = json.loads(raw)
    if not requests:
        raise ValueError("At least one fixed request is required")
    versions = {}
    for name in ("torch", "transformers", "triton", "causal-conv1d", "flash-linear-attention", "fla-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        device_map="auto", torch_dtype="auto", local_files_only=True,
    )
    runtime = model.runtime
    torch.cuda.synchronize()
    report = dict(
        model=args.model, revision=args.revision, gpu=torch.cuda.get_device_name(0),
        versions=versions, cuda=torch.version.cuda, load_seconds=time.perf_counter() - started,
        requests_sha256=hashlib.sha256(raw).hexdigest(), warmup=args.warmup, repeats=args.repeats,
        device=str(runtime.device), kernels=runtime.kernels,
        scope="Local system_one stages; excludes HTTP, client image encoding, model loading and warmups",
        cases=[],
    )
    for request in requests:
        payload = dict(request["payload"])
        payload.pop("model", None)
        samples, warmups = [], []
        answers = []
        for index in range(args.warmup + args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            prepared = runtime.prepare(payload["state"], payload["questions"], payload.get("images"), None)
            torch.cuda.synchronize()
            prepared_at = time.perf_counter()
            probabilities, tokens = runtime.run(prepared)
            torch.cuda.synchronize()
            run_at = time.perf_counter()
            response = runtime.respond(prepared, probabilities, tokens)
            end = time.perf_counter()
            if any("error" in answer for answer in response["answers"].values()):
                raise RuntimeError(f"Invalid benchmark response: {response}")
            sample = dict(total_seconds=end - start, prepare_seconds=prepared_at - start,
                          run_seconds=run_at - prepared_at, respond_seconds=end - run_at)
            if index < args.warmup:
                warmups.append(sample)
            else:
                samples.append(sample)
                answers.append(response)
        result = dict(id=request["id"], image_count=len(payload.get("images", [])),
                      input_tokens=answers[-1]["usage"]["input_tokens"], samples=samples,
                      warmups=warmups, responses=answers,
                      median_seconds=statistics.median(s["total_seconds"] for s in samples))
        report["cases"].append(result)
        print(json.dumps({k: result[k] for k in ("id", "input_tokens", "median_seconds")}), flush=True)
    report["median_seconds"] = statistics.median(
        sample["total_seconds"] for case in report["cases"] for sample in case["samples"]
    )
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(output=str(args.output), median_seconds=report["median_seconds"])), flush=True)


if __name__ == "__main__":
    main()
