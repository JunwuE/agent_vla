"""Start one model service using the same configuration as the robot agent."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from vla_agent.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("vla", "decision", "flora"))
    parser.add_argument("--config", default="configs/robocasa.yaml")
    parser.add_argument("--decision-python", default="/home/junwue/envs/decision3/bin/python")
    args = parser.parse_args()
    config = load_config(args.config)
    scripts = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/share/project/robot-rfm/cache/huggingface")
    env.setdefault("OPENPI_DATA_HOME", "/share/project/robot-rfm/cache/openpi")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.component == "vla":
        c = config.vla
        endpoint = urlparse(c.endpoint)
        recipe = "omnisf" if c.training_config == "robocasa365_target_atomic" else "robocasa"
        command = [c.python, str(scripts / "serve_pi05.py"), "--source", c.openpi_path,
                   "--recipe", recipe, "--checkpoint", c.checkpoint, "--norm-stats", c.norm_stats,
                   "--normalization", c.normalization, "--warmup"]
    elif args.component == "decision" and config.decision.kind != 'flora':
        c = config.decision
        endpoint = urlparse(c.endpoint)
        command = [args.decision_python, str(scripts / "serve_decision3.py"),
                   "--model", c.model, "--revision", c.revision, "--warmup"]
    else:
        c = config.flora
        endpoint = urlparse(c.endpoint)
        command = [c.python, str(scripts / "serve_flora.py"), "--model", c.model,
                   "--revision", c.revision, "--warmup"]
        if c.quantization == "int8":
            command.append("--int8")
        if c.gpu_memory_gib is not None:
            command += ["--gpu-memory-gib", str(c.gpu_memory_gib)]
    if not endpoint.hostname or not endpoint.port or any(value is None for value in command):
        parser.error("Model paths/revision and endpoint host/port must be configured")
    command += ["--host", endpoint.hostname, "--port", str(endpoint.port)]
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
