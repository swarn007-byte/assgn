"""Compare the same checkpoint in fresh CPU llama-server processes."""
from contextlib import contextmanager
import json
import shlex
from pathlib import Path
from socket import socket
from statistics import median
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
SERVER = CACHE / "llama-b12359/llama-server"
KEYS = ("f16", "q4_0")
PROMPTS = [
    ("To initialize an empty Git repository, run", ["git init"]),
    ("To find out which files have changed, run", ["git status"]),
    ("To stage the file README.md, run", ["git add README.md"]),
    ("To view the commit history, run", ["git log"]),
    ("To create a new branch named demo without switching to it, run", ["git branch demo"]),
    ("To temporarily store uncommitted changes, run", ["git stash"]),
    ("To download data from origin without merging, run", ["git fetch origin"]),
    ("To show the difference between the index and the previous commit, run",
     ["git diff --cached", "git diff --staged"]),
    ("To list the remotes along with their URLs, run", ["git remote -v"]),
    ("To save the staged snapshot with the message Fix login, run",
     ['git commit -m "Fix login"']),
]


def request(endpoint, route, data=None):
    if data is not None:
        data = json.dumps(data).encode()
    req = Request(endpoint + route, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as response:
        return json.load(response)


@contextmanager
def start(key, checkpoint="progit"):
    with socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    with (CACHE / f"server-{key}.log").open("w") as log:
        with subprocess.Popen([
            str(SERVER), "-m", str(CACHE / f"{checkpoint}-{key}.gguf"),
            "--host", "127.0.0.1", "--port", str(port), "-t", "4", "-tb", "4",
            "-ngl", "0", "-c", "1024", "-np", "1", "--cache-ram", "0", "--no-ui",
        ], stdout=log, stderr=log) as server:
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError(f"llama-server quit. See {log.name}")
                    try:
                        if request(endpoint, "/health")["status"] == "ok":
                            break
                    except (URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    raise TimeoutError(f"llama-server never became ready. See {log.name}")
                yield endpoint, server.pid
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def complete(endpoint, prompt, *, force_length=True):
    return request(endpoint, "/completion", {
        "prompt": prompt, "n_predict": 32, "ignore_eos": force_length, "temperature": 0,
        "seed": 713, "cache_prompt": False,
    })


def peak_rss(pid):
    status = Path(f"/proc/{pid}/status").read_text()
    return int(next(line for line in status.splitlines() if line.startswith("VmHWM:")).split()[1])


def command_matches(output, expected):
    try:
        return shlex.split(output.strip()) in [shlex.split(command) for command in expected]
    except ValueError:
        return False


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CACHE / "base")
    results = {key: {"file_bytes": (CACHE / f"commands-{key}.gguf").stat().st_size,
                     "peak_rss_kib": 0, "timing_runs": [], "samples": []} for key in KEYS}
    for order in (KEYS, KEYS[::-1], KEYS):
        for key in order:
            result = results[key]
            timings_run = []
            with start(key, "commands") as (endpoint, pid):
                for prompt, _ in PROMPTS:
                    ids = request(endpoint, "/tokenize", {"content": prompt, "add_special": False})["tokens"]
                    assert ids == tokenizer.encode(prompt, add_special_tokens=False), (key, prompt)
                complete(endpoint, "Git tracks changes to files.")
                for prompt, _ in PROMPTS:
                    timings = complete(endpoint, prompt)["timings"]
                    assert timings["predicted_n"] == 32 and timings["predicted_ms"] > 0, timings
                    timings_run.append(timings)
                result["timing_runs"].append(timings_run)
                if not result["samples"]:
                    for prompt, expected in PROMPTS:
                        output = complete(endpoint, prompt, force_length=False)["content"]
                        result["samples"].append({
                            "prompt": prompt, "accepted_commands": expected, "output": output,
                            "contains_expected": command_matches(output, expected),
                        })
                result["peak_rss_kib"] = max(result["peak_rss_kib"], peak_rss(pid))
    for result in results.values():
        result["median_decode_tokens_per_second"] = median(
            row["predicted_per_second"] for run in result["timing_runs"] for row in run
        )
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print({key: {k: v for k, v in result.items() if k not in {"samples", "timing_runs"}}
           for key, result in results.items()})


if __name__ == "__main__":
    main()
