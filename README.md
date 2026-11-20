# tndrr AI/ML internship assignment

[Three-page PDF report](report.pdf)

| Task | Code and recorded results |
| --- | --- |
| LoRA | [Training and evaluation](task1_lora/experiment.py), [60 held-out predictions](task1_lora/results.json), [base checkpoint](task1_lora/checkpoint), [adapter](task1_lora/adapter) |
| Retrieval | [RAG pipeline](task2_rag/rag.py), [15 questions and gold passages](task2_rag/questions.json), [ranked passages and answers](task2_rag/results.json) |
| Local inference | [GGUF conversion](task3_local/convert.py), [llama.cpp benchmark](task3_local/benchmark.py), [ten paired outputs and timings](task3_local/results.json) |

## Set up and reproduce

Use Linux x86-64 with Python 3.12, curl, tar, sha256sum and Poppler's
`pdftotext`. The experiments were run on CPU. From this checkout:

```sh
./setup.sh
.venv/bin/python task1_lora/experiment.py
.venv/bin/python task3_local/benchmark.py
```

Setup verifies pinned downloads, installs Python dependencies in `.venv` and
uses the bundled base weights to create F16 and Q4_0 GGUF files in `.cache`.
The benchmark starts and stops its own local servers. It checks tokenization
against Transformers and measures each setting on identical prompts.

To reproduce the base weights from the source text, run
`.venv/bin/python task1_lora/pretrain.py` before rerunning setup. This writes
the training output under `.cache/base`. The released checkpoint was trained
on *Pro Git* pages 101 onward. See the [model card](task1_lora/checkpoint/MODEL_CARD.md)
for the source and license.

## Run retrieval and question answering

Start the local generator from the checkout in one terminal:

```sh
.cache/llama-b12359/llama-server -m .cache/progit-f16.gguf --host 127.0.0.1 \
  --port 8080 -t 4 -tb 4 -ngl 0 -c 1024 -np 1 --cache-ram 0 --no-ui
```

Then run the health check and evaluation in another terminal:

```sh
curl --fail --silent http://127.0.0.1:8080/health
.venv/bin/python task2_rag/rag.py --endpoint http://127.0.0.1:8080
.venv/bin/python task2_rag/rag.py --endpoint http://127.0.0.1:8080 \
  --ask "When I amend the latest commit, does the previous commit remain in the branch history?"
```

The single question should return a quotation from PDF page 53. Running
`rag.py` without an endpoint evaluates retrieval only and overwrites its
results without answer fields. Use the endpoint when rebuilding the PDF:

```sh
.venv/bin/python build_report.py
```

The JSON files record actual runs. Timing and RSS vary by hardware and load.
The tiny checkpoint is experimental and can produce incorrect Git commands.
