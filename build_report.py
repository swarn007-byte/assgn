"""Generate the three-page internship assignment report from measured results."""
from collections import Counter
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def build():
    lora = read("task1_lora/results.json")
    rag = read("task2_rag/results.json")
    local = read("task3_local/results.json")
    navy, teal, gray = "#12253B", "#006B63", "#506173"
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                          textColor=colors.HexColor(navy), spaceAfter=8)
    section = ParagraphStyle("Section", parent=body, fontName="Helvetica-Bold", fontSize=12,
                             leading=15, spaceBefore=10, spaceAfter=5)
    title = ParagraphStyle("Title", parent=section, fontSize=25, leading=28, spaceAfter=12)
    kicker = ParagraphStyle("Kicker", parent=section, fontSize=9, leading=12,
                            textColor=colors.HexColor(teal), spaceBefore=0, spaceAfter=7)
    right = ParagraphStyle("Right", parent=body, alignment=TA_RIGHT)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def heading(number, text):
        p(f"TNDRR AI/ML INTERNSHIP ASSIGNMENT  /  TASK {number}", kicker)
        p(text, title)

    def table(header, rows, widths):
        data = [[Paragraph(escape(str(cell)), body if i == 0 else right)
                 for i, cell in enumerate(row)] for row in [header, *rows]]
        grid = Table(data, colWidths=widths, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DEEEEB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([grid, Spacer(1, 6)])

    heading(1, "LoRA for Git intent classification")
    p("The base is a locally pretrained, open-weight 1.31M-parameter Llama. Its tokenizer "
      "and weights are bundled. Pretraining used 183,863 tokens from <i>Pro Git</i> PDF pages "
      "101 onward. The four labels are stage, commit, branch and push. Both models chose "
      "the highest-scoring of four single-token labels from the same prompt.")
    n = lora["test_count"]
    table(["Checkpoint", "Correct", "Accuracy"], [
        (name, f"{lora[key]}/{n}", f"{lora[key] / n:.1%}")
        for name, key in [("Frozen base", "base_correct"), ("LoRA", "lora_correct")]
    ], [225, 110, 140])
    p(f"The {lora['train_count']} training requests and {n} test requests use disjoint "
      "templates and slot values. The test has 15 requests per class, with five slot values "
      "per template. Accuracy measures this template-based test, so requests derived from "
      "the same template are correlated.")
    p("Adapter settings", section)
    p(f"Rank {lora['rank']}, alpha {lora['alpha']}, targets <font name='Courier'>q_proj</font> "
      f"and <font name='Courier'>v_proj</font>, AdamW learning rate {lora['learning_rate']}, "
      f"{lora['epochs']} epochs, batch size 12. Rank 8 limits the update size; alpha 16 "
      "gives a scale of 2. Query and value projections can change what gets attended to "
      "and what information attention carries. The learning rate and epoch count were "
      "fixed for this small supervised run. Only "
      f"{lora['trainable_parameters']:,} parameters were trained. Training loss fell from "
      f"{lora['epoch_losses'][0]:.2f} to {lora['epoch_losses'][-1]:.3f}.")
    p("Three held-out failures", section)
    names = dict(A="stage", B="commit", C="branch", D="push")
    for i, reason in [
        (20, "The request avoids the word <i>commit</i>; revision could evoke a line of history."),
        (35, "Independent line is an indirect cue, and the request omits <i>branch</i>."),
        (45, "The destination is a bare name, with no explicit <i>push</i> or <i>remote</i>."),
    ]:
        row = lora["examples"][i]
        assert row["label"] != row["lora"]
        p(f"“{escape(row['request'])}” → {names[row['lora']]} "
          f"(gold {names[row['label']]}). {reason}")
    p("7B training memory estimate", section)
    p("Assume 7B weights, BF16 weights and gradients, FP32 Adam moments and a master "
      "copy, 32 Llama-2-style layers and 4,096-wide Q/V projections. Decimal GB below. "
      "Rank-8 Q/V LoRA has 32 × 2 × 8 × (4,096 + 4,096) ≈ 4.19M adapter parameters.")
    p("<b>Full:</b> 7B × (2 + 2 + 8 + 4) bytes = 112 GB. "
      "<b>LoRA:</b> 7B × 2 bytes + 4.19M × 16 bytes ≈ 14.07 GB. "
      "<b>QLoRA:</b> 7B × 0.6 bytes (4-bit plus scales and higher-precision tensors, "
      "as a rough effective average) + 4.19M × 16 bytes ≈ 4.27 GB.")
    p("With an illustrative 2–6 GB for checkpointed activations and workspace at small "
      "microbatch and short context, totals are roughly 114–118 GB, 16–20 GB, and 6–10 GB. "
      "Sequence length, kernels and optimizer implementation can shift these totals.")

    story.append(PageBreak())
    heading(2, "Retrieval from scratch")
    p("Corpus: Scott Chacon and Ben Straub’s <i>Pro Git</i> 2.1.450, PDF pages 32–100 "
      "(69 pages). Chunk boundaries respect pages and use 25% overlap. The embedding "
      "is an L2-normalized, chunk-IDF-weighted mean of pretrained 50-dimensional GloVe "
      "word vectors. NumPy ranks by cosine similarity. The same 15 authored questions "
      "are used in both runs. A hit means a top-five chunk on the gold page contains "
      "the full annotated passage.")
    table(["Chunk size", "Indexed chunks", "Recall@5"], [
        (f"{size} words", run["chunks"], f"{run['hits']}/{run['total']} ({run['hits']/run['total']:.1%})")
        for size, run in rag.items()
    ], [225, 110, 140])
    p("The 160-word windows won on this question set. For the ignore-pattern exception, "
      "page 38 reached rank 4 only in the longer run. Larger windows can connect a "
      "short rule to the surrounding explanation, though the smaller index also "
      "changes the number of competing windows. The result is specific to these questions.")
    p("Two retrieval failures", section)
    p("<b>Staged diff flags:</b> The reference is on PDF page 41. The longer run returned "
      "pages 50, 51, 40, 40 and 51. The scoring vocabulary strips punctuation and averages "
      "word meanings, so rare exact command syntax contributes weakly. Preserve flags as "
      "tokens and combine BM25 with vector scores.")
    p("<b>Remote URLs:</b> The reference is on PDF page 57. The longer run returned pages "
      "100, 59, 100, 100 and 34. The paraphrase ‘remote nickname’ competes with many "
      "remote-related passages. Expand it with ‘shortname’ and re-rank for listing both "
      "fetch and push URLs.")
    p("From retrieved text to an answer", section)
    modes = Counter(row["generated_answer"]["mode"] for row in rag["160"]["rows"])
    p(f"The local Llama receives the top three excerpts. The system accepts generated "
      f"text only if it appears verbatim in a retrieved excerpt. All {modes['retrieved_quote']} "
      "questions used the quoted-evidence fallback, which ranks excerpts by word overlap "
      "and supplies their page. For the amend question it correctly quoted page 53, "
      "which explains that the previous commit disappears from the branch history. "
      "The guard prevents unsupported wording, while relevance still depends on retrieval "
      "and quote selection.")
    p("What BM25 and re-ranking add", section)
    p("BM25 rewards exact matches, with larger weights for uncommon terms, diminishing "
      "returns for repeated terms, and a length adjustment. This helps with command "
      "names or flags whose literal form matters. A re-ranker scores the question and "
      "each retrieved candidate together, so it can assess whether the passage actually "
      "answers the question. It only sees the candidates it is given.")
    p("The repository contains all 15 questions, verbatim gold passages, top-five ranks, "
      "and generated and displayed answers.")

    story.append(PageBreak())
    heading(3, "Local inference and quantisation")
    p("The same Pro Git checkpoint ran in llama.cpp b12359 on an AMD EPYC 9V74 cloud "
      "CPU. Each setting used four inference threads, a 1,024-token context, one slot, "
      "greedy decoding, one warm-up request, and 32 generated tokens per prompt. Decode "
      "rate below is the median of ten server timings and excludes prompt ingestion. "
      "Peak RSS is the server’s Linux VmHWM, including executable, model and buffers.")
    table(["Setting", "Decode tok/s", "Peak RSS MiB", "GGUF MiB", "Command hits"], [
        (key.upper(), f"{run['median_decode_tokens_per_second']:,.0f}",
         f"{run['peak_rss_kib']/1024:.2f}", f"{run['file_bytes']/2**20:.2f}",
         f"{sum(row['contains_expected'] for row in run['samples'])}/{len(run['samples'])}")
        for key, run in local.items()
    ], [85, 102, 99, 92, 97])
    matched = sum(a["output"] == b["output"]
                  for a, b in zip(local["f16"]["samples"], local["q4_0"]["samples"]))
    p(f"Q4_0 decoded faster in this run and reduced the GGUF size. {matched}/10 outputs "
      "were identical. Both supplied a memorized commit example for the initialization "
      "prompt, where <font name='Courier'>git init</font> was expected. The branch prompt "
      "diverged: F16 began with <font name='Courier'>git log</font>, Q4_0 with "
      "<font name='Courier'>git commit -s</font>. Both were wrong. This checkpoint’s "
      "0/10 baseline leaves little room for detecting quality loss with this rubric. "
      "The repo includes all ten paired outputs and command checks.")
    p("KV cache and weight quantisation", section)
    p("Each attention layer saves key and value vectors for past tokens. During decoding, "
      "the new token’s query attends to these saved vectors, avoiding repeated projection "
      "of the whole prefix. Cache memory grows with context and active sequences. "
      "Weight quantisation represents weights with fewer bits plus block scales, "
      "shrinking the model and reducing memory traffic. The KV cache can keep a different "
      "precision from the weights.")
    p("Serving 20 users on one 24 GB GPU", section)
    p("Assume an 8B instruction model in 4-bit weights (~5 GiB loaded), 32 layers, "
      "8 KV heads, head size 128 and BF16 cache. Cap each request at 1,536 input plus "
      "512 output tokens. Cache per token = 2 × 32 × 8 × 128 × 2 bytes = 128 KiB. "
      "For 20 full sequences, 20 × 2,048 × 128 KiB = 5 GiB. Allow another 4–6 GiB for "
      "workspace, runtime and slack. Estimated total: 14–16 GiB before load testing.")
    p("Use vLLM continuous batching and paged KV allocation. Cap active requests at 20, "
      "queue overflow and stream tokens. Configure chunked prefill so long prompts do "
      "not monopolize decode. Load-test 20 simultaneous requests and watch peak VRAM, "
      "p95 time to first token and p95 inter-token latency. Reduce active concurrency "
      "or context if the measured service target requires it.")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#BDD2D2"))
        canvas.line(56, 47, A4[0] - 56, 47)
        canvas.setFillColor(colors.HexColor(gray))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(56, 33, "AI/ML internship assignment")
        canvas.drawRightString(A4[0] - 56, 33, str(doc.page))

    doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=56,
                            rightMargin=56, topMargin=47, bottomMargin=60,
                            title="tndrr AI/ML internship assignment")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
