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
    evidence = read("task2_rag/evidence_results.json")
    local = read("task3_local/results.json")
    ink, gray = "#222222", "#555555"
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                          textColor=colors.HexColor(ink), spaceAfter=8)
    section = ParagraphStyle("Section", parent=body, fontName="Helvetica-Bold", fontSize=12,
                             leading=15, spaceBefore=10, spaceAfter=5)
    title = ParagraphStyle("Title", parent=section, fontSize=25, leading=28, spaceAfter=12)
    kicker = ParagraphStyle("Kicker", parent=section, fontSize=9, leading=12,
                            textColor=colors.HexColor(gray), spaceBefore=0, spaceAfter=7)
    right = ParagraphStyle("Right", parent=body, alignment=TA_RIGHT)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def heading(number, text):
        p(f"Task {number}", kicker)
        p(text, title)

    def table(header, rows, widths):
        data = [[Paragraph(escape(str(cell)), body if i == 0 else right)
                 for i, cell in enumerate(row)] for row in [header, *rows]]
        grid = Table(data, colWidths=widths, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([grid, Spacer(1, 6)])

    heading(1, "LoRA for Git intent classification")
    p("The model predicts one of four Git actions from a plain-English request: "
      "stage, commit, branch or push. The base is a small Llama with 1.31M parameters. "
      "It was trained on 183,863 tokens from <i>Pro Git</i> PDF pages 101 onward. "
      "For each request, both checkpoints choose the label with the highest score.")
    n = lora["test_count"]
    table(["Checkpoint", "Correct", "Accuracy"], [
        (name, f"{lora[key]}/{n}", f"{lora[key] / n:.1%}")
        for name, key in [("Frozen base", "base_correct"), ("LoRA", "lora_correct")]
    ], [225, 110, 140])
    p(f"Training used {lora['train_count']} requests. The {n} test requests used different "
      "wording templates and slot values, with 15 requests per action. Requests made from "
      "one template are similar. This same test was reused after revising training, "
      "so the score is a development result.")
    p("LoRA settings", section)
    p(f"Rank {lora['rank']}, alpha {lora['alpha']}, Q/K/V/O attention projections, AdamW learning "
      f"rate {lora['learning_rate']}, {lora['epochs']} epochs and batches of 12. Rank keeps "
      "each update small. Alpha divided by rank gives a scale of 2. Updating attention "
      "helps the classifier use cues across the sentence. The learning rate limits each "
      "step and six epochs give several passes over the small training set. "
      f"The run updated {lora['trainable_parameters']:,} parameters. Training loss fell "
      f"from {lora['epoch_losses'][0]:.2f} to {lora['epoch_losses'][-1]:.3f}.")
    p("Three held-out failures", section)
    names = dict(A="stage", B="commit", C="branch", D="push")
    for i, reason in [
        (20, "Snapshot and pending changes can resemble staging when the verb <i>commit</i> is absent."),
        (36, "Independent line is an indirect cue, and the request omits <i>branch</i>."),
        (54, "The verb <i>sync</i> describes a broader operation than the explicit <i>push</i> intent."),
    ]:
        row = lora["examples"][i]
        assert row["label"] != row["lora"]
        p(f"“{escape(row['request'])}” → {names[row['lora']]} "
          f"(gold {names[row['label']]}). {reason}")
    p("7B training memory estimate", section)
    p("Assume a 7B model with 32 layers and four attention projections per layer, "
      "each 4,096 by 4,096. With rank 8, the adapters add "
      "32 × 4 × 8 × (4,096 + 4,096) ≈ 8.39M parameters.")
    p("<b>Full:</b> 7B × (2 + 2 + 8 + 4) bytes = 112 GB. Per parameter, that is "
      "2 bytes for the BF16 weight, 2 for its gradient, 8 for Adam's two FP32 values "
      "and 4 for an FP32 copy of the weight.")
    p("<b>LoRA:</b> 7B × 2 bytes for frozen weights + 8.39M × 16 bytes for trainable "
      "adapters ≈ 14.13 GB. <b>QLoRA:</b> 7B × 0.6 bytes for roughly 4-bit weights "
      "plus quantisation overhead + 8.39M × 16 bytes ≈ 4.33 GB.")
    p("Allowing another 2–6 GB for activations and temporary buffers gives roughly "
      "114–118 GB, 16–20 GB and 6–10 GB. These are estimates for a small batch and "
      "short context. The actual memory depends on the run.")

    story.append(PageBreak())
    heading(2, "Retrieval from scratch")
    p("The source is <i>Pro Git</i> 2.1.450 by Scott Chacon and Ben Straub, PDF pages "
      "32–100 (69 pages). Each page is split into chunks with 25% overlap. Each chunk "
      "gets a vector by averaging GloVe word vectors, giving rare words more weight. "
      "NumPy compares those vectors. The hybrid search also uses BM25 and rewards "
      "chunks that rank highly in either search. Recall@5 counts a hit when one of "
      "the first five chunks contains the full expected passage on the right page.")
    table(["Chunk size", "Indexed chunks", "Dense Recall@5", "Hybrid Recall@5"], [
        (f"{size} words", run["chunks"], f"{run['dense_hits']}/{run['total']}",
         f"{run['hits']}/{run['total']} ({run['hits']/run['total']:.1%})")
        for size, run in rag.items()
    ], [120, 95, 125, 135])
    p("The 160-word chunks scored better on the same 15 questions. For the ignore-pattern "
      "question, the correct chunk reached rank 2 with 160 words. A longer chunk can hold "
      "the rule and its explanation together. It also means there are fewer chunks to rank.")
    p("Two retrieval failures", section)
    p("<b>Staged diff flags:</b> The reference is on PDF page 41. The longer run returned "
      "pages 40, 40, 50, 39 and 40. The question omits the exact flags, so BM25 mainly sees "
      "generic words like ‘staged’ and ‘diff’. A command-aware query expansion could add "
      "the synonymous flags and find the matching passage.")
    p("<b>Rename via git mv:</b> The reference is on PDF page 46. The longer run returned "
      "pages 99, 45, 99, 89 and 88. These excerpts match general movement and staging terms. "
      "Boosting the exact command phrase <font name='Courier'>git mv</font> as a unit could "
      "lift passages that describe it.")
    p("How the answer is picked", section)
    modes = Counter(row["generated_answer"]["mode"] for row in rag["160"]["rows"])
    p(f"Llama gets the top three excerpts. Its answer is accepted if it matches text "
      f"from a source excerpt. All {modes['retrieved_quote']} questions used the fallback, "
      "which quotes a source using word overlap and shows the PDF page. For example, "
      "the amend question returned a quote from page 53 about replacing the old commit.")
    long = evidence["160"]
    p(f"In a separate test, a Llama trained to fill in missing text chose among four quotes. "
      f"It picked the full expected quote {long['llm_exact']}/15 times, versus "
      f"{long['lexical_exact']}/15 for word matching. Only {long['oracle_exact']}/15 questions "
      "had the full expected quote among the choices. Both tests reused the development questions.")
    p("BM25 and re-ranking", section)
    p("BM25 rewards exact words, especially rare ones. That helps with command names "
      "and flags. It also accounts for repetition and chunk length. A re-ranker takes "
      "the first search results and scores them again against the question. It can move "
      "a better result up the list if the first search found it already.")
    p("The repository holds the 15 questions, expected passages and measured ranks.")

    story.append(PageBreak())
    heading(3, "Local inference and quantisation")
    p("The same Git command model ran in llama.cpp b12359 on an AMD EPYC 9V74 CPU. "
      "Both formats used four threads, a 1,024-token context and greedy decoding. "
      "Speed is the median of 30 measurements per format, with 32 output tokens per "
      "measurement and prompt processing excluded. For quality, both formats answered "
      "the same 10 development prompts and could stop early. A hit means the command "
      "and its arguments matched. Peak RSS is the most RAM used by the server process.")
    table(["Setting", "Decode tok/s", "Peak RSS MiB", "GGUF MiB", "Command hits"], [
        (key.upper(), f"{run['median_decode_tokens_per_second']:,.0f}",
         f"{run['peak_rss_kib']/1024:.2f}", f"{run['file_bytes']/2**20:.2f}",
         f"{sum(row['contains_expected'] for row in run['samples'])}/{len(run['samples'])}")
        for key, run in local.items()
    ], [85, 102, 99, 92, 97])
    matched = sum(a["output"] == b["output"]
                  for a, b in zip(local["f16"]["samples"], local["q4_0"]["samples"]))
    p(f"Quantisation reduced the GGUF size. {matched}/10 completions were byte-identical "
      "across formats. The examples below show how the score treats commands and operands. "
      "The complete ten-prompt comparison is in the repository.")
    divergent = [(a, b) for a, b in zip(local["f16"]["samples"], local["q4_0"]["samples"])
                 if a["output"] != b["output"]]
    paired = list(zip(local["f16"]["samples"], local["q4_0"]["samples"]))
    if divergent:
        examples = divergent[:2]
    else:
        examples = [next((pair for pair in paired if pair[0]["contains_expected"]), paired[0]),
                    next((pair for pair in paired if not pair[0]["contains_expected"]), paired[-1])]
    for a, b in examples:
        left = escape(a["output"].strip()[:105])
        right_output = escape(b["output"].strip()[:105])
        expected = escape(a["accepted_commands"][0])
        p(f"“{escape(a['prompt'])}”<br/>Expected <font name='Courier'>{expected}</font>. "
          f"F16 <font name='Courier'>{left}</font>. Q4_0 <font name='Courier'>{right_output}</font>.")
    p("KV cache and weight quantisation", section)
    p("The KV cache stores the keys and values for tokens the model has already processed. "
      "When it adds a token, it reuses those values for attention. Longer inputs and more "
      "open requests need more cache. Quantisation uses fewer bits to store model weights. "
      "That reduces file size and weight reads. The cache can still use a different precision.")
    p("Serving 20 users on one 24 GB GPU", section)
    p("Assume an 8B instruction model in 4-bit weights (~5 GiB loaded), 32 layers, "
      "8 KV heads, head size 128 and BF16 cache. Cap each request at 1,536 input plus "
      "512 output tokens. Cache per token = 2 × 32 × 8 × 128 × 2 bytes = 128 KiB. "
      "For 20 full sequences, 20 × 2,048 × 128 KiB = 5 GiB. Allow another 4–6 GiB for "
      "workspace, runtime and slack. Estimated total: 14–16 GiB before load testing.")
    p("Use vLLM to group active requests and allocate cache as needed. Let up to 20 "
      "requests run, queue the rest and stream replies. Test 20 simultaneous requests "
      "to measure peak GPU memory, time to first token and time between output tokens.")

    p("Reproduce locally with <font name='Courier'>./setup.sh</font> and "
      "<font name='Courier'>./task3_local/run.sh</font>.")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
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
