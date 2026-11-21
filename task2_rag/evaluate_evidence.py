"""Evaluate sourced answers separately from passage retrieval."""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import CACHE, HERE, Retriever, lexical_words, load_vectors, read_pages, words
from train_selector import LABELS, format_prompt


def candidate_quotes(index, question):
    question_tokens = set(lexical_words(question))
    ranked = {}
    for rank, hit in enumerate(index.search(question, k=5)):
        fragments = re.split(r"(?<=[.!?])\s+|\s+(?=\$ git )", hit["text"])
        for i, fragment in enumerate(fragments):
            for width in (1, 2):
                if i + width > len(fragments):
                    continue
                text = " ".join(fragments[i:i + width]).strip()
                size = len(text.split())
                if not 5 <= size <= 62:
                    continue
                common = question_tokens & set(lexical_words(text))
                weight = sum(index.idf.get(term, 1) for term in common)
                if weight:
                    quote = (text, hit["page"])
                    score = weight / size**0.22 / (1 + rank * 0.08)
                    ranked[quote] = max(ranked.get(quote, 0), score)
    ordered = sorted(ranked, key=ranked.get, reverse=True)
    selected = []
    for quote, page in ordered:
        if not any(p == page and (quote in old or old in quote) for old, p in selected):
            selected.append((quote, page))
            if len(selected) == len(LABELS):
                break
    return selected


class Selector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CACHE / "evidence")
        self.model = AutoModelForCausalLM.from_pretrained(CACHE / "evidence").eval()
        self.ids = [self.tokenizer.encode(" " + label, add_special_tokens=False)[0] for label in LABELS]

    @torch.inference_mode()
    def select(self, question, options):
        # Pad with empty choices and mask them out before taking the maximum.
        choices = [quote for quote, _ in options] + [""] * (len(LABELS) - len(options))
        prompt = format_prompt(question, choices)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        scores = self.model(**inputs, use_cache=False).logits[0, -1, self.ids[:len(options)]]
        return int(scores.argmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json",
                        help="JSON array of [question, PDF page, verbatim passage]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    pages = read_pages()
    questions = json.loads(args.questions.read_text())
    if not questions:
        parser.error("the question set is empty")
    for question, page, gold in questions:
        if not question.strip() or not gold.strip() or gold not in pages.get(page, ""):
            parser.error(f"invalid question or evidence on PDF page {page}: {question}")
    vocab = {token for text in [*pages.values(), *(row[0] for row in questions)]
             for token in words(text)}
    vectors = load_vectors(vocab)
    selector = Selector()
    runs = {}
    for size in (64, 160):
        index = Retriever(pages, vectors, size)
        rows = []
        for question, page, gold in questions:
            hits = index.search(question)
            options = candidate_quotes(index, question)
            rank = next((i for i, hit in enumerate(hits, 1)
                         if hit["page"] == page and gold in hit["text"]), None)
            citations = [{"quote": quote, "page": number, "exact_gold": number == page and gold in quote}
                         for quote, number in options]
            selected = selector.select(question, options) if options else None
            rows.append({
                "question": question, "gold_page": page, "gold": gold, "retrieval_rank": rank,
                "candidates": citations, "llm_answer": citations[selected] if options else None,
                "lexical_answer": citations[0] if options else None,
            })
        runs[str(size)] = {
            "retrieval_hits": sum(row["retrieval_rank"] is not None for row in rows),
            "llm_exact": sum(bool((row["llm_answer"] or {}).get("exact_gold")) for row in rows),
            "lexical_exact": sum(bool((row["lexical_answer"] or {}).get("exact_gold")) for row in rows),
            "oracle_exact": sum(any(quote["exact_gold"] for quote in row["candidates"]) for row in rows),
            "total": len(rows), "rows": rows,
        }
    args.output.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n")
    print({size: {key: value for key, value in run.items() if key != "rows"}
           for size, run in runs.items()})


if __name__ == "__main__":
    main()
