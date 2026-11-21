"""Train a constrained evidence selector on cloze tasks outside the RAG index."""
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE = Path(__file__).resolve().parents[1] / ".cache"
LABELS = "ABCD"


def format_prompt(question, choices):
    options = "\n".join(f"{label}: {choice}" for label, choice in zip(LABELS, choices, strict=True))
    return f"Question: {question}\nEvidence choices:\n{options}\nBest evidence:"


def main():
    random.seed(173)
    torch.manual_seed(173)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(CACHE / "base")
    model = AutoModelForCausalLM.from_pretrained(CACHE / "base")
    pages = (CACHE / "progit.txt").read_text().split("\f")[100:]
    sentences = [
        text for page in pages
        for text in re.split(r"(?<=[.!?])\s+", " ".join(page.split()))
        if 11 <= len(text.split()) <= 34 and text[0].isalpha()
    ]
    choices = [tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    assert all(len(ids) == 1 for ids in choices)
    candidate_ids = [ids[0] for ids in choices]
    rows = []
    for sentence in random.sample(sentences, min(len(sentences), 1100)):
        terms = sentence.split()
        offset = random.randrange(2, len(terms) - 4)
        masked = " ".join([*terms[:offset], "____", *terms[offset + 4:]])
        options = random.sample(sentences, 3) + [sentence]
        random.shuffle(options)
        rows.append((format_prompt(f"Which excerpt supplies the missing text? {masked}", options),
                     options.index(sentence)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(5):
        random.shuffle(rows)
        losses = []
        for offset in range(0, len(rows), 12):
            prompts, targets = zip(*rows[offset:offset + 12])
            encoded = tokenizer(list(prompts), padding=True, return_tensors="pt")
            ends = encoded.attention_mask.sum(1) - 1
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded, use_cache=False).logits
            scores = logits[torch.arange(len(prompts)), ends][:, candidate_ids]
            loss = torch.nn.functional.cross_entropy(scores, torch.tensor(targets))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(epoch + 1, len(rows), round(sum(losses) / len(losses), 4), flush=True)
    model.save_pretrained(CACHE / "evidence")
    tokenizer.save_pretrained(CACHE / "evidence")


if __name__ == "__main__":
    main()
