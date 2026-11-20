"""Page-preserving retrieval with pretrained GloVe embeddings and NumPy."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
SIZES = (64, 160)
OVERLAP = 0.25
STOP = set("a an and are as at be by can do does for from give how i in is it me of on or run that the them there to was what when which while with you".split())


def lexical_words(text):
    return [
        word if word.startswith("-") else re.sub(r"(?<=[a-z])s$", "", word)
        for word in re.findall(r"--?[a-z][a-z-]*|[a-z0-9]+", text.lower())
        if word not in STOP
    ]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def read_pages():
    text = (CACHE / "progit.txt").read_text()
    return {n: " ".join(page.split()) for n, page in
            enumerate(text.split("\f"), 1) if 32 <= n <= 100}


def load_vectors(vocab):
    vectors = {}
    with gzip.open(CACHE / "glove.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            term, _, values = line.partition(" ")
            if term in vocab:
                vectors[term] = np.fromstring(values, sep=" ", dtype=np.float32)
    return vectors


class Retriever:
    def __init__(self, pages, vectors, chunk_size):
        self.passages = []
        stride = round(chunk_size * (1 - OVERLAP))
        for page, text in pages.items():
            terms = text.split()
            for start in range(0, len(terms), stride):
                end = min(start + chunk_size, len(terms))
                self.passages.append({"page": page, "start": start, "end": end,
                                      "text": " ".join(terms[start:end])})
                if end == len(terms):
                    break
        self.vectors = vectors
        frequencies = Counter()
        for passage in self.passages:
            frequencies.update(set(words(passage["text"])))
        self.idf = {
            term: math.log((len(self.passages) + 1) / (count + 1)) + 1
            for term, count in frequencies.items()
        }
        self.matrix = np.stack([self.embed(p["text"]) for p in self.passages])
        counts = [Counter(lexical_words(p["text"])) for p in self.passages]
        lengths = np.array([sum(row.values()) for row in counts])
        norm = 1.2 * (0.25 + 0.75 * lengths / lengths.mean())
        self.postings = {}
        for i, row in enumerate(counts):
            for term, frequency in row.items():
                weight = frequency * 2.2 / (frequency + norm[i])
                self.postings.setdefault(term, []).append((i, weight))

    def embed(self, text):
        features = [self.vectors[t] * self.idf.get(t, 1.0)
                    for t in words(text) if t in self.vectors]
        vector = np.sum(features, axis=0) if features else np.zeros(50, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def search(self, question, k=5, hybrid=True):
        dense = self.matrix @ self.embed(question)
        scores = dense
        if hybrid:
            lexical = np.zeros_like(dense)
            n = len(dense)
            for term in set(lexical_words(question)):
                postings = self.postings.get(term, [])
                if postings:
                    ids, weights = zip(*postings)
                    idf = math.log1p((n - len(ids) + 0.5) / (len(ids) + 0.5))
                    lexical[list(ids)] += idf * np.array(weights)
            # Reciprocal rank fusion makes cosine and BM25 comparable.
            scores = np.zeros_like(dense)
            for values in (dense, lexical):
                ranked = np.argsort(-values, kind="stable")
                valid = ranked[values[ranked] > 0]
                scores[valid] += 1 / (10 + np.arange(1, len(valid) + 1))
        ranks = np.argsort(-scores, kind="stable")[:k]
        return [dict(self.passages[i], score=float(scores[i])) for i in ranks]


def answer(endpoint, question, passages):
    excerpts = "\n".join(f"[PDF page {p['page']}] {p['text']}" for p in passages)
    prompt = (f"Use only the excerpts. Answer briefly and cite the PDF page. "
              f"If unsupported, say you do not know.\n{excerpts}\n"
              f"Question: {question}\nAnswer:")
    payload = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 0,
                          "cache_prompt": False, "stop": ["\nQuestion:"]}).encode()
    request = Request(endpoint.rstrip("/") + "/completion", data=payload,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        generated = json.load(response)["content"].strip()

    # A tiny completion model often invents commands. Only show its text when the
    # retrieved evidence contains it verbatim; otherwise quote retrieved evidence.
    candidate = re.sub(r"\s*\[PDF page \d+\]\s*$", "", generated).strip()
    if candidate:
        for passage in passages:
            if candidate in passage["text"]:
                return {"text": candidate, "page": passage["page"], "mode": "generated",
                        "raw_generation": generated}

    terms = set(words(question)) - {
        "a", "an", "and", "are", "do", "does", "for", "git", "how", "i", "in", "is",
        "it", "of", "on", "the", "to", "what", "which", "with",
    }
    excerpts = [
        (part.strip(), passage["page"])
        for passage in passages
        for part in re.split(r"(?<=[.!?])\s+", passage["text"])
        if part.strip()
    ]
    quote, page = max(excerpts, key=lambda row: len(terms & set(words(row[0]))))
    if not terms & set(words(quote)):
        return {"text": "I could not find supporting evidence.", "page": None,
                "mode": "abstained", "raw_generation": generated}
    return {"text": quote, "page": page, "mode": "retrieved_quote", "raw_generation": generated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="llama-server URL for the generator")
    parser.add_argument("--ask", help="retrieve and answer one question")
    parser.add_argument("--size", choices=SIZES, type=int, default=160)
    args = parser.parse_args()
    if args.ask and not args.endpoint:
        parser.error("--ask requires --endpoint")

    pages = read_pages()
    questions = json.loads((HERE / "questions.json").read_text())
    assert len(pages) >= 30 and len(questions) == 15
    vocab = {token for text in [*pages.values(), *(row[0] for row in questions),
                                 args.ask or ""] for token in words(text)}
    vectors = load_vectors(vocab)
    if args.ask:
        index = Retriever(pages, vectors, args.size)
        hits = index.search(args.ask, 3)
        print(json.dumps({"answer": answer(args.endpoint, args.ask, hits), "sources": hits},
                         ensure_ascii=False, indent=2))
        return

    runs = {}
    for size in SIZES:
        index = Retriever(pages, vectors, size)
        rows = []
        for question, page, gold in questions:
            assert gold in pages[page], (page, gold)
            hits = index.search(question)
            dense = index.search(question, hybrid=False)
            dense_rank = next((i for i, hit in enumerate(dense, 1)
                               if hit["page"] == page and gold in hit["text"]), None)
            # A hit requires the annotated passage in full, on the annotated PDF page.
            rank = next((i for i, hit in enumerate(hits, 1)
                         if hit["page"] == page and gold in hit["text"]), None)
            gold_windows = sum(p["page"] == page and gold in p["text"] for p in index.passages)
            rows.append({"question": question, "gold_page": page, "gold": gold,
                         "first_gold_rank": rank, "dense_gold_rank": dense_rank, "gold_windows": gold_windows, "retrieved": hits})
        runs[str(size)] = {"hits": sum(row["first_gold_rank"] is not None for row in rows),
                           "total": len(rows), "dense_hits": sum(row["dense_gold_rank"] is not None for row in rows),
                           "chunks": len(index.passages), "rows": rows}
    if args.endpoint:
        best = max(runs.values(), key=lambda run: run["hits"])
        for row in best["rows"]:
            row["generated_answer"] = answer(args.endpoint, row["question"], row["retrieved"][:3])
    (HERE / "results.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n")
    print({size: (run["hits"], run["total"], run["chunks"]) for size, run in runs.items()})


if __name__ == "__main__":
    main()
