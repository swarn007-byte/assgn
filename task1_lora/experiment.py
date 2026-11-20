"""Evaluate the same four-way candidate scoring before and after LoRA."""
import json
from pathlib import Path
import random

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".cache/base"
RESULT = Path(__file__).with_name("results.json")
SEED = 713
RANK, ALPHA, LR, EPOCHS = 8, 16, 0.003, 8
TARGETS = ["q_proj", "v_proj"]
CATEGORIES = {
    "A": ("stage", [
        "Stage {file} for my next commit.", "Add {file} to the Git index.",
        "Include changes to {file} in the staging area.", "Prepare {file} to be committed.",
        "Tell Git to track {file} for the next snapshot.", "Run git add on {file}.",
    ], [
        "Put the latest edits to {file} in the index.",
        "Queue {file} up for the snapshot without making the snapshot yet.",
        "Can we include {file} in what will be saved next?",
    ]),
    "B": ("commit", [
        "Commit staged changes with the message {message}.", "Save the index as a commit: {message}.",
        "Record a snapshot labeled {message}.", "Create a commit called {message}.",
        "Put the staged work into Git history with the note {message}.", "Run git commit with {message}.",
    ], [
        "Write the prepared changes to history and describe them as {message}.",
        "Turn the pending snapshot into a permanent revision titled {message}.",
        "I want the staged work saved under the message {message}.",
    ]),
    "C": ("branch", [
        "Create a branch named {branch}.", "Make a new Git branch: {branch}.",
        "Add the branch {branch} at HEAD.", "Start a new line of history called {branch}.",
        "Create and switch to {branch}.", "Run git branch {branch}.",
    ], [
        "Give me a new development line named {branch}.",
        "Set up {branch} as an independent line starting from here.",
        "Open {branch} so I can keep this work separate.",
    ]),
    "D": ("push", [
        "Push the current branch to {remote}.", "Upload my commits to {remote}.",
        "Publish the local branch on {remote}.", "Send my Git commits to {remote}.",
        "Update the remote {remote} from my local branch.", "Run git push to {remote}.",
    ], [
        "Make the revisions I just saved available on {remote}.",
        "Sync my outgoing changes to the server named {remote}.",
        "Forward what I have locally to the shared repo at {remote}.",
    ]),
}
TRAIN = dict(file=["readme.md", "index.html", "server.py", "main.go", "notes.txt", "Makefile"],
             message=["fix parser", "update docs", "first draft", "add flag", "clean imports", "bump build"],
             branch=["fix-nav", "release", "docs", "trial", "api", "work"],
             remote=["origin", "fork", "central", "mirror", "team", "upstream"])
TEST = dict(file=["settings.toml", "layout.css", "Dockerfile", "report.md", "test_cli.py"],
            message=["handle retries", "tighten scope", "correct totals", "polish help", "remove dead paths"],
            branch=["receipt", "layout-v2", "perf-pass", "tooling", "search"],
            remote=["lab", "backup", "company", "review", "archive"])


def data(held_out):
    rows = []
    for label, (_, training, test) in CATEGORIES.items():
        slots = TEST if held_out else TRAIN
        templates = test if held_out else training
        for template in templates:
            key = template.split("{")[1].split("}")[0]
            rows.extend((template.format(**{key: value}), label) for value in slots[key])
    return rows


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE)
    candidate_ids = [tokenizer.encode(" " + label, add_special_tokens=False) for label in CATEGORIES]
    assert all(len(ids) == 1 for ids in candidate_ids), candidate_ids
    candidate_ids = [ids[0] for ids in candidate_ids]
    assert len(set(candidate_ids)) == len(CATEGORIES)
    train, test = data(False), data(True)
    assert len(test) >= 50 and not {x for x, _ in train} & {x for x, _ in test}

    def score(model, rows):
        prompts = [
            "Classify the Git request. A = stage; B = commit; C = branch; D = push.\n"
            f"Request: {request}\nAnswer:" for request, _ in rows
        ]
        batch = tokenizer(prompts, padding=True, return_tensors="pt")
        end = batch.attention_mask.sum(1) - 1
        logits = model(**batch, use_cache=False).logits
        return logits[torch.arange(len(rows)), end][:, candidate_ids]

    @torch.inference_mode()
    def predict(model):
        model.eval()
        predictions = []
        for i in range(0, len(test), 12):
            predictions.extend(score(model, test[i:i + 12]).argmax(1).tolist())
        return [list(CATEGORIES)[i] for i in predictions]

    before = predict(base)
    adapted = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=RANK, lora_alpha=ALPHA,
        target_modules=TARGETS,
    ))
    trained, total = adapted.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in adapted.parameters() if p.requires_grad), lr=LR)
    losses = []
    adapted.train()
    for _ in range(EPOCHS):
        random.shuffle(train)
        epoch_losses = []
        for i in range(0, len(train), 12):
            rows = train[i:i + 12]
            labels = torch.tensor([list(CATEGORIES).index(label) for _, label in rows])
            loss = torch.nn.functional.cross_entropy(score(adapted, rows), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(sum(epoch_losses) / len(epoch_losses))
    after = predict(adapted)
    adapted.save_pretrained(ROOT / ".cache/adapter")
    results = {
        "seed": SEED, "train_count": len(train), "test_count": len(test),
        "trainable_parameters": trained, "total_parameters": total,
        "rank": RANK, "alpha": ALPHA, "learning_rate": LR, "epochs": EPOCHS,
        "targets": TARGETS, "epoch_losses": losses,
        "base_correct": sum(p == label for (_, label), p in zip(test, before)),
        "lora_correct": sum(p == label for (_, label), p in zip(test, after)),
        "examples": [
            {"request": request, "label": label, "base": p, "lora": q}
            for (request, label), p, q in zip(test, before, after)
        ],
    }
    RESULT.write_text(json.dumps(results, indent=2) + "\n")
    print({k: v for k, v in results.items() if k != "examples"})


if __name__ == "__main__":
    main()
