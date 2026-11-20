"""Teach the local checkpoint to emit one command from a Git request."""
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE = Path(__file__).resolve().parents[1] / ".cache"
COMMANDS = [
    ("start a fresh Git repository", "git init"),
    ("inspect the working tree state", "git status"),
    ("stage {file} for the next snapshot", "git add {file}"),
    ("inspect past commits", "git log"),
    ("make a branch called {branch} and stay on this branch", "git branch {branch}"),
    ("shelve outstanding edits for later", "git stash"),
    ("retrieve commits from {remote} without integrating them", "git fetch {remote}"),
    ("inspect staged changes relative to HEAD", "git diff --cached"),
    ("show configured remotes and addresses", "git remote -v"),
    ("record indexed changes under the message {message}", 'git commit -m "{message}"'),
    ("switch to an existing branch named {branch}", "git switch {branch}"),
    ("show the unstaged patch", "git diff"),
    ("display a compact commit history", "git log --oneline"),
    ("list all local branches", "git branch"),
    ("display the current commit", "git show HEAD"),
    ("show names of configured remotes", "git remote"),
    ("upload commits to {remote}", "git push {remote}"),
    ("clone a repository from {remote}", "git clone {remote}"),
    ("integrate changes from a branch called {branch}", "git merge {branch}"),
    ("list stashed changes", "git stash list"),
    ("restore the most recent stash and drop it", "git stash pop"),
    ("make an annotated tag called {branch}", 'git tag -a {branch} -m "release"'),
]
PARAPHRASES = {
    "git init": ["initialize version control here", "turn this directory into a new repository",
                 "set up an empty repo", "begin tracking this project with Git"],
    "git status": ["see what has been modified", "check for changed or untracked files",
                   "see the current status of the repo", "check the index and working directory"],
    "git add {file}": ["put {file} in the index", "prepare {file} to commit", "add {file} to staging",
                       "include {file} in the next commit"],
    "git log": ["see the repository history", "list the recorded commits",
                "show previous revisions in full", "browse the commit log"],
    "git branch {branch}": ["create the local branch {branch} without checking it out",
                            "create a Git branch named {branch}", "add a new reference named {branch}",
                            "start a separate branch called {branch} without changing HEAD"],
    "git stash": ["put uncommitted work aside for now", "save my local edits temporarily",
                  "set aside modifications and clean the working tree", "shelve changes without committing"],
    "git fetch {remote}": ["download remote commits from {remote}", "retrieve updated refs from {remote}",
                           "get the newest changes from {remote} but don't merge them",
                           "fetch the remote named {remote} without touching local edits"],
    "git diff --cached": ["display what is going into the next commit", "view the staged patch",
                          "compare the index with the committed snapshot", "show the diff of staged changes"],
    "git remote -v": ["list fetch and push URLs", "show remote names together with URLs",
                      "print the configured remote addresses", "view remotes in verbose mode"],
    'git commit -m "{message}"': ["create a commit with the message {message}",
                                 "commit the staged work and call it {message}",
                                 "save the contents of the index with the note {message}",
                                 "write a revision named {message} from staged edits"],
}
TEMPLATES = [
    "Git request: {goal}\nCommand:", "Task: {goal}\nGit command:",
    "Action to perform: {goal}\nAnswer:", "Which command would you use to {goal}?\nCommand:",
    "I want to {goal}. Give one Git command.\nCommand:",
    "To {goal}, run", "Goal: {goal}.\nShell:",
]
OPTIONS = [
    dict(file="main.py", branch="topic", remote="upstream", message="Revise docs"),
    dict(file="index.html", branch="experiment", remote="mirror", message="Add flags"),
    dict(file="notes.txt", branch="release", remote="team", message="Refresh cache"),
    dict(file="config.yml", branch="cleanup", remote="fork", message="Clean imports"),
    dict(file="app.js", branch="feature", remote="origin", message="Update tests"),
    dict(file="build.sh", branch="sandbox", remote="backup", message="Handle errors"),
    dict(file="CONTRIBUTING.md", branch="compat", remote="central", message="Fix links"),
    dict(file="ARCHITECTURE.md", branch="notes", remote="server", message="Add diagrams"),
    dict(file="TODO.md", branch="wip", remote="studio", message="Write comments"),
    dict(file="lib/parser.py", branch="parser", remote="review", message="Improve parsing"),
]


def main():
    random.seed(281)
    torch.manual_seed(281)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(CACHE / "base")
    model = AutoModelForCausalLM.from_pretrained(CACHE / "base")
    rows = []
    examples = COMMANDS + [(goal, command) for command, goals in PARAPHRASES.items() for goal in goals]
    for goal, command in examples:
        for option in OPTIONS if "{" in command else OPTIONS[:1]:
            goal_text, command_text = goal.format(**option), command.format(**option)
            for template in TEMPLATES:
                prefix = tokenizer.encode(template.format(goal=goal_text))
                suffix = tokenizer.encode(" " + command_text) + [tokenizer.eos_token_id]
                rows.append((prefix + suffix, [-100] * len(prefix) + suffix))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015)
    model.train()
    for epoch in range(12):
        random.shuffle(rows)
        losses = []
        for i in range(0, len(rows), 24):
            batch = rows[i:i + 24]
            width = max(len(tokens) for tokens, _ in batch)
            ids = torch.tensor([tokens + [tokenizer.pad_token_id] * (width - len(tokens))
                                for tokens, _ in batch])
            labels = torch.tensor([labels + [-100] * (width - len(labels)) for _, labels in batch])
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(epoch + 1, len(rows), round(sum(losses) / len(losses), 4), flush=True)
    model.save_pretrained(CACHE / "commands")
    tokenizer.save_pretrained(CACHE / "commands")


if __name__ == "__main__":
    main()
