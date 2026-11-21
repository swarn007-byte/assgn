# Grounded answer evaluation

The retrieval score checks whether the full gold passage appears in a top-five
chunk. The answer score checks whether the displayed quote contains that passage
verbatim and cites its page. A correct paraphrase or a shorter valid quotation
can score zero under this deliberately strict answer metric.

The evidence selector is a local language model trained on cloze examples from
PDF pages 101 onward. Evaluation retrieves from pages 32–100. For each question
the model selects a label for one of up to four source quotes. The answer uses
the selected quote and its page directly.

The evaluation records three answer figures. Lexical uses the first candidate
under the lexical ordering. LLM uses the model's selection. The candidate
ceiling counts questions for which any of the supplied quotes contains the
gold passage. This shows when improving the selector alone cannot recover the
required quote. In the 160-word run the model selected the exact gold quote for
1/15 questions, lexical ordering for 3/15, and the candidate ceiling was 5/15.
The single-question mode therefore defaults to lexical ordering.

From the repository root, run:

```sh
./setup.sh
.venv/bin/python task2_rag/train_selector.py
.venv/bin/python task2_rag/evaluate_evidence.py --output .cache/evidence-dev.json
```

Try the grounded answer path:

```sh
.venv/bin/python task2_rag/evaluate_evidence.py   --ask "What counts as tracked besides files from the last snapshot?"
```

Add `--selector llm` to test the trained selector in the same mode.

## Add an independently written set

Write questions from the source text before running either ranker. For each
question, record its PDF page number and one short verbatim passage sufficient
to answer it. Use the `[question, PDF page, passage]` JSON row format in
`questions.json`. Save the new rows outside the checked-in development set.

```sh
.venv/bin/python task2_rag/evaluate_evidence.py \
  --questions .cache/my-questions.json --output .cache/evidence-holdout.json
```

Report the question count, both chunk sizes, the retrieval score, and the
answer score. Keep development and independent evaluation results separate.
