"""Train an openly reproducible small Llama checkpoint on Pro Git."""
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".cache/base"


def main():
    torch.manual_seed(713)
    torch.set_num_threads(4)
    pages = (ROOT / ".cache/progit.txt").read_text().split("\f")
    corpus = [" ".join(page.split()) for page in pages[100:] if len(page.split()) > 25]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=4096, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>", model_max_length=1024,
    )
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(fast), hidden_size=128, intermediate_size=384,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, tie_word_embeddings=True,
        bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    ))
    corpus_ids = torch.tensor([
        token for encoding in tokenizer.encode_batch(corpus)
        for token in [*encoding.ids, fast.eos_token_id]
    ])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    window = torch.arange(128)
    for step in range(1000):
        offsets = torch.randint(len(corpus_ids) - len(window), (12, 1))
        ids = corpus_ids[offsets + window]
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 999:
            print(step + 1, f"{loss.item():.3f}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters; {len(corpus_ids):,} corpus tokens")
    model.save_pretrained(OUTPUT)
    fast.save_pretrained(OUTPUT)


if __name__ == "__main__":
    main()
