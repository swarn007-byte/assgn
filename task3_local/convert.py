"""Export the locally trained Llama and its GPT-2-style tokenizer to GGUF."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
SOURCE = CACHE / "llama.cpp-b12359"
sys.path[:0] = [str(SOURCE), str(SOURCE / "gguf-py")]

import gguf
from conversion.llama import LlamaModel


class ByteLevelLlama(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def get_vocab_base_pre(self, tokenizer):
        # The tokenizer was trained with tokenizers.ByteLevel's default GPT-2 regex.
        return "gpt-2"

    def set_vocab(self):
        self._set_vocab_gpt2()
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_space_prefix(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", action="store_true")
    args = parser.parse_args()
    source = "commands" if args.commands else "base"
    output = "commands-f16.gguf" if args.commands else "progit-f16.gguf"
    ByteLevelLlama(CACHE / source, gguf.LlamaFileType.MOSTLY_F16,
                   CACHE / output, model_name=f"Pro Git 1.3M {source}").write()
