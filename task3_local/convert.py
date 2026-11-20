"""Export the locally trained Llama and its GPT-2-style tokenizer to GGUF."""
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
    ByteLevelLlama(CACHE / "base", gguf.LlamaFileType.MOSTLY_F16,
                   CACHE / "progit-f16.gguf", model_name="Pro Git 1.3M").write()
