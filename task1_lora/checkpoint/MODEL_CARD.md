# Pro Git 1.3M

This is an experimental 1,311,872-parameter decoder-only Llama checkpoint trained
from random initialization by `../pretrain.py`. It is intended for the experiments
in this repository. It is small enough for CPU use and frequently generates
incorrect commands or reproduces fragments of its training text.

The corpus is from [Pro Git 2.1.450](https://github.com/progit/progit2/releases/tag/2.1.450)
by Scott Chacon and Ben Straub. The script uses PDF pages 101 onward after
`pdftotext -layout`, skipping pages with at most 25 words. Pages 32–100 are
reserved for retrieval. The model uses 4 decoder layers, 128 hidden units, 4
attention heads, 2 KV heads and a 4,096-entry byte-level BPE vocabulary.

The bundled checkpoint and LoRA adapter are available for noncommercial use
under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/),
the license used by the source text. See the original book for its authorship
and license notices.
