"""Cut raw pretraining documents into the QAD recipe's sequence geometry.

All three QAD losses -- task CE, EAKLD on logits, LAFD on features -- are masked
to assistant answer tokens only, so the recipe's gradient geometry is "an
unsupervised prefix, then a shorter supervised target".  Raw text has no
assistant span to map that onto, so every corpus built from pretraining data has
to manufacture one the same way.

This lives here rather than in one builder script because the whole point of a
corpus swap is that geometry is *not* one of the variables.  Two builders that
each carry their own copy of the cut would drift, and a drifted window is
indistinguishable in the results from a corpus effect.
"""

from __future__ import annotations

import random
from typing import Any

# Floor on the supervised span: below this the gradient signal per sequence gets
# too thin to be worth the forward pass.
MIN_TARGET_TOKENS = 32


def window(
    text: str,
    tokenizer: Any,
    rng: random.Random,
    *,
    max_tokens: int,
    target_tokens: int,
    target_fraction: float,
    min_tokens: int,
    char_budget: int,
) -> tuple[str, str] | None:
    """Cut one context/target window, biased away from document openings.

    Always slicing from character zero would train exclusively on introductions,
    which in a corpus this heavy on arXiv means training on LaTeX preambles.  A
    random start costs nothing and spreads the windows over each document.

    The target length is a *fraction* of what the document actually supplies
    rather than a fixed 256 tokens.  Demanding a full 768+256 window would drop
    four out of five C4 documents, whose median length is 1103 characters, and
    the surviving mixture would collapse onto arXiv and CommonCrawl -- changing
    the corpus composition at the same time as the corpus, which defeats the
    point.  A fraction keeps the recipe's roughly 1:3 target-to-context ratio,
    while the 256-token cap prevents unusually long targets.
    """
    if len(text) < 200:
        return None
    # Tokenizing a 50k-character arXiv paper to keep 1024 tokens is most of the
    # runtime, so slice on characters first at roughly 3.5 chars per token.
    start = 0
    if len(text) > char_budget:
        start = rng.randrange(0, len(text) - char_budget)
    ids = tokenizer(text[start : start + char_budget], add_special_tokens=False)[
        "input_ids"
    ]
    ids = ids[:max_tokens]
    if len(ids) < min_tokens:
        return None
    cut = min(target_tokens, max(MIN_TARGET_TOKENS, int(len(ids) * target_fraction)))
    head = tokenizer.decode(ids[:-cut])
    # The target must not begin with whitespace.  Qwen3's chat template renders
    # the assistant turn as "...<think>\n\n</think>\n\n" + content.lstrip('\n'),
    # which strips leading newlines but not leading spaces.  A target starting
    # with an indent therefore puts "\n\n" next to "    " at the seam, and the
    # BPE merges across it -- newline-plus-indent is one of the most common
    # tokens in a code-heavy corpus.  The collator's prompt-prefix check then
    # fails and takes the whole distributed job down with it.  Four of 558,000
    # documents hit this, all GitHub or StackExchange code.
    tail = tokenizer.decode(ids[-cut:]).lstrip()
    if not head.strip() or not tail:
        return None
    return head, tail
