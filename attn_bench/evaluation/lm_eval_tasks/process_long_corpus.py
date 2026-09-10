"""process_results for the long-corpora perplexity tasks.

lm-eval 0.4.13's wikitext preprocess_wikitext.process_results with `page` -> `text`. No
detokenizer counterpart: WikiText needs one for its `@-@` markup, ours is natural text.
"""

import re


def process_results(doc, results):
    (loglikelihood,) = results
    # No .strip(), matching lm-eval: this is word_perplexity's denominator.
    _words = len(re.split(r"\s+", doc["text"]))
    _bytes = len(doc["text"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
