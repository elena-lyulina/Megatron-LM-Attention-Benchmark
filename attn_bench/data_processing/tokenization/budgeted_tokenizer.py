import multiprocessing
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer, MegatronTokenizedFile
from datatrove.utils.batching import batched


class SharedBudget:
    """Picklable shared token budget for use across multiprocess workers."""

    def __init__(self, node_budget: int):
        manager = multiprocessing.Manager()
        #  need to pass both the value and the lock, see https://github.com/python/cpython/issues/79967
        self._counter = manager.Value('q', 0)
        self._lock = manager.Lock()
        self.node_budget = node_budget

    def add_and_check(self, tokens: int) -> bool:
        """Atomically add tokens and return True if budget is reached."""
        with self._lock:
            self._counter.value += tokens
            return self._counter.value >= self.node_budget


class BudgetedMegatronDocumentTokenizer(MegatronDocumentTokenizer):
    """MegatronDocumentTokenizer that stops all workers once a shared token budget is reached.

    Args:
        budget (SharedBudget): shared budget object across all workers
        See MegatronDocumentTokenizer for remaining args.
    """

    def __init__(self, budget: SharedBudget, **kwargs):
        super().__init__(**kwargs)
        self.budget = budget

    def write_tokens(self, data, filename: str) -> MegatronTokenizedFile:
        from tokenizers import Encoding

        unshuff = MegatronTokenizedFile(
            self.output_folder,
            filename,
            upload_block_size=self.upload_block_size,
            token_size=self.token_size,
        )
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = self.tokenizer.encode_batch(
                    [document.text for document in batch]
                )
                for encoded in encoded_batch:
                    tokens = encoded.ids
                    unshuff.write(tokens)
                    self.stat_update("tokens", value=len(tokens))

            if self.budget.add_and_check(sum(len(e.ids) for e in encoded_batch)):
                break

        unshuff.close()
        return unshuff