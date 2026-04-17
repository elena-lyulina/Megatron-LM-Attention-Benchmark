import multiprocessing
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer, MegatronTokenizedFile
from datatrove.utils.batching import batched


class BudgetedMegatronDocumentTokenizer(MegatronDocumentTokenizer):
    """MegatronDocumentTokenizer that stops all workers once a shared token budget is reached.

    Args:
        shared_counter (multiprocessing.Value): shared 'q' counter incremented by all workers
        node_budget (int): total tokens this node should produce before stopping
        See MegatronDocumentTokenizer for remaining args.
    """

    def __init__(self, shared_counter: multiprocessing.Value, node_budget: int, **kwargs):
        super().__init__(**kwargs)
        self.shared_counter = shared_counter
        self.node_budget = node_budget

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

            with self.shared_counter.get_lock():
                self.shared_counter.value += sum(len(e.ids) for e in encoded_batch)
                budget_reached = self.shared_counter.value >= self.node_budget

            if budget_reached:
                break

        unshuff.close()
        return unshuff