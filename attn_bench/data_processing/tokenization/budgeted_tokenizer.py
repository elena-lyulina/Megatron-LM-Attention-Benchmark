from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer, MegatronTokenizedFile
from datatrove.utils.batching import batched


class BudgetedMegatronDocumentTokenizer(MegatronDocumentTokenizer):
    """MegatronDocumentTokenizer that stops after a per-worker token budget is reached.

    Each worker tracks its own local counter — no IPC needed.
    Total tokens produced ≈ per_worker_budget * num_workers (within one batch of error per worker).

    Args:
        per_worker_budget (int): token budget for this worker
        See MegatronDocumentTokenizer for remaining args.
    """

    def __init__(self, per_worker_budget: int, **kwargs):
        super().__init__(**kwargs)
        self.per_worker_budget = per_worker_budget

    def write_tokens(self, data, filename: str) -> MegatronTokenizedFile:
        from tokenizers import Encoding

        unshuff = MegatronTokenizedFile(
            self.output_folder,
            filename,
            upload_block_size=self.upload_block_size,
            token_size=self.token_size,
        )
        tokens_written = 0
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = self.tokenizer.encode_batch(
                    [document.text for document in batch]
                )
                for encoded in encoded_batch:
                    tokens = encoded.ids
                    unshuff.write(tokens)
                    self.stat_update("tokens", value=len(tokens))

            tokens_written += sum(len(e.ids) for e in encoded_batch)
            if tokens_written >= self.per_worker_budget:
                break

        unshuff.close()
        return unshuff