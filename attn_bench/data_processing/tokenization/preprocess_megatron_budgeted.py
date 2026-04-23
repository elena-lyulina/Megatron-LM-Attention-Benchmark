"""
Copied from https://github.com/swiss-ai/data-pipeline-pretrain/blob/yxu/support-local-tokenizer-and-parquet/examples/tokenize_megatron/preprocess_megatron.py
Modified to support token budgets and explicit EOS control
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from attn_bench.data_processing.tokenization.megatron_tokenizer_budgeted import (
    BudgetedMegatronDocumentTokenizer,
    MegatronDocumentTokenizer,
)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--add-bos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Add BOS token from the tokenizer before each document (default: True).",
    )
    group.add_argument(
        "--add-eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Add EOS token from the tokenizer after each document (default: True).",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder to store the tokenized documents.",
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks",
        type=int,
        default=8,
        help="Total number of tasks to run the preprocessing step. Default: 8",
    )
    group.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers executing concurrently. Default: -1 (== --n-tasks)",
    )

    group = parser.add_argument_group(title="Dataset configs")
    group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a folder recursively containing multiple .parquet files.",
    )
    group.add_argument(
        "--paths-file",
        type=str,
        required=True,
        help="A file with one path per line (without the dataset prefix) to read.",
    )
    group.add_argument("--column", type=str, default="text")

    group = parser.add_argument_group(title="Token budget")
    group.add_argument(
        "--node-budget",
        type=int,
        default=None,
        help="Token budget for this node. If not set, tokenizes everything.",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tokenization batch size — smaller = less overshoot past budget. Default: 1000",
    )

    return parser.parse_args()


def main(args):
    n_tasks = args.n_tasks
    # Check number of files > n tasks
    with open(args.paths_file, "rb") as f:
        number_of_files = sum(1 for _ in f)
    if n_tasks > number_of_files:
        n_tasks = number_of_files

    tokenizer_kwargs = dict(
        output_folder=args.output_folder,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    if args.node_budget is not None:
        tokenizer = BudgetedMegatronDocumentTokenizer(
            per_worker_budget=args.node_budget // n_tasks,
            batch_size=args.batch_size,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = MegatronDocumentTokenizer(**tokenizer_kwargs)

    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.dataset,
                paths_file=args.paths_file,
                text_key=args.column,
            ),
            tokenizer,
        ],
        tasks=n_tasks,
        workers=args.n_workers,
        logging_dir=args.logging_dir,
    ).run()


if __name__ == "__main__":
    main(get_args())