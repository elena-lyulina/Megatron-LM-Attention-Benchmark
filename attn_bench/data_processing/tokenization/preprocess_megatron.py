"""
Copied from https://github.com/swiss-ai/data-pipeline-pretrain/blob/yxu/support-local-tokenizer-and-parquet/examples/tokenize_megatron/preprocess_megatron.py

To process HuggingFace Datasets:
    python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/emotion --n-tasks 16 hf --dataset dair-ai/emotion
To process Jsonl files:
    python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/c4-es --n-tasks 16 jsonl --dataset raw_datasets/c4-es-json-files
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from data_pipeline_pretrain.pipeline.tokens import MegatronDocumentTokenizer


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
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: <|endoftext|>",
    )
    group.add_argument(
        "--no-add-special-tokens",
        action="store_true",
        help="Do not add special tokens (BOS/EOS) during tokenization. Use this if your data already has special tokens from apply_chat_template.",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder to store the tokenized documents",
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
    # Subparsers for processing either Hugging Face datasets or jsonl files
    sp = parser.add_subparsers(
        dest="readers",
        required=True,
        description="Type of dataset to process. It can be either a Hugging Face Dataset loaded with datasets.load_data ('hf') or a .jsonl dataset ('jsonl')",
    )

    p1 = sp.add_parser(name="hf")
    p1.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or format (e.g., 'parquet', 'csv', or HuggingFace hub name like 'dair-ai/emotion')",
    )
    p1.add_argument(
        "--data-files",
        type=str,
        default=None,
        help="Path to data files (for parquet/csv formats). Default: None",
    )
    p1.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )
    p1.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split of the data to process. Default: train",
    )

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a .jsonl file or a folder containing multiple .jsonl files",
    )
    p2.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )
    p2.add_argument(
        "--glob-pattern",
        type=str,
        default=None,
        help="A glob pattern to filter files to read. Default: None",
    )

    args = parser.parse_args()

    return args


def main(args):
    # Build datatrove reader
    if args.readers == "hf":
        # Build dataset_options
        dataset_options = {"split": args.split}
        if hasattr(args, 'data_files') and args.data_files:
            dataset_options["data_files"] = args.data_files

        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options=dataset_options,
        )
    else:
        datatrove_reader = JsonlReader(
            data_folder=args.dataset,
            text_key=args.column,
            glob_pattern=args.glob_pattern,
        )

    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            datatrove_reader,
            MegatronDocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
                add_special_tokens=not args.no_add_special_tokens,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)