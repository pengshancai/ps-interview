import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorForSeq2Seq,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, LlamaForCausalLM
import pickle
from transformers import pipeline, set_seed

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default=None,
#         help="The name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--dataset_config_name",
#         type=str,
#         default=None,
#         help="The configuration name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
#     )
#     parser.add_argument(
#         "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
#     )
#     parser.add_argument(
#         "--validation_split_percentage",
#         default=5,
#         help="The percentage of the train set used as validation set in case there's no validation split",
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         type=str,
#         help="Path to pretrained model or model identifier from huggingface.co/models.",
#         required=False,
#     )
#     parser.add_argument(
#         "--config_name",
#         type=str,
#         default=None,
#         help="Pretrained config name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--tokenizer_name",
#         type=str,
#         default=None,
#         help="Pretrained tokenizer name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--use_slow_tokenizer",
#         action="store_true",
#         help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
#     )
#     parser.add_argument(
#         "--per_device_train_batch_size",
#         type=int,
#         default=8,
#         help="Batch size (per device) for the training dataloader.",
#     )
#     parser.add_argument(
#         "--per_device_eval_batch_size",
#         type=int,
#         default=8,
#         help="Batch size (per device) for the evaluation dataloader.",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=5e-5,
#         help="Initial learning rate (after the potential warmup period) to use.",
#     )
#     parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
#     parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
#     parser.add_argument(
#         "--max_train_steps",
#         type=int,
#         default=None,
#         help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
#     )
#     parser.add_argument(
#         "--gradient_accumulation_steps",
#         type=int,
#         default=1,
#         help="Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument(
#         "--lr_scheduler_type",
#         type=SchedulerType,
#         default="linear",
#         help="The scheduler type to use.",
#         choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
#     )
#     parser.add_argument(
#         "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
#     )
#     parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
#     parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default=None,
#         help="Model type to use if training from scratch.",
#         choices=MODEL_TYPES,
#     )
#     parser.add_argument(
#         "--block_size",
#         type=int,
#         default=None,
#         help=(
#             "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
#             " this size for training. Default to the model max input length for single sentence inputs (take into"
#             " account special tokens)."
#         ),
#     )
#     parser.add_argument(
#         "--preprocessing_num_workers",
#         type=int,
#         default=None,
#         help="The number of processes to use for the preprocessing.",
#     )
#     parser.add_argument(
#         "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
#     )
#     parser.add_argument(
#         "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
#     )
#     parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
#     parser.add_argument(
#         "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
#     )
#     parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
#     parser.add_argument(
#         "--checkpointing_steps",
#         type=str,
#         default=None,
#         help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
#     )
#     parser.add_argument(
#         "--resume_from_checkpoint",
#         type=str,
#         default=None,
#         help="If the training should continue from a checkpoint folder.",
#     )
#     parser.add_argument(
#         "--with_tracking",
#         action="store_true",
#         help="Whether to enable experiment trackers for logging.",
#     )
#     parser.add_argument(
#         "--report_to",
#         type=str,
#         default="all",
#         help=(
#             'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
#             ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
#             "Only applicable when `--with_tracking` is passed."
#         ),
#     )
#     parser.add_argument(
#         "--low_cpu_mem_usage",
#         action="store_true",
#         help=(
#             "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
#             "If passed, LLM loading time and RAM consumption will be benefited."
#         ),
#     )
#     args = parser.parse_args()
#
#
#     # Sanity checks
#     if args.dataset_name is None and args.train_file is None and args.validation_file is None:
#         raise ValueError("Need either a dataset name or a training/validation file.")
#     else:
#         if args.train_file is not None:
#             extension = args.train_file.split(".")[-1]
#             assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
#         if args.validation_file is not None:
#             extension = args.validation_file.split(".")[-1]
#             assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
#
#     if args.push_to_hub:
#         assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
#
#     return args
#
#
# def main():
#     args = parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

output_dir = '../ps-interview_/dump/gpt2_1.0/'
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.to(device)

data_path = '../ps-interview_/data/processed_clm/test.json'
data_files = {'test': data_path}
extension = data_path.split(".")[-1]
dataset = load_dataset(extension, data_files=data_files)




generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

for rec in dataset['test']:
    src, tgt = rec['text'].split("Summarize the customer\'s issue in the above dialog in one sentence.")
    inputs = tokenizer(src + "Summarize the customer\'s issue in the above dialog in one sentence.", return_tensors='pt').to(device)
    x = model.generate(**inputs, max_length=1024)
    y = tokenizer.decode(x.cpu().numpy()[0])
    print(y)
    break
    # x = generator(src, max_length=1024, num_return_sequences=5)
    #
