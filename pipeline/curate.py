#!/usr/bin/env python
# coding: utf-8

# # Curate the final dataset
#

import argparse
import logging
import os
import re
import pandas as pd

from dart_math.data import RespSampleVLLM
from dart_math.eval import EvaluatorMathBatch
from dart_math.utils import PROJ_HOME, init_logging, load_jsonl, save_jsonl


init_logging()


parser = argparse.ArgumentParser(
    description="Grade: extract answers & judge correctness", allow_abbrev=False
)

parser.add_argument(
    "--sample_dir",
    type=str,
    nargs="+",
    default=[os.path.join(PROJ_HOME, "data/res")],
    help="Path to save results of generation (and evaluation).",
)
parser.add_argument(
    "--fname_pattern",
    type=str,
    default=".*",
    help="Regex pattern to filter file names in `sample_dir`.",
)
parser.add_argument("--k_u", type=int, default=50, help="$k_u$ for DARS-Unifrom.")
parser.add_argument(
    "--out_dset_path",
    type=str,
    default=os.path.join(PROJ_HOME, "data/dset.jsonl"),
    help="Path to save the dataset.",
)


args, unk_args = parser.parse_known_args()


all_samples = []

for sample_dir in args.sample_dir:
    for samples_fname in os.listdir(sample_dir):
        if not re.match(args.fname_pattern, samples_fname):
            continue
        samples_fpath = os.path.join(sample_dir, samples_fname)
        samples = load_jsonl(samples_fpath)
        all_samples.extend(samples)
        logging.info(f"Loaded {len(samples)} samples from {samples_fpath=}")


logging.info(
    f"Loaded {len(all_samples)} samples in total from {args.sample_dir=} with {args.fname_pattern=}."
)


all_correct_samples = [sample for sample in all_samples if sample["correct"]]
logging.info(f"Found {len(all_correct_samples)} correct samples.")


correct_df = pd.DataFrame(all_correct_samples)


# Show the distribution of dataset
correct_df["dataset"].value_counts().plot(kind="bar")


query_grouped_correct_df = correct_df.groupby("query")
# Get the number of correct samples for each query
query_correct_count = query_grouped_correct_df.size()
query_correct_count.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


# Visualize the distribution of correct samples for each query with histogram
query_correct_count.plot(kind="hist", bins=100)


# Pick k_u correct samples for each query
chosen_correct_df = query_grouped_correct_df.head(args.k_u)


chosen_correct_df.to_json(args.out_dset_path, orient="records", lines=True)
