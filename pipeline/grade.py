#!/usr/bin/env python
# coding: utf-8

# # Grade: extract answers & judge correctness
#

import argparse
import logging
import os

from dart_math.data import RespSampleVLLM
from dart_math.eval import EvaluatorMathBatch
from dart_math.utils import PROJ_HOME, init_logging, load_jsonl, save_jsonl


init_logging()


parser = argparse.ArgumentParser(
    description="Grade: extract answers & judge correctness", allow_abbrev=False
)

parser.add_argument(
    "--gen_fpath",
    type=str,
    nargs="+",
    default=[os.path.join(PROJ_HOME, "data/res/gen.jsonl")],
    help="Path to save results of generation (and evaluation).",
)

args, unk_args = parser.parse_known_args()


evaluator = EvaluatorMathBatch()


for gen_fpath in args.gen_fpath:
    logging.info(f"Grade results in {gen_fpath=}")
    samples = [RespSampleVLLM(**sample) for sample in load_jsonl(gen_fpath)]
    evaluator.batch_eval(samples)
    save_jsonl([sample.to_dict() for sample in samples], gen_fpath)
