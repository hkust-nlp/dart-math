#!/usr/bin/env python
# coding: utf-8

# # Generation
#
# > Generate with specified stopping criteria

import argparse
import logging
import os
import sys
import time

from vllm import LLM, SamplingParams

from dart_math.data import RespSampleVLLM, load_query_dps
from dart_math.eval import EvaluatorMathBatch
from dart_math.gen import gen, get_prompt_template4model, is_dp_dars_finished
from dart_math.utils import PromptTemplate, get_pathname_from_name_or_path, init_logging

init_logging()


parser = argparse.ArgumentParser(description="vLLM generation", allow_abbrev=False)

parser.add_argument(
    "--gen_save_path",
    type=str,
    required=True,
    help="Path save results of generation (and evaluation).",
)

# Device
parser.add_argument(
    "--gpu_mem_util",
    type=float,
    default=0.85,
    help="GPU memory utilization for vLLM. Default: 0.85 in case of OOM.",
)

parser.add_argument(
    "--swap_space", type=float, default=60, help="CPU swap space in GB for vLLM."
)

# Model
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="mistralai/Mistral-7B-v0.1",
    help="HF-style model name or path.",
)

parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    help="Data type for the model.",
)

# Data
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=["math"],
    help="Dataset(s) for evaluation.",
)

# Generation configurations
parser.add_argument(
    "--temperature",
    type=float,
    default=0,
    help="Temperature for sampling.",
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
    help="Top-p for sampling.",
)

parser.add_argument(
    "--max_new_toks",
    type=int,
    default=2048,
    help="Maximum number of new tokens.",
)

parser.add_argument(
    "--n_shots",
    type=int,
    default=-1,
    help="Number of shots for prompting. -1 means adaptive to datasets.",
)

parser.add_argument(
    "--prompt_template",
    type=str,
    default="auto",
    help="ID / Path to the file of prompt template.",
)

parser.add_argument(
    "--n_paths",
    type=int,
    default=1,
    help="Number of generated completions per request. NOTE: might cause bug in vLLM (0.4.2).",
)

parser.add_argument(
    "--save_gen_path_bs",
    type=int,
    default=2**14,
    help="# Completions = # Paths per request * # Requests. Values <= 0 mean adaptive.",
)

parser.add_argument(
    "--inf_seed",
    type=int,
    default=0,
    help="Random seed for inference. -1 means using us timestamp mod 2^32.",
)

# Stopping criteria
parser.add_argument(
    "--max_n_trials",
    nargs="+",
    type=int,
    default=1,
    help="(List of) maximum number of trials for each query. Non-positive means no limit.",
)
parser.add_argument(
    "--do_eval",
    action="store_true",
    help="Whether to evaluate the generated completions.",
)
parser.add_argument(
    "--min_n_corrects",
    nargs="+",
    type=int,
    default=0,
    help="(List of) minimum number of correct completions per query needed to stop generation. Non-positive means no goal.",
)

args, unk_args = parser.parse_known_args(sys.argv)


if args.inf_seed == -1:
    args.inf_seed = int(time.time() * 10**6) % 2**32


model_dirname = get_pathname_from_name_or_path(args.model_name_or_path)


prompt_template = (
    get_prompt_template4model(args.model_name_or_path)
    if args.prompt_template == "auto"
    else PromptTemplate.load_from_id_or_path(args.prompt_template)
)


if args.temperature <= 1e-5:
    args.temperature = 0
    args.n_paths = 1
    args.top_p = 1
    logging.warning(
        f"Temperature is too small. Setting temperautre = 0, n_paths = 1, top_p = 1 for vLLM."
    )

sampling_params = SamplingParams(
    n=args.n_paths,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_new_toks,
    stop=[prompt_template.query_prompt.strip(), prompt_template.resp_prompt.strip()],
    skip_special_tokens=True,
    seed=args.inf_seed,
)

print(f"sampling_params = {sampling_params}")


query_dps = load_query_dps(args.datasets, args.max_n_trials, args.min_n_corrects)
for query_dp in query_dps:
    query_dp.prompt_template = prompt_template


os.environ["TOKENIZERS_PARALLELISM"] = "false"


llm = LLM(
    model=args.model_name_or_path,
    tokenizer=args.model_name_or_path,
    tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
    dtype=args.dtype,
    seed=args.inf_seed,
    gpu_memory_utilization=args.gpu_mem_util,
    swap_space=args.swap_space,
    trust_remote_code=True,
)
logging.info("LLM loaded!")


gen(
    llm,
    sampling_params,
    query_dps=query_dps,
    dp_stop_criteria=is_dp_dars_finished,
    resp_sample_cls=RespSampleVLLM,
    batch_evaluator=(EvaluatorMathBatch() if args.do_eval else None),
    save_path=args.gen_save_path,
    n_paths_per_save=args.save_gen_path_bs,
)


logging.info("Generation done!")
