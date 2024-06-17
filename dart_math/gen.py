import logging
import os
from typing import Callable

import orjson
from vllm import LLM, SamplingParams

from .data import DS_ID2N_SHOTS, ICL_EGS, QueryDataPoint, RespSampleBase, RespSampleVLLM
from .eval import EvaluatorBatchBase
from .utils import PromptTemplate, get_pathname_from_name_or_path

BASE_MODEL_IDS = [
    "deepseek-ai--deepseek-math-7b-base",
    "mistralai--Mistral-7B-v0.1",
    "meta-llama--Llama-2-7b-hf",
    "meta-llama--Llama-2-13b-hf",
    "meta-llama--Llama-2-70b-hf",
    "meta-llama--Meta-Llama-3-8B",
    "meta-llama--Meta-Llama-3-70B",
    "EleutherAI--llemma_7b",
    "EleutherAI--llemma_34b",
    "QWen--QWen-1.5-72B",
]

DEEPSEEK_INSTR_MODEL_IDS = [
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/deepseek-math-7b-rl",
]

MATH_SHEPHERD_MODEL_IDS = [
    "peiyi9979/mistral-7b-sft",
    "peiyi9979/math-shepherd-mistral-7b-rl",
]

# Data Preprocessing


# %% ../nbs/02_gen.ipynb 0
def get_prompt_template4model(
    model_name_or_path: str,  # HF ID or path to the model.
) -> PromptTemplate:
    # Get the prompt template suitable for the model.
    if model_name_or_path in BASE_MODEL_IDS + MATH_SHEPHERD_MODEL_IDS:
        prompt_template = "qa"
    elif model_name_or_path.startswith("dart-math"):
        prompt_template = "alpaca"
    elif model_name_or_path in DEEPSEEK_INSTR_MODEL_IDS:
        prompt_template = "deepseekmath"
    elif model_name_or_path.startswith("Xwin-LM/Xwin-Math"):
        prompt_template = "xwinmath"
    elif model_name_or_path.startswith("TIGER-Lab--MAmmoTH2"):
        prompt_template = "mammoth2-cot"
    else:  # default
        prompt_template = "alpaca"

    prompt_template = PromptTemplate.load_from_id_or_path(prompt_template)
    if "MMIQC" in model_name_or_path:
        prompt_template.prompt_before_resp = (
            'Please solve the following problem and put your answer at the end with "The answer is: ".'
            + " "
        )

    return prompt_template


def get_n_shots(
    dataset: str,
    model: str,
) -> int:
    """Get the number of ICL examples adaptive to the dataset and model."""
    if model in BASE_MODEL_IDS or "mammoth2" in model.lower():
        n_shots = DS_ID2N_SHOTS.get(dataset)
    else:
        n_shots = 0

    return n_shots


def get_icl_egs(
    dataset: str, n_shots: int = None, model: str | None = None
) -> list[tuple[str, str]]:
    """Get the ICL examples for the dataset.

    Parameters
    ----------
    dataset : str
        Preset dataset ID.
    n_shots : int, default: None
        Number of examples in the few-shot prompt. `None` / Negative means adaptive to the datasets.
    model : str | None, default: None
        HF ID or path to the model.

    Returns
    -------
    list[tuple[str, str]]
        ICL examples adaptive to the dataset (and model).
    """
    n_shots = get_n_shots(dataset, model) if n_shots is None or n_shots < 0 else n_shots
    return [] if n_shots == 0 else ICL_EGS[dataset][:n_shots]


# Stopping criteria


def is_dp_dars_finished(dp):
    return (dp.max_n_trials > 0 and dp.n_trials >= dp.max_n_trials) or (
        dp.min_n_corrects > 0 and dp.n_corrects > dp.min_n_corrects
    )


# IO


def get_res_fname(
    model_name_or_path: str,  # HF ID or path to the model.
    max_new_toks: int,  # Maximum length of the model output in token.
    temperature: float,  # Temperature for sampling.
    top_p: float,  # Top-p for sampling.
    prompt_template: str,  # ID or path to the prompt template.
    dataset: str,  # Name of the dataset to generate on.
    n_shots: int,  # Number of egs in few-shot prompt.
    tag: str,  # Tag describing sample number informantion for the result file.
    inf_seed: int,  # Seed for randomness.
) -> str:  # Path to the result file.
    """Get the JSONL file name to save results to."""
    # LLM-specific
    res_filename = get_pathname_from_name_or_path(model_name_or_path)
    res_filename += "-" + f"outlen{max_new_toks}"
    if temperature <= 0:
        temperature = 0
        top_p = 1
    res_filename += "-" + f"t{temperature}"
    res_filename += "-" + f"p{top_p}"
    res_filename += "-" + f"prompt-{get_pathname_from_name_or_path(prompt_template)}"

    # Dataset-specific
    res_filename += "-" + get_pathname_from_name_or_path(dataset)
    res_filename += "-" + f"{n_shots}shot"

    # Run specific
    res_filename += "-" + f"tag-{tag}"
    res_filename += "-" + f"seed{inf_seed}"

    res_filename += ".jsonl"

    return res_filename


# Pipeline


def gen(
    llm: LLM,
    sampling_params: SamplingParams,
    query_dps: list[QueryDataPoint],
    dp_stop_criteria: Callable[[QueryDataPoint], bool],
    resp_sample_cls: type = RespSampleVLLM,
    batch_evaluator: EvaluatorBatchBase | None = None,
    save_path: str | None = "auto",
    n_paths_per_save: int | None = None,
) -> list[RespSampleBase] | None:
    """Generate responses on the given query data points with specified stopping criteria.

    Parameters
    ----------
    llm : LLM
        The `vllm` model to generate with (or other objects with compatible `generate` interfaces).
    sampling_params : SamplingParams
        The sampling parameters for the `llm` (or other objects with compatible interfaces). NOTE: `n > 1` might cause bugs in `vllm` for now (0.4.2).
    query_dps : list[QueryDataPoint]
        The query-level data points to generate responses on.
    dp_stop_criteria : Callable[[QueryDataPoint], bool]
        The function to check if a query data point should be stopped generating on.
    resp_sample_cls : type, default: RespSampleVLLM
        The class to collect the generated response as.
    batch_evaluator : EvaluatorBatchBase | None, default: None
        The batch evaluator to evaluate the generated responses. `None` means no evaluation.
    save_path : str | None, default: "auto"
        Path to save the generated reponses to.
        `"tag:{tag}"` means saving to `"dart-math/data/gen/{gen-info}-sample-{tag}-seed{seed}.jsonl"`.
        `None` or `""` means no saving.
    n_paths_per_save : int | None, default: None
        Response-level samples or `None` if saving.

    Returns
    -------
    list[RespSampleBase] | None
        The generated responses or `None` if saving.
    """
    if save_path and "seed" not in save_path.lower():
        seed = sampling_params.seed
        prefix, ext = os.path.splitext(save_path)
        save_path = f"{prefix}-seed{seed}{ext}"  # Add seed to the file name

    model_name_or_path = llm.llm_engine.model_config.model

    all_new_samples = []
    sched_finished = False
    while not sched_finished:  # Loop on batches
        batch_dps = []
        batch_input_strs = []

        # Collect input strings batch
        batch_collected = False
        while not (batch_collected or sched_finished):
            # Loop on `query_samples` repeatedly
            sched_finished = True  # Speculate that all data points are finished
            for dp in query_dps:  # Loop on data points
                if dp_stop_criteria(dp):
                    continue  # Skip finished data point

                # Still have data points to generate
                sched_finished = False
                dp.n_trials += 1  # For `dp_stop_criteria`
                batch_dps.append(dp)

                # Make the input string
                icl_egs = get_icl_egs(dp.dataset, dp.n_shots, model_name_or_path)

                input_str = dp.prompt_template.make_full_prompt(dp.query, icl_egs)
                batch_input_strs.append(input_str.strip())
                # NOTE: `.strip()` is important for SFT/RL models and also acceptable for both 0-shot on SFT/RL models and few-shot on base models

                # Check if batch is full
                batch_collected = (
                    n_paths_per_save is not None
                    and n_paths_per_save > 0
                    and len(batch_input_strs) * sampling_params.n >= n_paths_per_save
                )
                if batch_collected or sched_finished:
                    break

        # rand_idx = random.choice(range(len(batch_input_strs)))
        logging.info(f"batch_input_strs[{0}]: {batch_input_strs[0]}")

        # Generate responses batch
        batch_req_outputs = llm.generate(batch_input_strs, sampling_params)
        logging.info(
            f"batch_req_outputs[0].outputs[0]: {batch_req_outputs[0].outputs[0]}"
        )

        # Collect outputs batch
        batch_new_samples = []
        for dp, req_output in zip(batch_dps, batch_req_outputs):
            for gen_path in req_output.outputs:
                batch_new_samples.append(resp_sample_cls.collect(dp, gen_path))

        if batch_evaluator is not None:
            answers, corrects = batch_evaluator.batch_eval(batch_new_samples)

            for sample, ans, correct in zip(batch_new_samples, answers, corrects):
                sample.ans = str(ans)
                sample.correct = correct if isinstance(correct, bool) else str(correct)
                sample.n_corrects += correct is True

        # Save responses batch incrementally
        if save_path is None or save_path == "":  # No saving, return
            all_new_samples += batch_new_samples
        else:  # Save to file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "a") as f:
                for sample in batch_new_samples:
                    f.write(orjson.dumps(sample.to_dict()).decode() + "\n")

    if save_path is None:
        return all_new_samples
    else:
        return None
