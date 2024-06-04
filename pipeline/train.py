"""Modified from
- https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
- https://github.com/MeetKai/functionary/blob/main/functionary/train/train.py
"""

import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from dart_math.train import make_supervised_dset, monkey_patch4pack
from dart_math.utils import PROJ_HOME, init_logging
from transformers import Trainer

init_logging()


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m", metadata={"help": "Model name or path"}
    )


@dataclass
class DataArguments:
    data_path: list[str] = field(
        default=None, metadata={"help": "Path(s) to the training data."}
    )
    query_field: list[str] = field(
        default_factory=lambda: ["query"],
        metadata={"help": "Field name(s) for the query."},
    )
    resp_field: list[str] = field(
        default_factory=lambda: ["response"],
        metadata={"help": "Field name(s) for the response."},
    )
    prompt_template: str = field(
        default="alpaca",
        metadata={"help": "ID / Path to the file of prompt template."},
    )
    tokenized_cache_home: str = field(
        default=os.path.join(PROJ_HOME, "data/cache-tokenized"),
        metadata={"help": "Path to the tokenized cache home"},
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Seed for shuffling the dataset, default = 42"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    sliding_window: Optional[int] = field(
        default=None, metadata={"help": "Sliding window length in token."}
    )
    pack_len: int = field(
        default=0,
        metadata={
            "help": "Length of the packed sequence. Default as 0, use the `model_max_length`. Negative means disabling packing."
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    # 64 from https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    transformers.logging.set_verbosity_info()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        remaining_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if training_args.pack_len == 0:
        if LOCAL_RANK == 0:
            logging.info(
                """The shorter the `pack_len`, the less compute we need, so it should be no more than `model_max_length`,
                    setting `pack_len = model_max_length`"""
            )
        training_args.pack_len = training_args.model_max_length

    if LOCAL_RANK == 0:
        logging.debug(os.environ)
        logging.info(training_args)
        logging.info(model_args)
        logging.info(data_args)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        if LOCAL_RANK == 0:
            logging.info(f"Rope scaling factor: {scaling_factor}")
    # No cache for training
    config.use_cache = False
    # Set sliding window if it exists
    if (
        getattr(config, "sliding_window", None)
        and training_args.sliding_window is not None
    ):
        config.sliding_window = training_args.sliding_window
    # c.f. https://huggingface.co/docs/transformers/model_doc/mistral#sliding-window-attention
    # The Flash Attention-2 model uses also a more memory efficient cache slicing mechanism
    # - as recommended per the official implementation of Mistral model
    # that use rolling cache mechanism we keep the cache size fixed (self.config.sliding_window),
    # support batched generation only for padding_side="left"
    # and use the absolute position of the current token to compute the positional embedding.

    if training_args.pack_len > 0:
        if LOCAL_RANK == 0:
            logging.info("pack=True, using monkey patch")

        monkey_patch4pack(config)

    if LOCAL_RANK == 0:
        logging.info("Loading Model ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False if "stage3" in training_args.deepspeed else True,
        resume_download=True,
    )

    if LOCAL_RANK == 0:
        logging.info("Building tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    if LOCAL_RANK == 0:
        logging.info(f"Before adding, tokenizer length: {len(tokenizer)}")

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if LOCAL_RANK == 0:
        logging.info(f"After adding, tokenizer length: {len(tokenizer)}")

    if LOCAL_RANK == 0:
        logging.info("Building data module ...")

    supervied_dset = make_supervised_dset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        query_field=data_args.query_field,
        tokenized_cache_home=data_args.tokenized_cache_home,
        shuffle_seed=data_args.shuffle_seed,
        pack_len=training_args.pack_len,
        prompt_template=data_args.prompt_template,
    )
    # Dumb data collator, providing samples as is
    data_collator = transformers.DefaultDataCollator()

    if LOCAL_RANK == 0:
        logging.info("Building the trainer module ...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=supervied_dset,
        data_collator=data_collator,
    )

    if LOCAL_RANK == 0:
        logging.info("Training ...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if LOCAL_RANK == 0:
        logging.info("Saving model")
    trainer.save_model(output_dir=training_args.output_dir)

    if LOCAL_RANK == 0:
        logging.info("Trying to save trainer state ...")
    try:
        trainer.save_state()
    except Exception as e:
        if LOCAL_RANK == 0:
            logging.warning(f"Failed to save trainer state due to {e}")

    if LOCAL_RANK == 0:
        logging.info("All done!")


if __name__ == "__main__":
    train()
