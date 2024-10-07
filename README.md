# 🎯DART-Math


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

> Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving
> \[NeurIPS 2024\]
>
> [Yuxuan Tong](https://tongyx361.github.io), Xiwen Zhang, Rui Wang,
> Ruidong Wu, [Junxian He](https://jxhe.github.io)

📝 [Paper@arXiv](https://arxiv.org/abs/2407.13690) \| 🤗
[Datasets&Models@HF](https://huggingface.co/collections/hkust-nlp/dart-math-665704599b35de59f8fdf6c1)
\| 🐱 [Code@GitHub](https://github.com/hkust-nlp/dart-math) \| 🏆
[Published@NeurIPS 2024](https://nips.cc/virtual/2024/poster/92959)

🐦
[Thread@X(Twitter)](https://x.com/tongyx361/status/1811413243350454455)
\| 🐶 [中文博客@知乎](https://zhuanlan.zhihu.com/p/708371895) \| 📊
[Leaderboard@PapersWithCode](https://paperswithcode.com/paper/dart-math-difficulty-aware-rejection-tuning#results)
\| 📑
[BibTeX](https://github.com/hkust-nlp/dart-math?tab=readme-ov-file#%EF%B8%8F-citation)

> \[!IMPORTANT\]
>
> 🔥 **News!!!**
>
> - \[2024/09/25\] 🎉 *DART-Math* is accepted to [*NeurIPS
>   2024*](https://nips.cc/virtual/2024/poster/92959)!
> - \[2024/07/21\] Excited to find **our [`DART-Math-DSMath-7B`
>   (Prop2Diff)](https://huggingface.co/hkust-nlp/dart-math-dsmath-7b-prop2diff)
>   [comparable](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)
>   to the AIMO winner
>   [NuminaMath-7B](https://huggingface.co/AI-MO/NuminaMath-7B-CoT)** on
>   CoT, but based solely on
>   [MATH](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math-query-info)
>   &
>   [GSM8K](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-gsm8k-query-info)
>   prompt set, leaving much room to improve! Besides, our [`DART`
>   method](https://github.com/hkust-nlp/dart-math?tab=readme-ov-file#dars--difficulty-aware-rejection-sampling)
>   is also fully [compatible with tool-integrated
>   reasoning](https://github.com/hkust-nlp/dart-math?tab=readme-ov-file#tool-integrated-reasoning-reasoning-in-natural-language-interleaved-with-python-code).
>   Join the discussion under this [X
>   thread](https://x.com/tongyx361/status/1815112376649134172)!

<div align="center">

<img src="https://tongyx361.github.io/assets/dart-math/main-results.png" alt="Main results averaged on 2 in-domain and 4 challenging out-of-domain mathematical reasoning benchmarks." height=300px>
<img src="https://tongyx361.github.io/assets/dart-math/main-nresp-vs-query.png" alt="Number of responses v.s. query descending in difficulty in DART-Math datasets and similar-sized VRT baseline" height=300px>

</div>

<div align="left">

<sup> Figure 1: <strong>Left:</strong> Average accuracy on 6
mathematical benchmarks. We compare with models fine-tuned on the best,
public instruction tuning datasets for mathematical problem-solving:
MetaMath <a href="https://openreview.net/forum?id=N8N0hgNDRt">(Yu et
al., 2024)</a> with 395K examples, MMIQC
<a href="https://arxiv.org/abs/2401.09003">(Liu et al., 2024a)</a> with
2.3 million examples, as well as vanilla rejection tuning (VRT) with
590K examples. Both <em>DART-Math (Uniform)</em> and <em>DART-Math
(Prop2Diff)</em> use 590K training examples. <strong>Right:</strong>
Number of responses for each query descending by difficulty across 3
synthesis strategies. Queries are from the MATH training split
<a href="https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html">(Hendrycks
et al., 2021)</a>. VRT is the baseline biased towards easy queries,
while <em>Uniform</em> and <em>Prop2Diff</em> are proposed in this work
to balance and bias towards difficult queries respectively. Points are
slightly shifted and downsampled for clarity. </sup>

</div>

| Dataset | Setting | \# of Samples | [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | [GSM8K](https://huggingface.co/datasets/gsm8k) | [College](https://github.com/hkust-nlp/dart-math/tree/main/data/dsets/mwpbench/college-math-test.jsonl) | Download |
|:---|:---|---:|---:|---:|---:|:--:|
| `DART-Math-Uniform` | Unifrom | 591k | 52.9 | **88.2** | 40.1 | 🤗 [HuggingFace](https://huggingface.co/datasets/hkust-nlp/dart-math-uniform) |
| `DART-Math-Hard` | Prop2Diff | 585k | **53.6** | 86.8 | **40.7** | 🤗 [HuggingFace](https://huggingface.co/datasets/hkust-nlp/dart-math-hard) |
| `DART-Math-Pool-MATH` | – | 1615k | – | – | – | 🤗 [HuggingFace](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math) |
| `DART-Math-Pool-GSM8K` | – | 2739k | – | – | – | 🤗 [HuggingFace](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-gsm8k) |

<sup>MATH and GSM8K are **in-domain**, while College(Math) is
**out-of-domain**. Performance here are of `DART-Math` models fine-tuned
from
[DeepSeekMath-7B](https://huggingface.co/deepseek-ai/deepseek-math-7b-base).
**Bold** means the best score on the respective base model here.</sup>

| Model | [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | [GSM8K](https://huggingface.co/datasets/gsm8k) | [CollegeMath](https://github.com/hkust-nlp/dart-math/tree/main/data/dsets/mwpbench/college-math-test.jsonl) | Download |
|:---|---:|---:|---:|:--:|
| `DART-Math-Llama3-70B` (Uniform) | 54.9 | **90.4** | **38.5** | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-llama3-70b-uniform) |
| `DART-Math-Llama3-70B` (Prop2Diff) | **56.1** | 89.6 | 37.9 | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-llama3-70b-prop2diff) |
| `DART-Math-DSMath-7B` (Uniform) | 52.9 | **88.2** | 40.1 | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-dsmath-7b-uniform) |
| `DART-Math-DSMath-7B` (Prop2Diff) | **53.6** | 86.8 | **40.7** | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-dsmath-7b-prop2diff) |
| `DART-Math-Mistral-7B` (Uniform) | 43.5 | **82.6** | 26.9 | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-mistral-7b-uniform) |
| `DART-Math-Mistral-7B` (Prop2Diff) | **45.5** | 81.1 | **29.4** | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-mistral-7b-prop2diff) |
| `DART-Math-Llama3-8B` (Uniform) | 45.3 | **82.5** | 27.1 | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-llama3-8b-uniform) |
| `DART-Math-Llama3-8B` (Prop2Diff) | **46.6** | 81.1 | **28.8** | 🤗 [HuggingFace](https://huggingface.co/hkust-nlp/dart-math-llama3-8b-prop2diff) |

<sup>MATH and GSM8K are <b>in-domain</b>, while CollegeMath is
<b>out-of-domain</b>. **Bold** means the best score on the respective
base model here.</sup>

## `DART-Math` Models: SOTA on Various In-Domain and Out-of-Domain Benchmarks

`DART-Math` models achieve performance **superior or competitive to
previous SOTAs** on 2 in-domain and 4 challenging out-of-domain
mathematical reasoning benchmarks, despite using **much smaller
datasets** and **no proprietary model like GPT-4**.

| Model | [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | [GSM8K](https://huggingface.co/datasets/gsm8k) | [College](https://github.com/hkust-nlp/dart-math/tree/main/data/eval-dsets/mwpbench/college-math-test.jsonl) | [DM](https://github.com/hkust-nlp/dart-math/tree/main/data/eval-dsets/deepmind-mathematics.json) | [Olympiad](https://github.com/hkust-nlp/dart-math/tree/main/data/eval-dsets/olympiadbench/OE_TO_maths_en_COMP.json) | [Theorem](https://github.com/hkust-nlp/dart-math/tree/main/data/eval-dsets/theoremqa.json) | AVG |
|:---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4 (0314) | [52.6](https://arxiv.org/abs/2403.04706) | [94.7](https://arxiv.org/abs/2403.04706) | [24.4](https://arxiv.org/abs/2403.02884) | – | – | – | – |
| Llama3-70B-MetaMath | 44.9 | 88.0 | 31.9 | 53.2 | 11.6 | 21.9 | 41.9 |
| [`DART-Math-Llama3-70B`](https://huggingface.co/hkust-nlp/dart-math-llama3-70b-prop2diff) | **56.1** | **89.6** | **37.9** | **64.1** | **20.0** | **28.2** | **49.3** |
| DeepSeekMath-7B-MetaMath | 43.7 | 81.8 | 33.7 | 53.0 | 13.6 | 23.2 | 41.5 |
| [DeepSeekMath-7B-RL](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl) | 53.1 | 88.4 | 41.3 | 58.3 | 18.7 | 35.9 | 49.3 |
| [`DART-Math-DSMath-7B`](https://huggingface.co/hkust-nlp/dart-math-dsmath-7b-prop2diff) | **53.6** | **86.8** | **40.7** | **61.6** | **21.7** | **32.2** | **49.4** |
| Mistral-7B-MetaMath | 29.8 | 76.5 | 19.3 | 28.0 | 5.9 | 14.0 | 28.9 |
| [`DART-Math-Mistral-7B`](https://huggingface.co/hkust-nlp/dart-math-mistral-7b-prop2diff) | **45.5** | **81.1** | **29.4** | **45.1** | **14.7** | **17.0** | **38.8** |
| Llama3-8B-MetaMath | 32.5 | 77.3 | 20.6 | 35.0 | 5.5 | 13.8 | 30.8 |
| [`DART-Math-Llama3-8B`](https://huggingface.co/hkust-nlp/dart-math-llama3-8b-prop2diff) | **46.6** | **81.1** | **28.8** | **48.0** | **14.5** | **19.4** | **39.7** |

<sup>**Abbreviations**: College (CollegeMath), DM (DeepMind
Mathematics), Olympiad (OlympiadBench-Math), Theorem (TheoremQA).
**Bold** means the best score by SFT on the respective base model here.
`DART-Math` models here are fine-tuned on the [`DART-Math-Hard`
dataset](https://huggingface.co/datasets/hkust-nlp/dart-math-hard).</sup>

## `DART-Math` Datasets: SOTA & Data-Efficient & Open-Source

`DART-Math` are the **state-of-the-art** and **data-efficient**
**open-source** instruction tuning datasets for mathematical reasoning.

Most of previous datasets are **constructed with ChatGPT**, and many of
them are **not open-source**, especially for ones of the best
performance.

| Math SFT Dataset | \# of Samples | [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | [GSM8K](https://huggingface.co/datasets/gsm8k) | [College](https://github.com/hkust-nlp/dart-math/tree/main/data/eval-dsets/mwpbench/college-math-test.jsonl) | Synthesis Agent(s) | Open-Source |
|:---|---:|---:|---:|---:|:---|:--:|
| [WizardMath](https://arxiv.org/abs/2308.09583) | 96k | 32.3 | 80.4 | 23.1 | GPT-4 | ✗ |
| [MetaMathQA](https://arxiv.org/abs/2309.12284) | 395k | 29.8 | 76.5 | 19.3 | GPT-3.5 | [✓](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| [MMIQC](https://arxiv.org/abs/2401.09003) | **2294k** | 37.4 | 75.4 | *28.5* | **GPT-4+GPT-3.5+Human** | [**✓**](https://huggingface.co/datasets/Vivacem/MMIQC) |
| [Orca-Math](https://arxiv.org/abs/2402.14830) | 200k | – | – | – | GPT-4 | [✓](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) |
| [Xwin-Math-V1.1](https://arxiv.org/abs/2403.04706) | **1440k** | *45.5* | **84.9** | 27.6 | **GPT-4** | **✗** |
| [KPMath-Plus](https://arxiv.org/abs/2403.02333) | **1576k** | **46.8** | 82.1 | – | **GPT-4** | **✗** |
| [MathScaleQA](https://arxiv.org/abs/2403.02884) | 2021k | 35.2 | 74.8 | 21.8 | GPT-3.5+Human | ✗ |
| [`DART-Math-Uniform`](https://huggingface.co/datasets/hkust-nlp/dart-math-uniform) | **591k** | 43.5 | *82.6* | 26.9 | **DeepSeekMath-7B-RL** | [**✓**](https://huggingface.co/datasets/hkust-nlp/dart-math-uniform) |
| [`DART-Math-Hard`](https://huggingface.co/datasets/hkust-nlp/dart-math-hard) | **585k** | *45.5* | 81.1 | **29.4** | **DeepSeekMath-7B-RL** | [**✓**](https://huggingface.co/datasets/hkust-nlp/dart-math-hard) |

<sup>MATH and GSM8K are **in-domain**, while College(Math) is
**out-of-domain**. Performance here are of models fine-tuned from
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), except
for Xwin-Math-V1.1 based on
[Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf).
**Bold**/*Italic* means the best/second best score here.</sup>

## `DARS` – Difficulty-Aware Rejection Sampling

Our analysis of previous datasets reveals **severe biases towards easy
queries**, with **frequent failures to generate any correct response for
the most challenging queries**.

This primarily arises from their constuction method, **vanilla rejection
sampling**, where **the same number** of responses are sampled for each
query, yet the likelihood of obtaining correct responses for difficult
queries is significantly lower, sometimes even zero.

Motivated by the observation above and the intuitive that difficult
samples are critical for learning complexing reasoning, we propose
**Difficulty-Aware Rejection Sampling** (`DARS`) to eliminate the bias
towards easy queries. Specifically, we introduce two strategies to
increase the number of correct responses for difficult queries:

1.  **Uniform**, which involves sampling responses for each query until
    **each query accumulates $k_u$ correct responses**, where $k_u$ is a
    preset hyperparameter determined by the desired size of the
    synthetic dataset;
2.  **Prop2Diff**, where we continue sampling responses until the number
    of correct responses for each query is **proportional to its
    difficulty score**. The most challenging queries will receive $k_p$
    responses and kp is a hyperparameter. This method introduces a
    deliberate bias in the opposite direction to vanilla rejection
    sampling, towards more difficult queries, inspired by previous works
    that demonstrate **difficult samples can be more effective to
    enhance model capabilities** ([Sorscher et al.,
    2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html);
    [Liu et al., 2024b](https://openreview.net/forum?id=BTKAeLqLMw)).

See [Figure 1
(Right)](https://tongyx361.github.io/assets/dart-math/main-nresp-vs-query.png)
for examples of `DART-Math-Uniform` by `DARS-Uniform` and
`DART-Math-Hard` by `DARS-Prop2Diff`.

## 🚀 Quick Start / Reproduction

### ⚙️ Setup

We recommend using [Conda](https://docs.conda.io/projects/miniconda) and
[pip](https://pip.pypa.io/en/stable/#) to manage your environment. Run
the following commands to setup your environment:

``` shell
git clone https://github.com/hkust-nlp/dart-math.git && cd dart-math
conda create --name dart-math --yes python=3.11
conda activate dart-math
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

For common users/developers, please just run the following command the
install the `dart-math` package:

``` shell
pip install -e "."
```

For intended contributors, we recommend installing the package with the
`dev` extras:

``` shell
pip install -e ".[dev]"
pre-commit install
conda install quarto -c conda-forge # for building the documentation
```

### 🔨 Training

We implement an efficient training pipeline utilizing various
techniques. Notably, [**sequence
packing**](https://hkust-nlp.github.io/dart-math/train.html#sequence-packing)
accelerates training by 6-8x in our setting and possibly more in other
settings. (See [how to integrate sequence packing in 4 lines of
code](https://hkust-nlp.github.io/dart-math/train.html#accelerating-several-times-with-sequence-packing-in-4-lines-of-code).)

Please refer to

- the [training Python
  script](https://github.com/hkust-nlp/dart-math/blob/main/pipeline/train.py)
  for code of training based on the [HuggingFace
  `Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer)
  and utilizing [sequence
  packing](https://hkust-nlp.github.io/dart-math/train.html#sequence-packing).
- the
  [single-node](https://github.com/hkust-nlp/dart-math/blob/main/scripts/train-single-node.sh)/[multi-node](https://github.com/hkust-nlp/dart-math/blob/main/scripts/train-multi-node.sh)
  training `bash` script for code of training based on [HuggingFace
  `accelerate`](https://huggingface.co/docs/accelerate/index) and
  [`deepspeed`](https://www.deepspeed.ai)

Here, we provide some example commands as well as reproduction
instructions for our work:

#### Single-Node Training

For example, to reproduce training `DART-Math-Llama3-8B-Prop2Diff` on a
node of 8 A100 GPUs, please run the following command:

``` shell
bash scripts/train-single-node.sh \
    --data_path "hkust-nlp/dart-math-hard" \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --lr "5e-5" --bs 64 --n_grad_acc_steps 1 --n_epochs 1 \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --output_dir "models/dart-math-llama3-8b-prop2diff"
```

To reproduce other training settings, just refer to the paper and modify
the `--data_path`, `--model_path`, `--lr`, `--n_grad_acc_steps`,
`--n_epochs` and `--output_dir` arguments accordingly.

#### Multi-Node Training

To reproduce training `DART-Math-Llama3-70B-Prop2Diff` on 4 nodes of 8
A100 GPUs, please first edit the `cfgs/deepspeed/hostfile` according to
your enviroment and then run the following command:

``` shell
bash scripts/train-multi-node.sh \
    --data_path "hkust-nlp/dart-math-hard" \
    --model_path "meta-llama/Meta-Llama-3-70B" \
    --lr "2e-5" --bs 64 --n_grad_acc_steps 1 --n_epochs 1 \
    --n_nodes 4 \
    --output_dir "models/dart-math-llama3-70b-prop2diff"
```

To reproduce training `DART-Math-Llama3-70B-Uniform` on 4 nodes of 8
A100 GPUs, just change `--data_path` to `"hkust-nlp/dart-math-uniform"`.

<details>

<summary>

The off-the-shelf command to train `DART-Math-Llama3-70B-Uniform`
</summary>

``` shell
bash scripts/train-multi-node.sh \
    --data_path "hkust-nlp/dart-math-uniform" \
    --model_path "meta-llama/Meta-Llama-3-70B" \
    --lr "2e-5" --bs 64 --n_grad_acc_steps 1 --n_epochs 1 \
    --n_nodes 4 \
    --output_dir "models/dart-math-llama3-70b-prop2diff"
```

</details>

### ⚖️ Evaluation

We utilize [vLLM](https://docs.vllm.ai/en/latest/index.html) to
accelerate inference and an elaborate answer extraction and correctness
judgement pipeline based on regular expressions and
[SymPy](https://www.sympy.org) symbolic calculation, which is able to
correctly process

- most **mathematical objects** such as matrices (vectors), intervals,
  symbols besides numbers,
- as well as some **special texts** like bool expressions, dates and
  times.

For example, to reproduce one pass of greedy decoding with
`DART-Math-Mistral-7B-Prop2Diff` on the 6 benchmarks in Table 2 on GPU
0, please run the following command:

``` shell
CUDA_VISIBLE_DEVICES="0" python pipeline/gen.py \
    --gen_save_path "data/res/dart-math-mistral-7b-prop2diff.jsonl" \
    --model_name_or_path "hkust-nlp/dart-math-mistral-7b-prop2diff" \
    --datasets "math/test" "gsm8k/test" "mwpbench/college-math/test" "deepmind-mathematics" \
        "olympiadbench/OE_TO_maths_en_COMP" "theoremqa" \
    --max_new_toks 2048 --temperature 0 \
    --prompt_template "cot" --n_shots -1 \
    --inf_seed -1 \
    --max_n_trials 1
```

To reproduce other inference settings, just refer to the paper and
modify the `--model_name_or_path` and `--gen_save_path` arguments
accordingly.

- We observed that Llama-3-8B(-Base) tends to decode EoS immediately
  sometimes. Try use `--ignore_eos` as a workaround.

For other general inference settings, please modify the command or
directly modify the
[script](https://github.com/hkust-nlp/dart-math/blob/main/pipeline/gen.py).

- To test **base** models, please add the corresponding **ID** to
  `BASE_MODEL_IDS` from
  [dart_math.utils](https://github.com/hkust-nlp/dart-math/blob/main/dart_math/utils.py).
- To test **instruct** models, please add the corresponding **prompt
  template** to `PROMPT_TEMPLATE_ID2DICT` from
  [dart_math.utils](https://github.com/hkust-nlp/dart-math/blob/main/dart_math/utils.py)
  and specify with `--prompt_template`.

You can also add the `--gen_only` option to only generate responses
without evaluation and use the
[`EvaluatorMathBatch`](https://hkust-nlp.github.io/dart-math/eval.html#evaluatormathbatch)
to grade the generations by yourself. Please check the [grading
script](pipeline/grade.py) for example.

### 🗂 Data Synthesis

Our data synthesis pipeline is compatible with the evaluation pipeline,
please **modify the `--min_n_corrects` and `--max_n_trials` arguments**
to meet your needs.

For example, to reproduce the **synthesis of `DART-Math-Uniform`**,
amortizing the workload to multiple GPUs, please run the following
command:

``` shell
gpu_ids_list=("0" "1" "2" "3" "4" "5" "6" "7")
min_n_corrects=40
min_n_corrects_per_gpu=$((min_n_corrects / ${#gpu_ids_list[@]})) # 5 here

mkdir -p logs
for gpu_ids in "${gpu_ids_list[@]}"; do
    exp_name="dart-math-uniform-gpu${gpu_ids}"
    CUDA_VISIBLE_DEVICES="${gpu_ids}" python pipeline/gen.py \
        --gen_save_path "data/res/${exp_name}.jsonl" \
        --model_name_or_path "deepseek-ai/deepseek-math-7b-rl" \
        --datasets "math/train" "gsm8k-fix/train" \
        --max_new_toks 2048 --temperature 1.6 --top_p 0.95 \
        --prompt_template "deepseekmath" --n_shots 0 \
        --inf_seed -1 \
        --min_n_corrects "${min_n_corrects_per_gpu}" --max_n_trials 0 \
        >"logs/${exp_name}.log" 2>&1 &
    # NOTE: `--max_n_trials 0` means possible infinite trials, kill the job manually when needed
done
```

<sup>NOTE: Some **erroneous labels** exist in the GSM8K dataset, so we
tried to fix them and produced
[`gsm8k-fix`](https://huggingface.co/datasets/hkust-nlp/gsm8k-fix).
</sup>

To reproduce the data synthesis of the **Vanilla Rejection Tuning (VRT)
baseline** in the paper, just set
`--max_n_trials 52 --min_n_corrects 0`.

<details>

<summary>

The off-the-shelf command to reproduce the data synthesis of the Vanilla
Rejection Tuning (VRT) baseline in the paper
</summary>

``` shell
CUDA_VISIBLE_DEVICES="0" python pipeline/gen.py \
    --gen_save_path "data/res/dart-math-uniform.jsonl" \
    --model_name_or_path "deepseek-ai/deepseek-math-7b-rl" \
    --datasets "math/train" "gsm8k-fix/train" \
    --max_new_tokens 2048 --temperature 1.6 --top_p 0.95 \
    --prompt_template "cot" --n_shots 0 \
    --inf_seed -1 \
    --max_n_trials 52 --min_n_corrects 0 # no requirement for correct responses
```

</details>

After the synthesis, you can use the [curation
script](pipeline/curate.py) to curate the final dataset.

## [`dart-math` Package](https://hkust-nlp.github.io/dart-math): Efficient and Flexible Training & Inference & Evaluation Pipelines

We package our code of effcient and flexible training & inference &
evaluation pipelines into `dart-math` and document it at [this
website](https://hkust-nlp.github.io/dart-math/quick-start.html).

The `dart-math` package provides the following useful features besides
ones mentioned above:

### **Tool-integrated reasoning**: reasoning in natural language interleaved with Python code

Example command to evaluate DeepSeekMath-7B-RL with tool-integrated
reasoning (following the DeepSeekMath offical setting):

``` shell
CUDA_VISIBLE_DEVICES="0" python pipeline/gen.py \
    --gen_save_path "data/res/dsmath-7b-rl-tool-math-test.jsonl" \
    --model_name_or_path "deepseek-ai/deepseek-math-7b-rl" \
    --datasets "math-test" \
    --max_new_toks 2048 --temperature 0 \
    --prompt_template "deepseekmath-tool" --n_shots 0 \
    --max_n_calls 1 --trunc_len 50 50 \
    --inf_seed -1 \
    --max_n_trials 1
# Reproduced performance (with our evaluator): 56.08%
# (58.8% reported originally with DeepSeekMath evaluator)
```

For other general inference settings, please modify the options related
to the [`Generator.code_exec_cfg`
attribute](https://hkust-nlp.github.io/dart-math/gen.html#:~:text=means%20no%20evaluation.-,code_exec_cfg,-dart_math.exec.CodeExecCfg)
in the command or the
[script](https://github.com/hkust-nlp/dart-math/blob/main/pipeline/gen.py).

## 🍀 Contribution

### File Structure

``` tree
dart-math
├── data
├── cfgs # Configurations
├── utils # Repository utilities
├── dart_math # Package code for common utilities
├── nbs # Notebooks and other files to run tests and generate documentation with https://nbdev.fast.ai
├── pipeline # Reusable (Python / Shell) scripts or notebooks
└── scripts # Setting-specific scripts
```

### Checklist Before Commit

#### [`prepare-commit.sh`](utils/prepare-commit.sh)

Run the [`prepare-commit.sh`](utils/prepare-commit.sh) to clean the
notebooks and export scripts for pipeline notebooks, generate
documentation, run tests, render README if needed:

``` shell
bash utils/prepare-commit.sh
```

Please refer to the comments in the script for how it works.

#### Manual Modification List

- Add `if __name__ == "__main__":` to scripts that might use vLLM tensor
  parallelism
  - [`gen.py`](pipeline/gen.py)

## 🌟 Star History

<a href="https://star-history.com/#hkust-nlp/dart-math&Date"> <picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hkust-nlp/dart-math&type=Date&theme=dark" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hkust-nlp/dart-math&type=Date" />
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hkust-nlp/dart-math&type=Date" />
</picture> </a>

## 🙏 Acknowledgements

Thanks to:

- [`nbdev`](https://nbdev.fast.ai/) for generating the [wonderful
  documentation website](https://hkust-nlp.github.io/dart-math),
- [`stanford_alpaca`](https://github.com/tatsu-lab/stanford_alpaca) for
  reference code about training,
- [`functionary`](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)
  for reference code about [sequence
  packing](https://hkust-nlp.github.io/dart-math/train.html#sequence-packing).
- @HYZ17 for extensive tests and helpful suggestions.

## ☕️ Citation

If you find our data, model or code useful for your work, please kindly
cite [our paper](https://arxiv.org/abs/2407.13690):

``` latex
@article{tong2024dartmath,
  title={DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving},
  author={Yuxuan Tong and Xiwen Zhang and Rui Wang and Ruidong Wu and Junxian He},
  year={2024},
  eprint={2407.13690},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.13690},
}
```
