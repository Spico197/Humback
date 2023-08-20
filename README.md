# 🐋 Humback

An **unofficial** implementation of [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06259.pdf) .

The proposed Humback is a novel framework that can augment the instruction data for supervised fine-tuning with high quality.

🚧 Currently, this repo is under construction and not finished.

![Humback Framework](./figs/humback.png)

## 🌴 Dependencies

- Python==3.11.4
- PyTorch==2.0.1
- Others: [requirements.txt](./requirements.txt)

## 🚀 QuickStart

Procedure (2 iters):
1. Prepare seed data and unlabelled data.
2. Train the backward model $M_{yx}$ on the reversed seed data.
3. Self-augment the seed data via $M_{yx}$.
4. Train a forward model $M_{0}$ on the seed data.
5. Self-curate the unlabelled data $A_{k}^{(1)}$ via $M_{0}$ (tag quality scores).
6. Train a forward model $M_{1}$ on the self-curated unlabelled data $A_{k}^{(1)}$.
7. Use $M_{1}$ to self-curate the unlabelled data $A_{k}^{(2)}$.
8. Train a forward model $M_{2}$ on the self-curated unlabelled data $A_{k}^{(2)}$.

### Seed Data Pre-processing

We follow the original paper and use [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) to construct the seed data.

The processed data could be found [here](https://github.com/Spico197/Humback/releases/tag/data) .

```bash
$ bash data/seed/download.sh
$ python data/seed/convert.py
# #data: 3286, #dump: 3200
# Instruction len: 149±266, Response len: 1184±799
```

### Unlabelled Data Pre-processing

Since ClueWeb22 is not a free open-source dataset, we sample texts from [falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) instead.

The processed data could be found [here](https://github.com/Spico197/Humback/releases/tag/data) .

```bash
$ python data/unlabelled/falcon_refinedweb.py
```

### Train Backward Model $M_{yx}$

| Item                   | Value                                                                       |
| :--------------------- | :-------------------------------------------------------------------------- |
| Foundation Model       | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| GPUs                   | 8 * A100 40GB                                                               |
| Mixed Precision        | bf16                                                                        |
| Gradient Checkpointing | on                                                                          |
| ZeRO-Offload           | Stage 2                                                                     |
| Batch size             | 32                                                                          |
| Steps                  | 500                                                                         |

```bash
# The first Myx training takes about 30min (on the seed data).
$ bash scripts/train_backward_Myx.sh
```

The pre-trained $M_{yx}$ is available at [Huggingface](https://huggingface.co/Spico/Humback-Myx).

### Self-Augmentation via $M_{yx}$

```bash
# Taking about 6:40:45 on the unlabelled data with 8*A100.
$ bash scripts/self_aug.sh
```

### Train Seed Model $M_{0}$

Hyper parameters are the same as $M_{yx}$.

```bash
$ bash scripts/train_seed.sh
```

The pre-trained $M_{0}$ is available at [Huggingface](https://huggingface.co/Spico/Humback-M0) (Uploading).

### Self-Curation Prompting

```bash
$ bash scripts/self_curation.sh
$ cat outputs/m1/unlabelled_curated_data.jsonl data/seed/seed.jsonl > data/curated/m1.jsonl
```

### Train Models $M_{i}$

Most hyper parameters are the same as $M_{yx}$ except for the number of steps (the original Humback trains 1600 steps on 512k samples).

| Item  | Value |
| :---- | :---- |
| Steps | 1400  |

```bash
# change the `--data_path` in `scripts/train_seed.sh`
$ bash scripts/train_seed.sh
```

## 📑 Experimental Results

Other models: [HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) .

| Model           | Average |   ARC | HellaSwag |  MMLU | TruthfulQA |
| :-------------- | ------: | ----: | --------: | ----: | ---------: |
| Llama-2-7b      |   54.32 | 53.07 |     78.59 | 46.87 |      38.76 |
| Llama-2-7b-chat |   56.34 | 52.90 |     78.55 | 48.32 |      45.57 |
| Vicuna-7b-v1.3  |   55.62 | 50.43 |     76.92 | 48.14 |      47.01 |
| Humback $M_{0}$ |   58.13 | 56.31 |     81.20 | 47.45 |      47.59 |
| Humback $M_{1}$ |         |       |           |       |            |
| Humback $M_{2}$ |         |       |           |       |            |

## 💌 Acknowledgments

- Paper: [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06259.pdf)
- Code: [FastChat](https://github.com/lm-sys/FastChat)
- Code: [vLLM](https://github.com/vllm-project/vllm)
- Code: [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- Code: [transformers](https://huggingface.co/transformers/)

## 📝 TODO

- [ ] Run the second curation iteration.
- [ ] Distinguish the seed data and the augmentation-curated data with different system prompts as described in the last paragraph of section 2.3.

## 📜 Reference

```bibtex
@misc{li2023selfalignment,
    title={Self-Alignment with Instruction Backtranslation},
    author={Xian Li and Ping Yu and Chunting Zhou and Timo Schick and Luke Zettlemoyer and Omer Levy and Jason Weston and Mike Lewis},
    year={2023},
    eprint={2308.06259},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
