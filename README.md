# 🐋 Humback

An **unofficial** implementation of [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06259.pdf) .

The proposed Humback is a novel framework that can augment the instruction data for supervised fine-tuning with high quality.

![Humback Framework](./figs/humback.png)

## 🌴 Dependencies

- Python==3.11.4
- PyTorch==2.0.1
- Others: [requirements.txt](./requirements.txt)

## 🚀 QuickStart

### Seed Data Pre-processing

```bash
$ bash data/seed/download.sh
$ python data/seed/convert.py
# #data: 3286, #dump: 3200
# Instruction len: 149±266, Response len: 1184±799
```

### Unlabelled Data Pre-processing

Since ClueWeb22 is not a free open-source dataset, we sample texts from [falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) instead.

```bash
$ python data/unlabelled/falcon_refinedweb.py
```

### Train Backward Model $M_{yx}$


### Self-Augmentation via $M_{yx}$


### Self-Curation Prompting


### Train Seed Model $M_{0}$


### Train Models $M_{i}$ Iteratively


## 📑 Experimental Results



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
