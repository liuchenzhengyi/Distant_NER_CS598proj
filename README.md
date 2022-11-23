## Description

This is the project for course CS 598 Text Mining: A New Paradigm

The code is adapted from [BOND](https://github.com/cliang1453/BOND)


## Data

We release five open-domain distantly/weakly labeled NER datasets here: [dataset](dataset). For gazetteers information and distant label generation code, please directly email cliang73@gatech.edu.

## Environment

Python 3.7, Pytorch 1.3, Hugging Face Transformers v2.3.0.

## Training & Evaluation

We provides the training scripts for all five open-domain distantly/weakly labeled NER datasets in [scripts](scripts). 

For our proposed MoE model:
```
cd BOND
./scripts/run_scripts.sh
```

For baseline model:
```
cd BOND
./scripts/conll_self_training.sh
```
For Stage I training and evaluation on CoNLL03
```
cd BOND
./scripts/conll_baseline.sh
```

