# AutoLTS

This repository is the official implementation of [AutoLTS: Automating Cycling Stress Assessment via Contrastive Learning and Spatial Post-processing](123.com). 

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Create a folder for checkpoints:
```
mkdir checkpoint
```

Download the datasets [here](123.com) and put them under `./data/`

## Training

To train the image encoder, run:

```train
python MoCo_train.py --device=<GPU name> -ne=100 -nc=1 --lr=30 -bs=256 --memsize=25600 --aware --awaretype=hie --label=lts_wo_volume --no-hlinc --temperature=0.07 --lambd=0.95
```
To train the final LTS prediction module, run:
```
python MoCo_clf_train.py --device=<GPU name> -ne=100 -nc=1 --lr=0.0003 -bs=128 --no-local --label=lts_wo_volume --checkpoint=<Name of the trained encoder> --sidefea sce1_prob
```

## Evaluation

To evaluate the model, run:

```eval
python test.py --device=<GPU name> --modelname=MoCoClfFeaV3 --label=lts_wo_volume --checkpointname=<model name> --sidefea sce1_prob
```

## Pre-trained Models

You can download pretrained models here:

- [AutoLTS-Random-Sce1]()
- [AutoLTS-Random-Sce2]()
- [AutoLTS-Random-Sce3]()
- [AutoLTS-York-ce1]()
- [AutoLTS-York-Sce2]()
- [AutoLTS-York-Sce3]()
- [AutoLTS-Etobicoke-Sce1]()
- [AutoLTS-Etobicoke-Sce2]()
- [AutoLTS-Etobicoke-Sce3]()
- [AutoLTS-Scarborough-Sce1]()
- [AutoLTS-Scarborough-Sce2]()
- [AutoLTS-Scarborough-Sce3]()


