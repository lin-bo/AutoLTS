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

Download the image data [here](https://utoronto-my.sharepoint.com/:f:/g/personal/imbo_lin_mail_utoronto_ca/EmxhwgYm-UBKv5fnoWUEdGoB4PzT8G2RzFHEW8u79IOw0w?e=kNlPei) and put the folder under `./data/streetview`
Download the datasets [here](https://utoronto-my.sharepoint.com/:f:/g/personal/imbo_lin_mail_utoronto_ca/EltmiXVh5FZAikzA5xxKNB4Bj1eDFY9vK-EKJ1z4svFrng) and put them under `./data/`

## Training

To train the image encoder, run:

```train
python MoCo_train.py --device=<GPU name> -ne=100 -nc=1 --lr=30 -bs=256 --memsize=25600 --aware --awaretype=hie --label=lts_wo_volume --no-hlinc --temperature=0.07 --lambd=0.95
```
To train the road feature prediction models, run:
```
python Res50_train.py --device=<GPU name> -ne=100 -nc=1 --lr=0.0001 -bs=128 --label=<name of the labels, choose from speed_actual_onehot, cyc_infras_onehot, n_lanes_onehot, oneway_onehot, parking_onehot, and road_type_onehot>
```

To generate the road feature predictions, run
```
python gen_prediction.py --device=<GPU name> --model=Res50 -bs=128 --modelname=Res50 --label=<Road feature name> --checkpoint=<model name> --prob_pred
```

To run the post-processing module, run 
```
python postprocessing.py
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
- [AutoLTS-York-Sce1]()
- [AutoLTS-York-Sce2]()
- [AutoLTS-York-Sce3]()
- [AutoLTS-Etobicoke-Sce1]()
- [AutoLTS-Etobicoke-Sce2]()
- [AutoLTS-Etobicoke-Sce3]()
- [AutoLTS-Scarborough-Sce1]()
- [AutoLTS-Scarborough-Sce2]()
- [AutoLTS-Scarborough-Sce3]()


