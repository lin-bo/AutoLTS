# AutoLTS

This repository is the implementation of [AutoLTS: Automating Cycling Stress Assessment via Contrastive Learning and Spatial Post-processing](123.com). 

<p align="center">
<img src="/img/mdl_arch_final_with_illustrations.png" width=700>
</p>
  
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Create a folder for checkpoints:
```
mkdir checkpoint
```

Download the image data [here](https://drive.google.com/file/d/1DQxCbSr7J9_h5asjnq2hGE-iHwG1BGsx/view?usp=share_link) and put the folder under `./data/streetview/dataset`.

Download other data [here](https://drive.google.com/file/d/1UI1LvhHZln0In6eXn1PUhZ7m902EWgcM/view?usp=share_link) and put them under `./data/`

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

You can download pretrained models following this [link](https://drive.google.com/drive/folders/1f76sAj2vtxgJmZz-3LEg8oB2zftOl7r5?usp=share_link). 
Models are named by <Split Name>-<scenario number>.pt 
where split name can be random, york, etobicoke, and scarborough, and scenario number can be sce1, sce2, and sce3.


