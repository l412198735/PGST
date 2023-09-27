# Phrase Grounding-based Style Transfer for Single-Domain Generalized Object Detection

<img src="docs/lead.png" width="800"> 




## Updates
* 12/06/2021: the paper on arxiv xxxxx.

* 09/26/2023: Project page built. <br/>

## Installation and Setup

***Environment***

This repo requires Pytorch>=1.9 and torchvision. We recommand using docker to setup the environment. You can use this pre-built docker image ``docker pull pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`` or this one ``docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`` depending on your GPU.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
python setup.py build develop --user
```

Our code is build on [GLIP](https://github.com/microsoft/GLIP), please install the environment following GLIP.

### Datasets

#### Diverse Weather [[Download link](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)]
#### Convered Annotation [[Download link](https://drive.google.com/drive/folders/1V86fFrNzK0X1amRXr6ev-gH23gJzpsMB?usp=sharing)]
Download Diverse Weather and their coco annotation and place in the structure as shown.

```
    DATASET/
        /diverseWeather
            /daytime_clear
               /VOC2007
               voc07_test.json
               voc07_train.json
            /daytime_foggy
            /dusk_rainy
            /night_rainy
            /night_sunny

```

## Step1: Source Augmentation

Below is the script for source augmentation:
```
python ./tools/finetune.py 
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml
    --ft-tasks configs/adverse_weather/daytimeclear.yaml 
    --skip-test --custom_shot_and_epoch_and_general_copy 0_200_1 
    --evaluate_only_best_on_test --push_both_val_and_test  
    MODEL.WEIGHT glip_t.pth
    SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 8 
    TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True SOLVER.STEP_PATIENCE 2 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 4 SOLVER.MODEL_EMA 0.0 SOLVER.MAX_EPOCH 1 SOLVER.BASE_LR 0.0001 SOLVER.WEIGHT_DECAY 0.025 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full OUTPUT_DIR OUTPUT

```
Pretrained GLIP-T weights can be download from [[pretrain models](https://huggingface.co/harold/GLIP/tree/main)]

## Step2: Style Transfer

Below is the script for Daytime Sunny to Night sunny:
```
python ./tools/finetune.py 
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml
    --ft-tasks configs/adverse_weather/aug/daytimeclear_nightsunny.yaml 
    --skip-test --custom_shot_and_epoch_and_general_copy 0_200_1 
    --evaluate_only_best_on_test --push_both_val_and_test 
    --f_aug True 
    --f_aug_save_dir 
    ./f_aug
    MODEL.WEIGHT daytimeclear.pth
    SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 8 
    TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True SOLVER.STEP_PATIENCE 2 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 4 SOLVER.MODEL_EMA 0.0 SOLVER.MAX_EPOCH 1 SOLVER.BASE_LR 0.0001 SOLVER.WEIGHT_DECAY 0.025 SOLVER.TUNING_HIGHLEVEL_OVERRIDE language_prompt_v2 OUTPUT_DIR OUTPUT

```


## Step3: Fine-tuning for Generalization

Below is the style script for Daytime Sunny to Night sunny:
```
python ./tools/finetune.py 
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml
    --ft-tasks configs/adverse_weather/aug/daytimeclear_nightsunny.yaml 
    --skip-test --custom_shot_and_epoch_and_general_copy 0_200_1 
    --evaluate_only_best_on_test --push_both_val_and_test 
    --adapt True 
    --f_aug_save_dir ./f_aug/daytimeclear_nightsunny/
    MODEL.WEIGHT checkpoints/glip_t_daytimeclear.pth SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 8 
    TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True SOLVER.STEP_PATIENCE 2 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 4 SOLVER.MODEL_EMA 0.0 SOLVER.MAX_EPOCH 1 SOLVER.BASE_LR 0.0001 SOLVER.WEIGHT_DECAY 0.025 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full OUTPUT_DIR OUTPUT

```

## Acknowledgements
We mainly appreciate for these good projects and their authors' hard-working.
- This work is based on GLIP.
-The implementation of our detector relies on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
- The style transfer method is based on [AdaIN](https://github.com/xunhuang1995/AdaIN-style). 