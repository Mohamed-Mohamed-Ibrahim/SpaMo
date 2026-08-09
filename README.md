# Adaptive EASLT

## Introduction

This repository adds two components on top of the published EASLT architecture — dynamic temporal segmentation with adaptive masking, and discriminative learning-rate groups for optimization.


## Environment

Install dependencies using:
```bash
pip install -r requirements.txt
```


## Data Preparation

We validate our method on Phoenix-2014T dataset:
- [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)


### Feature Streams

Adaptive EASLT utilizes three complementary feature streams:
1. **Spatial Features**: Extracted with a CLIP ViT model to capture static visual configuration (hand shapes, body posture).
2. **Motion Features**: Extracted with VideoMAE to capture temporal kinematic dynamics.
3. **Emotion Features**: Extracted with a ViT fine-tuned on FER2013 over detected facial ROIs, to capture continuous affective state.
    
#### Extracting Spatial Features

To extract spatial features using the CLIP ViT model:

```bash
python scripts/vit_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name openai/clip-vit-large-patch14 \
    --video_root /PATH/TO/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
    --cache_dir /PATH/TO/CACHE_DIR \
    --save_dir /PATH/TO/SAVE_DIR \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 32 \
    --device cuda:0
```

Key parameters:
- `--model_name`: CLIP ViT model variant (default: openai/clip-vit-large-patch14)
- `--s2_mode`: Use "s2wrapping" for multi-scale feature extraction
- `--scales`: Scales for multi-scale feature extraction (default: 1 2)

#### Extracting Motion Features

To extract motion features using VideoMAE:

```bash
python scripts/mae_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name MCG-NJU/videomae-large \
    --video_root /PATH/TO/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
    --cache_dir /PATH/TO/CACHE_DIR \
    --save_dir /PATH/TO/SAVE_DIR \
    --overlap_size 8 \
    --batch_size 32 \
    --device cuda:0
```

#### Extracting Emotion Features

To extract emotion features using a ViT pretrained on the FER2013 dataset:

```bash
python ./scripts/emotion_feature_extractor.py \
  --input_dir /path/to/input_videos \
  --output_dir /path/to/output_features \
  --batch_size 32 \
  --stride 8
```


For convenience, you can download our pre-extracted features from [here](https://www.kaggle.com/datasets/muhammad5286/adaptive-easlt-phoenix14t).

You can access the extracted How2Sign spatial and motion features from here:
* **[Spatial Features](https://www.kaggle.com/datasets/mohamedmoibrahim/spamo-how2sign-spatial-features)**
* **[Motion Features](https://www.kaggle.com/datasets/muhammad5286/spamo-how2sign-motion-features)**


## Model Training and Evaluation

### Training

Train the Adaptive EASLT model with:

```bash
python main.py -c configs/finetune.yaml -e bleu
```

### Evaluation

Evaluate a trained model using:

```bash
python main.py -c configs/finetune.yaml -e bleu --train False --test True --ckpt /PATH/TO/CHECKPOINT
```

Replace `/PATH/TO/CHECKPOINT` with your model checkpoint path.
Pre-trained checkpoints are available for download [here](https://www.kaggle.com/datasets/muhammad5286/adaptive-easlt-phoenix14t).


## 📄 Project Report

For an in-depth look at the architecture, methodology, and results, please read the full [Graduation Project Report](./docs/Sign%20Language%20Translation%20Graduation%20Project%20Report.pdf).


## Citation

```bash
@inproceedings{hwang2025efficient,
  title={An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs},
  author={Hwang, Eui Jun and Cho, Sukmin and Lee, Junmyeong and Park, Jong C},
  booktitle={NAACL},
  year={2025}
}
```
