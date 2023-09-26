PyTorch  code for ['Weakly Supervised Referring Expression Grounding via Adaptive Curriculum Self-Distillation']. This paper is submitted to IEEE Transactions on Multimedia.

### Preliminary
1. Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install mask-faster-rcnn, REFER and refer-parser2. Follow Step 1 & 2 in Training to prepare the data and features.

2. Please follow the step in [DTWREG](https://github.com/insomnia94/DTWREG) to acquire the parsed discriminative triads.

The experiments are conducted on one GPU (NVIDIA RTX A6000).

- python == 3.7.13
- pytorch == 1.10

### Feature Encoding
1. follow the feature extraction in our previous project (https://github.com/dami23/WREG_KD)


### Training and evaluation
1. training

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/train_model.py --dataset {DATASET} --splitBy {SPLITBY} --exp_id {EXP_ID}


2. evaluation

   CUDA_VISIBLE_DEVICES={GPU_ID} python ./tools/eval.py --dataset {DATASET} --splitBy {SPLITBY} --split {SPLIT} --id {EXP_ID}

   {DATASET} = refcoco, refcoco+, refcocog. {SPLITBY} = unc for refcoco and refcoco+, google for refcocog.

   The acquired results with different settings are listed in output/easy_results.txt

### Pretrained Models
All trained models by the proposed approach can be downloaded [here](https://drive.google.com/drive/folders/1oFfVtYDX2CStJyPsS9q3vUtpi5g6ev2c?usp=drive_link).
