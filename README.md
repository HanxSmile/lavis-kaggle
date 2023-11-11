# Bengali.AI Speech Recognition 4th place Training code

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n vigc python=3.8
conda activate vigc
```

2. Build from source

```bash
git clone git@github.com:HanxSmile/lavis-kaggle.git -b dev
cd lavis-kaggle
pip install -e .
```

## Train

### Prepare Data

Edit the path info of the competition data in 

1. `vigc/configs/datasets/wav2vec_bengali_asr/concat_filtered_aug_train.yaml`:

```yaml
datasets:
  wav2vec_filtered_concat_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali
#    model_name: arijitx/wav2vec2-xls-r-300m-bengali
    data_type: audios
    musan_dir: /mnt/petrelfs/share_data/hanxiao/musan_bg_noise
    seg_nums: 2
    ratio: 0.7
    build_info:
      annotation: /mnt/petrelfs/hanxiao/working/whisper/baseline0901.csv
      data_root: /mnt/petrelfs/share_data/hanxiao/bengaliai-speech
```

> `model_name`: The tokenizer model path
>
> `musan_dir`: The path of the musan bg noise data
>
> `annotation`: A csv file which stores  competition data's each sample's loss
>
> ```python
> ,id,sentence,input_sec,loss
> 0,98a7c0548aef,ম্যাচপরবর্তী সাক্ষাৎকারে ভাগ্যবান সমর্থকের সেই প্রশ্ন করা হবে তাঁর প্রিয় অধিনায়ককে।,0.828,0.0
> 1,a7dfce973640,তিনি এই বিভাগে অস্কারের জন্য তৃতীয় সর্বকনিষ্ঠ মনোনীত অভিনেতা।,0.756,0.0
> 2,f67690866494,তৎকালীন সময়ে যোদ্ধাদের পা রক্ষার জন্য একটি আদর্শ সরঞ্জাম হিসেবে বিবেচনা করা হতো।,0.936,0.0
> 3,7f02f6b1748f,ঘটনার সময় তাঁর স্ত্রী রাফিদা আহমেদও দুর্বৃত্তদের হাতে গুরুতর আহত হন।,0.7559375,0.0
> ```
>
> `data_root`: The path of the competition data

2. `vigc/configs/datasets/wav2vec_bengali_asr/concat_seg_filtered_aug_train.yaml`

```yaml
datasets:
  wav2vec_filtered_concat_seg_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali
#    model_name: arijitx/wav2vec2-xls-r-300m-bengali
    data_type: audios
    musan_dir: /mnt/petrelfs/share_data/hanxiao/musan_bg_noise
    concat_seg_nums: 2
    split_seg_nums: 3
    ratio: 0.7
    build_info:
      annotation: /mnt/petrelfs/hanxiao/working/whisper/baseline0901.csv
      data_root: /mnt/petrelfs/share_data/hanxiao/bengaliai-speech
```

3. `vigc/configs/datasets/wav2vec_bengali_asr/filtered_seg_aug_train.yaml`

```yaml
datasets:
  wav2vec_filtered_seg_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali
#    model_name: arijitx/wav2vec2-xls-r-300m-bengali
    data_type: audios
    musan_dir: /mnt/petrelfs/share_data/hanxiao/musan_bg_noise
    seg_nums: 3
    ratio: 0.7
    build_info:
      annotation: /mnt/petrelfs/hanxiao/working/whisper/baseline0901.csv
      data_root: /mnt/petrelfs/share_data/hanxiao/bengaliai-speech
```

### Specify the pretrain model in config file

Edit the model path in `vigc/projects/wav2vec_bengali_asr/train_filtered_wav2vec_1b.yaml`

```yaml
model:
  # model_name: /mnt/petrelfs/hanxiao/work/lavis-kaggle/vigc/output/wav2vec_bengali_asr/facebook-1b-finetuned-stage-2/best_hf
  model_name: /mnt/lustre/hanxiao/work/lavis-kaggle/vigc/output/facebook_1b/finetuned-stage3/latest_hf_stage3_0920_dropout_0_8_noise/latest_hf
  load_finetuned: False
  finetuned: ""
  post_process_flag: True
  arch: bengali_1b_wav2vec
  model_type: default
  freeze_encoder: True
  loss_reduction: "sum"
  processor_name: "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali"
#  processor_name: arijitx/wav2vec2-xls-r-300m-bengali

datasets:
  # train
  wav2vec_filtered_seg_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-large-mms-1b-bengali
    sample_ratio: 1
    seg_nums: 3
    ratio: 0.7

  wav2vec_filtered_concat_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-large-mms-1b-bengali
    sample_ratio: 1
    seg_nums: 2
    ratio: 0.7

  wav2vec_filtered_concat_seg_aug_asr:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-large-mms-1b-bengali
    sample_ratio: 1
    concat_seg_nums: 2
    split_seg_nums: 3
    ratio: 0.7

  # test
  wav2vec_bengali_asr_test:
    model_name: /mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-large-mms-1b-bengali

run:
  runner: runner_iter
  task: whisper_bengali_asr_task
  # optimizer
  lr_sched: "linear_warmup_cosine_3_long_tail_lr"
  # init_lr: 3e-5
  # min_lr: 1e-5
  # warmup_lr: 1e-6

  init_lr: 1e-5
  min_lr: 5e-6
  warmup_lr: 1e-6

  weight_decay: 0.05

  batch_size_train: 4
  batch_size_eval: 4
  accum_grad_iters: 2

  num_workers: 4
  warmup_steps: 1000

  iters_per_inner_epoch: 1000
  max_iters: 120000
  # max_iters: 240000

  seed: 210
  output_dir: "./output/facebook_1b/finetuned-stage4"

  amp: True
  resume_ckpt_path: null

  evaluate: False
  train_splits: [ "train" ]
  valid_splits: [ "eval" ]

  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
```

> `model_name`: The path of the pretrain model
>
> * In stage 1: model_name is the path of  [facebook/wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b) 
> * In stage 2: model_name is the path of latest checkpoint of stage 1 training
> * In stage 3: model_name is the path of latest checkpoint of stage 2 training
>
> `processor_name`: the path of `arijitx/wav2vec2-xls-r-300m-bengali`

### Train Script

```bash
torchrun --nproc-per-node 8 --master_port 29511 train.py --cfg-path vigc/projects/wav2vec_bengali_asr/train_filtered_wav2vec_1b.yaml
```

