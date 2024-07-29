src_path=/mnt/data/hanxiao/MyCode/lavis-kaggle/vigc/output/whisper_fleurs_new_ja/20240727030
dst_path=/mnt/data/hanxiao/models/audio/hx-asr/models
lang=ja

cd /mnt/data/hanxiao/MyCode/lavis-kaggle

python vigc/projects/asr/convert_ckpt_to_hf.py  --src-path ${src_path}/checkpoint_best.pth  --dst-path ${src_path}/whisper-large-${lang}
cp /mnt/data/hanxiao/models/audio/whisper-large-v3-tokenizer/*  ${src_path}/whisper-large-${lang}

ct2-transformers-converter --model ${src_path}/whisper-large-${lang}  --output_dir ${src_path}/faster-whisper-large-${lang}  --quantization float16

cp /mnt/data/hanxiao/models/audio/whisper-large-v3-tokenizer/*   ${src_path}/faster-whisper-large-${lang}
mv ${src_path}/faster-whisper-large-${lang}  ${dst_path} 