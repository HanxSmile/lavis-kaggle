nohup srun -p mllm_exp --gres=gpu:2 --quotatype=reserved  accelerate launch --multi_gpu --num_processes=2 albef_baseline.py >> /mnt/petrelfs/hanxiao/input/ASL/logs/train.log  2>&1 &

nohup srun -p mllm_exp --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 train.py --cfg-path /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/train/vicuna7b_train.yaml  >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/train4.log  2>&1 &


nohup srun -p mllm_exp2 --gres=gpu:1 --quotatype=reserved  torchrun --nproc-per-node 1 --master_port 29510 train.py --cfg-path /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/train/vicuna7b_finetune_linear.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/train4.log  2>&1 &



nohup srun -p mllm_exp2 --gres=gpu:8 --quotatype=reserved  torchrun --nproc-per-node 8 --master_port 29511 train.py --cfg-path /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/train/vicuna7b_train_lora.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/train_lora_16k.log  2>&1 &

nohup srun -p mllm_exp --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29511 evaluate.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/eval/vicuna7b_finetune_linear_caption.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/evaluate_linear.log  2>&1 &

nohup srun -p mllm_exp --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29511 evaluate.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/eval/vicuna7b_finetune_lora_caption.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/evaluate_lora.log  2>&1 &

nohup srun -p mllm_exp --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29511 evaluate.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/eval/vicuna7b_pretrained_caption.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/evaluate_pretrained.log  2>&1 &


nohup srun -p mllm_exp --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29512 train.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/train/vicuna7b_mixup_train_eval.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/mixup_3.log  2>&1 &

nohup srun -p llmeval2 --gres=gpu:8 --quotatype=reserved  torchrun --nproc-per-node 8 --master_port 29514 train.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/earlier_stage/vicuna7b_a_okvqa_vqa.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/log/vicuna7b-a-okvqa-baseline.log  2>&1 &

nohup srun -p llmeval2 --gres=gpu:8 --quotatype=spot  torchrun --nproc-per-node 8 --master_port 29513 evaluate.py --cfg-path  /mnt/petrelfs/hanxiao/work/lavis/runtime_configs/instruct_blip/eval/vqa_eval/vicuna7b_instruct_blip_vqg.yaml >> /mnt/petrelfs/hanxiao/input/instruct-blip/output/coco_2017_generate.log  2>&1 &