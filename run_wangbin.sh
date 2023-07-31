




# InstructBLIP-7B + LLaVA
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30003 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqga/llava-150k/normal_vqga.yaml