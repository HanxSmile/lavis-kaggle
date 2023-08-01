# Train 
# MiniGPT4-7B + LLaVA
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30001 train.py  --cfg-path vigc/projects/mini_gpt4_vicuna7b/vqga/llava-150k/normal_vqga.yaml
# MiniGPT4-13B + LLaVA
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30002 train.py  --cfg-path vigc/projects/mini_gpt4_vicuna13b/vqga/llava-150k/normal_vqga.yaml

# InstructBLIP-7B + LLaVA
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30003 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqga/llava-150k/normal_vqga.yaml
# InstructBLIP-13B + LLaVA
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30004 train.py  --cfg-path vigc/projects/instruct_blip_vicuna13b/vqga/llava-150k/normal_vqga.yaml



# Evaluation MiniGPT4-7B COCO
# minigpt4 conv
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30005 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_conv.yaml
# minigpt4 detail
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30006 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_detail.yaml
# minigpt4 complex
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30007 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_complex.yaml


# Evaluation MiniGPT4-13B COCO
# minigpt4 conv
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30008 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_conv.yaml
# minigpt4 detail
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30009 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_detail.yaml
# minigpt4 complex
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30010 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_complex.yaml


# Evaluation MiniGPT4-7B Object365
# minigpt4 conv
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30011 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_conv.yaml
# minigpt4 detail
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30012 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_detail.yaml
# minigpt4 complex
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30013 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_complex.yaml


# Evaluation MiniGPT4-13B Object365
# minigpt4 conv
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30014 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_conv.yaml
# minigpt4 detail
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30015 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_detail.yaml
# minigpt4 complex
srun -p bigdata --gres=gpu:8 python -m torch.distributed.run --nproc_per_node=8 --master_port=30016 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_complex.yaml