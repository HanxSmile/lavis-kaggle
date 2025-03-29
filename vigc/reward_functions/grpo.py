import re
from vigc.common.registry import registry


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text):
    reward = 0
    if text.count("<think>") == 1:
        reward += 0.125

    if text.count("</think>") == 1:
        reward += 0.125

    if text.count("<answer>") == 1:
        reward += 0.125

    if text.count("</answer>") == 1:
        reward += 0.125
    return reward


@registry.register_reward_func("correctness_reward")
def correctness_reward(model_input, model_output, response, **kwargs):
    response = extract_answer(response)
    reward = 0.0
    if str(model_output).strip() == response.strip():
        reward = 2.0
    return reward


# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
@registry.register_reward_func("digit_reward")
def digit_reward(model_input, model_output, response, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    if not re.match(pattern, response.strip()):
        return 0.0
    response = extract_answer(response)
    pattern = r"^-?(0|([1-9][0-9]*))(\.[\d]+)?$"
    reward = 0.0
    if re.match(pattern, response):
        reward = 0.5
    return reward


# 格式奖励
@registry.register_reward_func("hard_format_reward")
def hard_format_reward(model_input, model_output, response, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    reward = 0.0
    if re.match(pattern, response):
        reward = 0.5
    return reward


# 格式奖励
@registry.register_reward_func("soft_format_reward")
def soft_format_reward(model_input, model_output, response, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    reward = 0.0
    if re.match(pattern, response):
        reward = 0.5
    return reward


# 标记奖励（改善格式奖励稀疏问题）
@registry.register_reward_func("mark_reward")
def mark_reward(model_input, model_output, response, **kwargs):
    return mark_num(response)
