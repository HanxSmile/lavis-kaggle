from vigc.common.registry import registry

deepseek_r1_system_prompt = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "First thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e. <think> reasoning process here </think><answer> answer here </answer>. "
    "If the user's question is math related, "
    "please directly put your final answer which is a number within <answer> </answer> tags without other words."
)

registry.register_constant("deepseek_r1_system_prompt", deepseek_r1_system_prompt)
