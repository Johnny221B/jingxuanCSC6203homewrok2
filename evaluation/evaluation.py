import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

# 设置 OpenAI API 密钥
openai.api_key = "your_openai_api_key"

# 读取Qwen模型的权重
qwen_model_path = "/home/linyuliu/.cache/modelscope/hub/Qwen/Qwen-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path)

# 加载测试集数据
ds = load_dataset("FinGPT/fingpt-fineval", split="test")

# 定义一个函数来调用ChatGPT-4o API
def get_chatgpt_response(instruction, input_text):
    prompt = f"{instruction} {input_text}"
    response = openai.Completion.create(
        model="gpt-4",  # 使用 ChatGPT-4o
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# 定义一个函数来使用Qwen模型进行推理
def get_qwen_response(instruction, input_text):
    input_text = f"{instruction} {input_text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = qwen_model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 定义函数来计算正确率
def evaluate_models():
    correct_count = 0
    total_count = len(ds)

    # 遍历数据集
    for item in ds:
        instruction = item['instruction']
        input_text = item['input']
        correct_answer = item['output']

        # 获取ChatGPT的回答
        chatgpt_answer = get_chatgpt_response(instruction, input_text)

        # 获取Qwen的回答
        qwen_answer = get_qwen_response(instruction, input_text)

        # 比较ChatGPT的回答和正确答案
        if chatgpt_answer == correct_answer:
            correct_count += 1

        # 比较Qwen的回答和正确答案
        if qwen_answer == correct_answer:
            correct_count += 1

    # 计算正确率
    accuracy = correct_count / (total_count * 2)  # 因为我们同时比较了ChatGPT和Qwen的答案
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
# 执行评估
evaluate_models()
