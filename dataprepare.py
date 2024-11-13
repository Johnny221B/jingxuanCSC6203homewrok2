# 导入必要的库
from datasets import load_dataset
import json

# 加载数据集，指定为 "test" 部分
ds = load_dataset("FinGPT/fingpt-fineval", split="test")  # 加载 "test" 部分的数据集

# 初始化一个空列表，用于存储格式化后的数据
output_data = []

# 遍历数据集中的每个项目
for item in ds:
    # 将每个项目的 'instruction'、'input' 和 'output' 提取并格式化为字典
    formatted_item = {
        "instruction": item['instruction'],  # 提取 'instruction' 字段
        "input": item['input'],  # 提取 'input' 字段
        "output": item['output']  # 提取 'output' 字段
    }
    # 将格式化后的字典添加到 output_data 列表中
    output_data.append(formatted_item)

# 设置输出文件的保存路径
output_file_path = '/home/linyuliu/jxdata/LLama-Factory/data/fingpt_test.json'

# 将格式化后的数据保存为 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)  # 写入 JSON 文件，确保非 ASCII 字符正常保存，并进行缩进格式化

# 打印确认消息，告知数据已成功保存
print(f"Data has been successfully saved to {output_file_path}")
