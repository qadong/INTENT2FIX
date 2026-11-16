import json
import re

def transform_json_object(input_data):
    """
    将单个JSON对象从原始格式转换为目标格式。

    Args:
        input_data (dict): 从jsonl文件中读取的单行Python字典。

    Returns:
        dict: 转换后的目标格式字典。
    """
    # 从repo_url中提取 'org/repo'
    repo_url = input_data.get('repo_url', '')
    match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
    repo_slug = ""
    if match:
        repo_slug = match.group(1)
        # 如果URL以.git结尾，去掉它
        if repo_slug.endswith('.git'):
            repo_slug = repo_slug[:-4]

    # 获取其他字段
    number = input_data.get('pull_number')
    issue_numbers = [number] if number is not None else []
    base_commit = input_data.get('issue_base_commit', '')
    fix_patch = input_data.get('golden_patch', '')
    test_patch = input_data.get('test_patch', '')
    problem_statement = input_data.get('issue_description', '')

    # 构建目标字典
    output_data = {
        "repo": repo_slug,
        "pull_number": number,
        "instance_id": input_data.get('instance_id', ''),
        "issue_numbers": issue_numbers,
        "base_commit": base_commit,
        "patch": fix_patch,
        "test_patch": test_patch,
        "problem_statement": problem_statement,
        "hints_text": input_data.get('hints_text', ''),
        "created_at": input_data.get('created_at', '')
    }
    return output_data

def process_jsonl_file(input_file_path, output_file_path):
    """
    读取整个jsonl文件，逐行转换，并将结果写入新的jsonl文件。

    Args:
        input_file_path (str): 输入的jsonl文件路径。
        output_file_path (str): 输出的jsonl文件路径。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                # 跳过空行
                if not line.strip():
                    continue
                
                try:
                    # 1. 读取并解析当前行
                    original_data = json.loads(line)
                    
                    # 2. 转换数据格式
                    transformed_data = transform_json_object(original_data)
                    
                    # 3. 将转换后的字典写回新的jsonl文件
                    outfile.write(json.dumps(transformed_data) + '\n')
                    
                except json.JSONDecodeError:
                    print(f"警告：跳过文件 '{input_file_path}' 中第 {i+1} 行，因为它不是有效的JSON。")
                except Exception as e:
                    print(f"处理第 {i+1} 行时发生意外错误: {e}")

        print(f"处理完成！结果已保存到 '{output_file_path}'。")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file_path}'。请检查文件名和路径是否正确。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


# --- 使用示例 ---
# 请将 'your_input_data.jsonl' 替换为您的实际输入文件名
# 请将 'your_output_data.jsonl' 替换为您希望的输出文件名
input_filename = 'INTENT2FIX/benchmark/build_success.jsonl'
output_filename = 'INTENT2FIX/benchmark/swebench_cpp.jsonl'

# 运行处理函数
process_jsonl_file(input_filename, output_filename)