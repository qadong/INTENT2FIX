import json
import re
import os
import sys

def extract_files_from_patch(patch_text):
    if not patch_text:
        return set()
    # 查找 --- a/path 或 +++ b/path
    files = set()
    # 也可以使用 re.findall(r'^--- a/(.*)$', patch_text, re.MULTILINE)
    # 或者是 re.findall(r'^\+\+\+ b/(.*)$', patch_text, re.MULTILINE)
    # 我们通常关注 +++ b/ 后的路径，因为它代表修改后的文件
    matches = re.findall(r'^\+\+\+ [ab]/(.*)$', patch_text, re.MULTILINE)
    for m in matches:
        # 去掉可能存在的 \t 后的内容
        path = m.split('\t')[0].strip()
        if path != "/dev/null":
            files.add(path)
    
    # 有些 patch 可能是 git diff 格式，包含 diff --git a/path b/path
    matches_git = re.findall(r'^diff --git a/(.*) b/(.*)$', patch_text, re.MULTILINE)
    for m in matches_git:
        files.add(m[0].strip())
        files.add(m[1].strip())
        
    return files

def filter_code_files(files):
    """过滤掉文档文件，只保留代码相关的文件"""
    code_files = set()
    for f in files:
        # 过滤掉 .md 和 .txt 文件
        if not (f.endswith('.md') or f.endswith('.txt')):
            code_files.add(f)
    return code_files

def is_comment_only_line(line):
    """判断一行是否只是注释（去除前后空白后）"""
    stripped = line.strip()
    if not stripped:
        return True  # 空行也算注释相关
    
    # C/C++ 单行注释
    if stripped.startswith('//'):
        return True
    
    # C/C++ 多行注释的开始或结束
    if stripped.startswith('/*') or stripped.startswith('*/') or stripped.endswith('*/'):
        return True
    
    # 多行注释中间的内容（只包含 * 和空白）
    if stripped.startswith('*') and all(c in ' *\t' for c in stripped[1:]):
        return True
    
    # 检查是否整行都在注释块内（以 * 开头，后面是空白或注释内容）
    if stripped.startswith('*'):
        return True
    
    return False

def is_file_comment_only(patch_text, file_path):
    """检查某个文件在 patch 中是否只修改了注释"""
    if not patch_text:
        return False
    
    # 找到该文件对应的 diff 块
    lines = patch_text.split('\n')
    in_target_file = False
    has_code_change = False
    
    # 标准化文件路径用于匹配
    file_path_normalized = file_path.replace('\\', '/')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是文件头
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            # 提取路径
            parts = line.split('\t')
            if len(parts) > 0:
                path_part = parts[0]
                if path_part.startswith('--- a/'):
                    path = path_part[6:].strip()
                elif path_part.startswith('+++ b/'):
                    path = path_part[6:].strip()
                else:
                    path = path_part.strip()
                
                path_normalized = path.replace('\\', '/')
                
                # 检查是否匹配目标文件
                if (path_normalized == file_path_normalized or 
                    path_normalized.endswith('/' + file_path_normalized) or
                    file_path_normalized.endswith('/' + path_normalized)):
                    in_target_file = True
                elif in_target_file:
                    # 遇到新文件，结束当前文件的检查
                    break
        
        # 如果在目标文件的 diff 块中
        if in_target_file:
            # 遇到新的 diff 块或新文件，结束当前文件
            if line.startswith('diff --git'):
                break
            
            # 跳过 hunk 头（@@ -x,y +a,b @@）
            if line.startswith('@@'):
                i += 1
                continue
            
            # 检查修改的行（以 - 或 + 开头，但不是 --- 或 +++）
            if line.startswith('-') and not line.startswith('---'):
                # 删除的行
                content = line[1:].strip()
                if content and not is_comment_only_line(content):
                    has_code_change = True
                    break
            elif line.startswith('+') and not line.startswith('+++'):
                # 添加的行
                content = line[1:].strip()
                if content and not is_comment_only_line(content):
                    has_code_change = True
                    break
        
        i += 1
    
    # 如果没有代码修改，说明只修改了注释
    return not has_code_change

def filter_comment_only_files(patch_text, files):
    """过滤掉只修改了注释的文件"""
    code_files = set()
    for f in files:
        if not is_file_comment_only(patch_text, f):
            code_files.add(f)
    return code_files

def check_localization(golden_jsonl, predicted_jsonl):
    golden_data = {}
    with open(golden_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            inst_id = data.get('instance_id')
            if not inst_id:
                # 尝试构造 inst_id
                org = data.get('org')
                repo = data.get('repo', '').split('/')[-1]
                number = data.get('number') or data.get('pull_number')
                inst_id = f"{org}__{repo}-{number}"
            
            patch = data.get('patch', '')
            # 提取文件并过滤掉 .md 和 .txt 文档文件
            all_files = extract_files_from_patch(patch)
            code_files = filter_code_files(all_files)
            # 进一步过滤掉只修改了注释的文件
            golden_data[inst_id] = filter_comment_only_files(patch, code_files)

    results = []
    with open(predicted_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            
            org = data.get('org', 'unknown')
            repo = data.get('repo', 'unknown').split('/')[-1]
            number = data.get('number', 'unknown')
            inst_id = f"{org}__{repo}-{number}"
            
            fix_patch = data.get('fix_patch', '')
            predicted_files = extract_files_from_patch(fix_patch)
            
            golden_files = golden_data.get(inst_id, set())
            
            if not golden_files:
                # 如果没找到对应的 golden patch，尝试模糊匹配
                # 比如有些 repo 名可能不完全一致
                found = False
                for g_id, g_files in golden_data.items():
                    if inst_id in g_id or g_id in inst_id:
                        golden_files = g_files
                        found = True
                        break
                if not found:
                    results.append({
                        'inst_id': inst_id,
                        'status': 'GOLDEN_NOT_FOUND',
                        'predicted_files': list(predicted_files),
                        'golden_files': []
                    })
                    continue

            # 检查是否包含了所有正确的文件
            missing_files = golden_files - predicted_files
            extra_files = predicted_files - golden_files
            
            if not missing_files:
                status = 'CORRECT' if not extra_files else 'CORRECT_WITH_EXTRA'
            else:
                status = 'INCORRECT' if not (predicted_files & golden_files) else 'PARTIAL'
                
            results.append({
                'inst_id': inst_id,
                'status': status,
                'predicted_files': list(predicted_files),
                'golden_files': list(golden_files),
                'missing_files': list(missing_files),
                'extra_files': list(extra_files)
            })

    # 输出统计
    output_lines = []
    header = f"{'Instance ID':<40} | {'Status':<20} | {'Missing Files'}"
    separator = "-" * 120
    
    output_lines.append("补丁文件定位准确性分析报告")
    output_lines.append("=" * 120)
    output_lines.append(header)
    output_lines.append(separator)
    
    stats = {}
    
    # 按照状态排序，方便查阅
    results.sort(key=lambda x: (x['status'], x['inst_id']))

    for res in results:
        status = res['status']
        stats[status] = stats.get(status, 0) + 1
        missing = ", ".join(res.get('missing_files', []))
        line = f"{res['inst_id']:<40} | {status:<20} | {missing}"
        output_lines.append(line)

    output_lines.append("\n" + "=" * 120)
    output_lines.append("统计摘要:")
    total = sum(stats.values())
    for status in sorted(stats.keys()):
        count = stats[status]
        percentage = (count / total) * 100 if total > 0 else 0
        output_lines.append(f"{status:<20}: {count:>3} ({percentage:>6.2f}%)")
    output_lines.append(f"{'TOTAL':<20}: {total:>3}")
    output_lines.append("=" * 120)

    # 打印到控制台
    for line in output_lines:
        print(line)
        
    # 写入文件
    output_file = "patch_localization_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print(f"\n报告已保存至: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    golden = "./multi_INTENT2FIX/INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"
    predicted = "./multi_INTENT2FIX/gpt_patch.jsonl"
    
    if len(sys.argv) > 1:
        predicted = sys.argv[1]
    
    check_localization(golden, predicted)

