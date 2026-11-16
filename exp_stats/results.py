import json
import re
from pathlib import Path

# --- 配置 ---
# results_dir = Path('results_openrouter_anthropic_claude-sonnet-4.5')
results_dir = Path('results_c_deepseek_deepseek-chat_cotest')
# 输出文件
origin_output_file = 'c_deepseek_deepseek-chat_cotest_origin_patch.jsonl'
enhanced_output_file = 'c_deepseek_deepseek-chat_cotest_enhanced_patch.jsonl'

# --- 补丁过滤逻辑 (V6) ---

def filter_test_patches(patch_content: str) -> str:
    """
    解析一个 diff 补丁字符串，使用 '--- a/' 或 '--- /dev/null' 作为文件分隔符，
    并应用所有过滤规则。
    """
    if not patch_content:
        return ""

    kept_hunks = []
    
    # 1. 分割补丁 (V5 逻辑)
    file_diffs = re.split(r'(--- (?:a/|/dev/null)\S*)', patch_content)
    
    if not file_diffs:
        return ""

    for i in range(1, len(file_diffs), 2):
        if (i + 1) >= len(file_diffs):
            continue
            
        separator_line = file_diffs[i]
        diff_content = file_diffs[i+1]
        full_hunk = separator_line + diff_content
        
        # 2. 获取路径 (V5 逻辑)
        path_to_check_str = None
        path_match = re.search(r'--- a/(\S+)', separator_line)
        
        if path_match:
            path_to_check_str = path_match.group(1).strip()
        elif separator_line.startswith('--- /dev/null'):
            b_path_match = re.search(r'\+\+\+ b/(\S+)', diff_content)
            if b_path_match:
                path_to_check_str = b_path_match.group(1).strip()
        
        if not path_to_check_str:
            kept_hunks.append(full_hunk) # 无法识别，保留
            continue
        
        # 3. 按 / 分割路径
        path_obj = Path(path_to_check_str)
        is_test_file = False

        # --- V6 过滤规则 ---
        
        filename_lower = path_obj.name.lower()
        filename_stem_lower = path_obj.stem.lower() # 文件名 (不含扩展名)
        parent_parts_lower = [p.lower() for p in path_obj.parent.parts]

        # 1. 检查父目录是否包含精确的 "test" 或 "tests" 目录
        test_dir_names = {'test', 'tests', 'testing'}
        if any(part.lower() in test_dir_names for part in parent_parts_lower):
            is_test_file = True
        
        # 2. 检查文件名前缀 (必须是 test_ 等开头 或 reproduction POC 前缀)
        elif any(filename_lower.startswith(prefix) for prefix in ['test_', 'test-', 'unit_', 'unit-', 'unittest', 'bugrepro', 'repro', 'bug_repro', 'bug_']):
            is_test_file = True

        # 3. 检查文件名后缀 (在扩展名之前，例如 something_test.cpp 或 something.tests.cpp)
        elif any(filename_stem_lower.endswith(suffix) for suffix in ['_test', '.test', '_tests', '.tests']):
            is_test_file = True
            
        # 4. 特殊情况: 文件名本身就是 test 或 tests
        elif filename_stem_lower in ['test', 'tests']:
            is_test_file = True
        
        # ---------------------
        
        # 4. 过滤 & 重组
        if is_test_file:
            pass
        else:
            kept_hunks.append(full_hunk)

    return "".join(kept_hunks)


# --- 主脚本 (已修改为仅提取目标实例) ---

def main():

    target_instances_list = [
        "catchorg__Catch2-1422",
        "catchorg__Catch2-1448",
        "catchorg__Catch2-1608",
        "catchorg__Catch2-1609",
        "catchorg__Catch2-1614",
        "catchorg__Catch2-1616",
        "catchorg__Catch2-2128",
        "catchorg__Catch2-2187",
        "catchorg__Catch2-2288",
        "catchorg__Catch2-2394",
        "catchorg__Catch2-2723",
        "fmtlib__fmt-493",
        "fmtlib__fmt-967",
        "fmtlib__fmt-974",
        "fmtlib__fmt-1171",
        "fmtlib__fmt-1361",
        "fmtlib__fmt-1390",
        "fmtlib__fmt-1407",
        "fmtlib__fmt-1663",
        "fmtlib__fmt-1837",
        "fmtlib__fmt-2070",
        "fmtlib__fmt-2158",
        "fmtlib__fmt-2292",
        "fmtlib__fmt-2310",
        "fmtlib__fmt-2317",
        "fmtlib__fmt-2394",
        "fmtlib__fmt-2610",
        "fmtlib__fmt-2940",
        "fmtlib__fmt-3158",
        "fmtlib__fmt-3248",
        "fmtlib__fmt-3260",
        "fmtlib__fmt-3271",
        "fmtlib__fmt-3272",
        "fmtlib__fmt-3279",
        "fmtlib__fmt-3537",
    ]
    # 转换为 Set 以获得更快的查找速度
    # target_instances_set = set(target_instances_list)
    # -----------------------------------
    
    origin_data = []
    enhanced_data = []
    origin_empty_patch_count = 0
    enhanced_empty_patch_count = 0
    processed_count = 0
    skipped_count = 0
    origin_filtered_to_empty_list = []
    enhanced_filtered_to_empty_list = []
    skipped_non_target = 0 # 计数器：跳过了多少非目标实例

    if not results_dir.is_dir():
        print(f"错误: 目录 '{results_dir}' 未找到。")
        return

    print(f"正在处理 '{results_dir}' 目录...")
    # print(f"目标：仅提取 {len(target_instances_set)} 个特定实例。")

    for sub_dir in results_dir.iterdir():
        if not sub_dir.is_dir():
            continue

        instance_id = sub_dir.name
        
        # --- ★ 新增：检查实例是否在目标列表中 ---
        # if instance_id not in target_instances_set:
        #     skipped_non_target += 1
        #     continue  # 如果不是目标实例，则跳过
        # -----------------------------------
        
        # 1. 解析 instance_id
        org_part, repo_part, number_part = None, None, None
        try:
            parts = instance_id.split('__', 1)
            if len(parts) != 2: raise ValueError("缺少 '__' 分隔符")
            org_part = parts[0]
            repo_number_part = parts[1]
            
            repo_parts = repo_number_part.rsplit('-', 1)
            if len(repo_parts) != 2: raise ValueError("缺少 '-' number 分隔符")
            repo_part = repo_parts[0]
            number_part = repo_parts[1]
            if not number_part.isdigit(): raise ValueError("number 部分不是数字")

        except ValueError as e:
            print(f"警告: 目标文件夹 '{instance_id}' 命名格式不符 'org__repo-number' (错误: {e})，已跳过。")
            skipped_count += 1
            continue
        
        # 2. 判断文件存在情况并读取 patch 文件
        solution_patch_path = sub_dir / 'solution.patch'
        solution_before_path = sub_dir / 'solution_before_iteration.patch'
        enhanced_iter_path = sub_dir / 'enhanced_solution_iter1.patch'
        repaired_iter_path = sub_dir / 'repaired_solution_iter1.patch'
        
        has_solution = solution_patch_path.is_file()
        has_solution_before = solution_before_path.is_file()
        has_enhanced_iter = enhanced_iter_path.is_file()
        has_repaired_iter = repaired_iter_path.is_file()
        
        # --- 确定 origin 和 enhanced 的文件路径 ---
        
        # Origin 优先使用 solution_before_iteration.patch
        if has_solution_before:
            origin_patch_file_path = solution_before_path
        elif has_solution:
            origin_patch_file_path = solution_patch_path
        else:
            origin_patch_file_path = None

        # Enhanced 优先级：repaired > enhanced_iter > solution.patch
        if has_repaired_iter:
            enhanced_patch_file_path = repaired_iter_path
        elif has_enhanced_iter:
            enhanced_patch_file_path = enhanced_iter_path
        elif has_solution:
            enhanced_patch_file_path = solution_patch_path
        else:
            enhanced_patch_file_path = None

        if not origin_patch_file_path and not enhanced_patch_file_path:
            print(f"警告: 实例 '{instance_id}' 中未找到任何 patch 文件，已跳过。")
            skipped_count += 1
            continue
        
        # 3. 读取和过滤原始 patch 文件
        origin_fix_patch = ""
        origin_original_was_not_empty = False 

        if origin_patch_file_path:
            try:
                raw_patch_content = origin_patch_file_path.read_text(encoding='utf-8')
                origin_original_was_not_empty = bool(raw_patch_content.strip())
                origin_fix_patch = filter_test_patches(raw_patch_content)
                if not origin_fix_patch.strip():
                    origin_fix_patch = ""
            except Exception as e:
                print(f"警告: 无法读取或过滤文件 {origin_patch_file_path}。错误: {e}")
        
        # 4. 读取和过滤增强 patch 文件
        enhanced_fix_patch = ""
        enhanced_original_was_not_empty = False

        if enhanced_patch_file_path:
            try:
                raw_patch_content = enhanced_patch_file_path.read_text(encoding='utf-8')
                enhanced_original_was_not_empty = bool(raw_patch_content.strip())
                enhanced_fix_patch = filter_test_patches(raw_patch_content)
                if not enhanced_fix_patch.strip():
                    enhanced_fix_patch = ""
            except Exception as e:
                print(f"警告: 无法读取或过滤文件 {enhanced_patch_file_path}。错误: {e}")
        
        # 5. 统计和构建原始 patch 条目
        origin_filtered_is_empty = (origin_fix_patch == "")

        if origin_original_was_not_empty and origin_filtered_is_empty:
            origin_filtered_to_empty_list.append(instance_id)

        if origin_filtered_is_empty:
            origin_empty_patch_count += 1
        
        origin_entry = {
            "org": org_part,
            "repo": repo_part,
            "number": number_part,
            "fix_patch": origin_fix_patch
        }
        
        origin_data.append(origin_entry)
        
        # 6. 统计和构建增强 patch 条目（总是存在）
        enhanced_filtered_is_empty = (enhanced_fix_patch == "")

        if enhanced_original_was_not_empty and enhanced_filtered_is_empty:
            enhanced_filtered_to_empty_list.append(instance_id)

        if enhanced_filtered_is_empty:
            enhanced_empty_patch_count += 1
        
        enhanced_entry = {
            "org": org_part,
            "repo": repo_part,
            "number": number_part,
            "fix_patch": enhanced_fix_patch
        }
        
        enhanced_data.append(enhanced_entry)
        
        processed_count += 1

    # 4. 写入 JSONL 文件
    if processed_count == 0:
        print(f"\n警告: 在 '{results_dir}' 中没有找到任何目标实例文件夹。")
        if skipped_count > 0:
            print(f"(共跳过了 {skipped_count} 个格式不符的目标文件夹)")
        return

    # 写入原始 patch 文件
    try:
        with open(origin_output_file, 'w', encoding='utf-8') as f:
            for entry in origin_data:
                json_string = json.dumps(entry, ensure_ascii=False)
                f.write(json_string + '\n')
        
        print(f"\n原始补丁处理完成！")
        print(f"结果已保存到: {origin_output_file}")
        print(f"---")
        print(f"原始补丁统计:")
        print(f"  成功处理: {processed_count} 个目标实例。")
        print(f"  跳过 (非目标): {skipped_non_target} 个文件夹。")
        print(f"  跳过 (格式错误): {skipped_count} 个目标文件夹。")
        print(f"  'fix_patch' 为空: {origin_empty_patch_count} 个 (包含原始为空或过滤后变为空)")
        
        if origin_filtered_to_empty_list:
            print(f"---")
            print(f"[调试信息] 以下 {len(origin_filtered_to_empty_list)} 个实例因仅包含测试修改，过滤后 'fix_patch' 变为空:")
            for instance_name in origin_filtered_to_empty_list:
                print(f"  - {instance_name}")

    except Exception as e:
        print(f"错误: 写入 JSONL 文件 {origin_output_file} 失败。错误: {e}")

    # 写入增强 patch 文件
    try:
        with open(enhanced_output_file, 'w', encoding='utf-8') as f:
            for entry in enhanced_data:
                json_string = json.dumps(entry, ensure_ascii=False)
                f.write(json_string + '\n')
        
        print(f"\n增强补丁处理完成！")
        print(f"结果已保存到: {enhanced_output_file}")
        print(f"---")
        print(f"增强补丁统计:")
        print(f"  成功处理: {len(enhanced_data)} 个目标实例。")
        print(f"  'fix_patch' 为空: {enhanced_empty_patch_count} 个 (包含原始为空或过滤后变为空)")
        
        if enhanced_filtered_to_empty_list:
            print(f"---")
            print(f"[调试信息] 以下 {len(enhanced_filtered_to_empty_list)} 个实例因仅包含测试修改，过滤后 'fix_patch' 变为空:")
            for instance_name in enhanced_filtered_to_empty_list:
                print(f"  - {instance_name}")

    except Exception as e:
        print(f"错误: 写入 JSONL 文件 {enhanced_output_file} 失败。错误: {e}")

if __name__ == "__main__":
    main()
