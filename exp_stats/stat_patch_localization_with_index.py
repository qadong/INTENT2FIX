#!/usr/bin/env python3
"""
使用函数索引（functions.jsonl）来准确匹配patch中修改的函数
基于行号和文件路径进行匹配，而不是正则表达式
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
from collections import defaultdict

# 导入原有的函数
from stat_patch_localization import (
    extract_files_from_patch,
    normalize_instance_id,
    calculate_file_accuracy,
    calculate_function_accuracy,
    normalize_file_path,
    is_source_code_file
)

# 索引缓存：避免重复加载同一个instance的索引
_FUNCTION_INDEX_CACHE: Dict[str, Dict[str, List[Tuple[int, int, str, str]]]] = {}
_CLASS_INDEX_CACHE: Dict[str, Dict[str, List[Tuple[int, int, str, str]]]] = {}


def load_function_index(instance_id: str, index_base_dir: Optional[str] = None, use_cache: bool = True) -> Dict[str, Dict]:
    """
    加载指定instance的函数索引
    
    Args:
        instance_id: 实例ID，例如 "catchorg__Catch2-1422"
        index_base_dir: 索引基础目录，如果为None，尝试从环境变量或默认路径查找
        use_cache: 是否使用缓存
    
    Returns:
        {file_path: [(start_line, end_line, func_id, func_name), ...]}
        文件路径 -> 函数列表（按行号排序）
    """
    # 检查缓存
    cache_key = f"{instance_id}_{index_base_dir}"
    if use_cache and cache_key in _FUNCTION_INDEX_CACHE:
        return _FUNCTION_INDEX_CACHE[cache_key]
    
    if index_base_dir is None:
        # 尝试从环境变量获取
        index_base_dir = os.environ.get("INDEX_STORE_DIR")
        if not index_base_dir:
            # 尝试默认路径
            possible_paths = [
                "./cache/INTENT2FIX/index_results",
                "./INTENT2FIX/index",
                "./INTENT2FIX/code_intent/index",
                Path(__file__).parent / "index",
            ]
            for p in possible_paths:
                p = Path(p)
                if p.exists():
                    index_base_dir = str(p)
                    break
        
        if not index_base_dir:
            # 如果还是找不到，返回空索引而不是抛出异常
            print(f"警告: 无法找到索引目录。请设置 INDEX_STORE_DIR 环境变量。", file=sys.stderr)
            return {}
    
    index_base_dir = Path(index_base_dir)
    
    # 尝试多个可能的实例ID格式
    instance_variants = [
        instance_id,  # 原始格式: catchorg__Catch2-1422
        f"swe-bench_{instance_id}",  # 带前缀格式: swe-bench_catchorg__Catch2-1422
    ]
    
    # 尝试多个可能的路径
    possible_index_paths = []
    for variant in instance_variants:
        possible_index_paths.extend([
            index_base_dir / variant / "index_output" / "functions.jsonl",
            index_base_dir / variant / "index_output" / "entities" / "functions.jsonl",
            index_base_dir / variant / "intent_output" / variant / "index_output" / "functions.jsonl",
            index_base_dir / variant / "functions.jsonl",
        ])
    
    functions_path = None
    for p in possible_index_paths:
        if p.exists():
            functions_path = p
            break
    
    if not functions_path:
        # 返回空索引（只警告一次，避免重复输出）
        cache_key = f"{instance_id}_{index_base_dir}"
        if cache_key not in _FUNCTION_INDEX_CACHE:
            print(f"警告: 未找到 {instance_id} 的函数索引文件", file=sys.stderr)
            # 将空索引也存入缓存，避免重复警告
            _FUNCTION_INDEX_CACHE[cache_key] = {}
        return {}
    
    # 加载函数索引
    file_functions: Dict[str, List[Tuple[int, int, str, str]]] = defaultdict(list)
    
    with open(functions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                func_data = json.loads(line)
                file_path = func_data.get('def_file', '')
                if not file_path:
                    file_path = func_data.get('file', '')
                
                if not file_path:
                    continue
                
                start_line = func_data.get('def_start_line', 0)
                end_line = func_data.get('def_end_line', start_line)
                func_id = func_data.get('id', '')
                func_name = func_data.get('fq_name', '') or func_data.get('short_name', '')
                
                if start_line > 0 and func_id:
                    # 标准化文件路径（转换为相对路径）
                    file_path_normalized = _normalize_index_file_path(file_path)
                    file_functions[file_path_normalized].append(
                        (start_line, end_line, func_id, func_name)
                    )
            except json.JSONDecodeError:
                continue
    
    # 对每个文件的函数列表按行号排序
    for file_path in file_functions:
        file_functions[file_path].sort(key=lambda x: x[0])
    
    result = dict(file_functions)
    
    # 存入缓存
    if use_cache:
        _FUNCTION_INDEX_CACHE[cache_key] = result
    
    return result


def _normalize_index_file_path(file_path: str) -> str:
    """
    标准化索引中的文件路径，转换为相对路径
    索引中的路径可能是绝对路径，需要转换为相对路径以匹配patch中的路径
    """
    # 移除绝对路径前缀
    path = file_path.replace('\\', '/')
    
    # 尝试提取相对路径部分
    # 常见模式：/path/to/repo/subdir/file.cpp -> subdir/file.cpp
    # 或者已经是相对路径
    if path.startswith('/'):
        # 尝试找到常见的仓库根目录标记
        parts = path.split('/')
        # 查找 'include', 'src', 'lib' 等常见目录
        for i, part in enumerate(parts):
            if part in ['include', 'src', 'lib', 'examples', 'tests']:
                return '/'.join(parts[i:])
        # 如果找不到，返回最后两部分
        if len(parts) >= 2:
            return '/'.join(parts[-2:])
        return parts[-1] if parts else path
    
    return path


def load_class_index(instance_id: str, index_base_dir: Optional[str] = None, use_cache: bool = True) -> Dict[str, List[Tuple[int, int, str, str]]]:
    """
    加载指定instance的类索引
    
    Args:
        instance_id: 实例ID
        index_base_dir: 索引基础目录
        use_cache: 是否使用缓存
    
    Returns:
        {file_path: [(start_line, end_line, class_id, class_name), ...]}
    """
    # 检查缓存
    cache_key = f"{instance_id}_{index_base_dir}"
    if use_cache and cache_key in _CLASS_INDEX_CACHE:
        return _CLASS_INDEX_CACHE[cache_key]
    
    if index_base_dir is None:
        index_base_dir = os.environ.get("INDEX_STORE_DIR")
        if not index_base_dir:
            possible_paths = [
                "./cache/INTENT2FIX/index_results",
                "./INTENT2FIX/index",
            ]
            for p in possible_paths:
                p = Path(p)
                if p.exists():
                    index_base_dir = str(p)
                    break
        
        if not index_base_dir:
            return {}
    
    index_base_dir = Path(index_base_dir)
    instance_variants = [
        instance_id,
        f"swe-bench_{instance_id}",
    ]
    
    possible_index_paths = []
    for variant in instance_variants:
        possible_index_paths.extend([
            index_base_dir / variant / "index_output" / "classes.jsonl",
            index_base_dir / variant / "index_output" / "entities" / "classes.jsonl",
        ])
    
    classes_path = None
    for p in possible_index_paths:
        if p.exists():
            classes_path = p
            break
    
    if not classes_path:
        # 将空索引也存入缓存，避免重复警告
        cache_key = f"{instance_id}_{index_base_dir}"
        if cache_key not in _CLASS_INDEX_CACHE:
            _CLASS_INDEX_CACHE[cache_key] = {}
        return {}
    
    file_classes: Dict[str, List[Tuple[int, int, str, str]]] = defaultdict(list)
    
    with open(classes_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                class_data = json.loads(line)
                file_path = class_data.get('def_file', '') or class_data.get('file', '')
                if not file_path:
                    continue
                
                start_line = class_data.get('def_line', 0)
                end_line = class_data.get('def_end_line', start_line)
                class_id = class_data.get('id', '')
                class_name = class_data.get('fq_name', '') or class_data.get('short_name', '')
                
                if start_line > 0 and class_id:
                    file_path_normalized = _normalize_index_file_path(file_path)
                    file_classes[file_path_normalized].append(
                        (start_line, end_line, class_id, class_name)
                    )
            except json.JSONDecodeError:
                continue
    
    for file_path in file_classes:
        file_classes[file_path].sort(key=lambda x: x[0])
    
    result = dict(file_classes)
    
    # 存入缓存
    if use_cache:
        _CLASS_INDEX_CACHE[cache_key] = result
    
    return result


def extract_functions_from_patch_with_index(
    patch_text: str,
    instance_id: str,
    function_index: Optional[Dict[str, List[Tuple[int, int, str, str]]]] = None,
    class_index: Optional[Dict[str, List[Tuple[int, int, str, str]]]] = None
) -> Dict[str, Set[str]]:
    """
    使用函数索引从patch中提取函数名
    
    Args:
        patch_text: patch文本
        instance_id: 实例ID
        function_index: 函数索引，如果为None则自动加载
    
    Returns:
        {file_path: {function_ids_or_names}}
    """
    if function_index is None:
        function_index = load_function_index(instance_id)
    
    if not function_index:
        # 如果没有索引，返回空字典
        return {}
    
    file_functions: Dict[str, Set[str]] = defaultdict(set)
    
    lines = patch_text.split('\n')
    current_file = None
    modified_lines_by_file: Dict[str, Set[int]] = defaultdict(set)
    
    i = 0
    current_new_line = None  # 当前hunk在新文件中的起始行号
    current_line_offset = 0   # 当前hunk中的行号偏移
    
    while i < len(lines):
        line = lines[i]
        
        # 检测文件开始
        if line.startswith('diff --git'):
            match = re.match(r'^diff --git a/(.+?)(?:\s+b/|$)', line)
            if match:
                current_file = match.group(1).strip()
                if current_file == "/dev/null":
                    current_file = None
                current_new_line = None
                current_line_offset = 0
        elif line.startswith('--- a/'):
            match = re.match(r'^--- a/(.+?)(?:\s|$)', line)
            if match:
                current_file = match.group(1).split('\t')[0].strip()
                if current_file == "/dev/null":
                    current_file = None
                current_new_line = None
                current_line_offset = 0
        elif line.startswith('+++ b/'):
            match = re.match(r'^\+\+\+ b/(.+?)(?:\s|$)', line)
            if match:
                current_file = match.group(1).split('\t')[0].strip()
                if current_file == "/dev/null":
                    current_file = None
                current_new_line = None
                current_line_offset = 0
        
        # 检测hunk头，提取行号信息
        elif line.startswith('@@'):
            # @@ -181,6 +181,17 @@ 格式
            # -181,6 表示原文件从第181行开始，共6行上下文
            # +181,17 表示新文件从第181行开始，共17行（包括新增的）
            hunk_match = re.match(r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
            if hunk_match and current_file:
                new_start = int(hunk_match.group(3))
                current_new_line = new_start
                current_line_offset = 0  # 重置偏移
                # 注意：hunk头中的行号是上下文行，实际修改的行需要从后续的+/-行中计算
        
        # 检测修改的行（+或-开头），记录行号
        elif current_file and current_new_line is not None:
            if line.startswith('+'):
                # 新增的行：在新文件中的行号 = current_new_line + current_line_offset
                if current_file:
                    actual_line = current_new_line + current_line_offset
                    modified_lines_by_file[current_file].add(actual_line)
                current_line_offset += 1
            elif line.startswith('-'):
                # 删除的行：不影响新文件的行号，但需要调整偏移
                # 不增加current_line_offset，因为新文件中没有这行
                pass
            elif line.startswith(' '):
                # 上下文行（未修改）：两边的行号都增加
                current_line_offset += 1
        
        i += 1
    
        # 根据修改的行号匹配函数和类
    for file_path, modified_lines in modified_lines_by_file.items():
        if not is_source_code_file(file_path):
            continue
        
        # 尝试多个可能的文件路径匹配
        matched_functions = set()
        matched_classes = set()
        
        # 1. 精确匹配函数
        if function_index and file_path in function_index:
            matched_functions.update(
                _find_functions_by_lines(function_index[file_path], modified_lines)
            )
        
        # 2. 标准化路径匹配函数
        normalized_path = normalize_file_path(file_path)
        if function_index:
            if normalized_path != file_path and normalized_path in function_index:
                matched_functions.update(
                    _find_functions_by_lines(function_index[normalized_path], modified_lines)
                )
            
            # 3. 文件名匹配函数（如果路径不同但文件名相同）
            if not matched_functions:
                file_basename = file_path.split('/')[-1]
                for indexed_file, funcs in function_index.items():
                    if indexed_file.endswith('/' + file_basename) or indexed_file == file_basename:
                        matched_functions.update(
                            _find_functions_by_lines(funcs, modified_lines)
                        )
                        break
        
        # 4. 如果函数匹配失败，尝试匹配类（检查修改的行是否在类的定义范围内）
        # 注意：类的定义通常在头文件中，但实现可能在.cpp文件中
        # 如果修改的行在某个类的范围内，我们可以推断这是该类的方法
        if not matched_functions and class_index:
            # 检查当前文件中的类
            if file_path in class_index:
                matched_classes.update(
                    _find_functions_by_lines(class_index[file_path], modified_lines)
                )
            
            # 检查对应的头文件中的类（.cpp -> .h）
            if not matched_classes:
                header_file = file_path.replace('.cpp', '.h').replace('.c', '.h')
                if header_file in class_index:
                    matched_classes.update(
                        _find_functions_by_lines(class_index[header_file], modified_lines)
                    )
            
            # 如果找到了类，尝试从函数索引中找到该类的方法
            if matched_classes:
                # 查找该类的方法（函数名包含类名）
                if function_index:
                    # matched_classes 是集合，需要从class_index中获取类名
                    matched_class_names = set()
                    if file_path in class_index:
                        for start, end, class_id, class_name in class_index[file_path]:
                            if class_id in matched_classes or class_name in matched_classes:
                                matched_class_names.add(class_name)
                    header_file = file_path.replace('.cpp', '.h').replace('.c', '.h')
                    if header_file in class_index:
                        for start, end, class_id, class_name in class_index[header_file]:
                            if class_id in matched_classes or class_name in matched_classes:
                                matched_class_names.add(class_name)
                    
                    for indexed_file, funcs in function_index.items():
                        if indexed_file == file_path or indexed_file == normalized_path:
                            for start, end, func_id, func_name in funcs:
                                # 检查函数名是否属于匹配的类
                                for class_name in matched_class_names:
                                    if class_name and '::' in func_name:
                                        # 提取类名部分
                                        func_class_parts = func_name.split('::')[:-1]
                                        func_class_name = '::'.join(func_class_parts)
                                        if func_class_name == class_name or class_name.endswith('::' + func_class_name):
                                            matched_functions.add(func_id if func_id else func_name)
                                            break
        
        # 5. 如果还是没有匹配到，尝试通过上下文推断（检查代码中是否有成员变量访问）
        # 例如：m_configData, m_config 等通常表示类成员
        if not matched_functions:
            # 从patch中提取代码片段，检查是否有类成员访问模式
            patch_lines = patch_text.split('\n')
            has_member_access = False
            for line in patch_lines:
                if re.search(r'\bm_[a-zA-Z_][\w]*\b', line) or re.search(r'\bthis->', line):
                    has_member_access = True
                    break
            
            if has_member_access and class_index:
                # 尝试找到相关的类
                # 检查头文件中的类定义
                header_file = file_path.replace('.cpp', '.h').replace('.c', '.h')
                if header_file in class_index:
                    # 找到所有相关的类
                    matched_class_names = []
                    for start, end, class_id, class_name in class_index[header_file]:
                        # 如果类名与文件名相关，可能是匹配的类
                        file_basename = file_path.split('/')[-1].replace('.cpp', '').replace('.c', '')
                        if file_basename.lower() in class_name.lower() or class_name.lower() in file_basename.lower():
                            matched_classes.add(class_id if class_id else class_name)
                            matched_class_names.append(class_name)
                    
                    # 尝试找到该类的方法
                    if matched_class_names and function_index and file_path in function_index:
                        for func_start, func_end, func_id, func_name in function_index[file_path]:
                            for class_name in matched_class_names:
                                if class_name and '::' in func_name:
                                    func_class_parts = func_name.split('::')[:-1]
                                    func_class_name = '::'.join(func_class_parts)
                                    if func_class_name == class_name:
                                        matched_functions.add(func_id if func_id else func_name)
                                        break
        
        if matched_functions:
            file_functions[file_path].update(matched_functions)
        elif matched_classes:
            # 如果没有找到具体的方法，至少记录类名
            file_functions[file_path].update(matched_classes)
    
    return dict(file_functions)


def _find_functions_by_lines(
    func_list: List[Tuple[int, int, str, str]],
    modified_lines: Set[int]
) -> Set[str]:
    """
    根据修改的行号找到对应的函数
    
    Args:
        func_list: [(start_line, end_line, func_id, func_name), ...]
        modified_lines: 修改的行号集合
    
    Returns:
        函数ID或名称的集合
    """
    matched_funcs = set()
    
    for start_line, end_line, func_id, func_name in func_list:
        # 检查是否有修改的行在这个函数的范围内
        if any(start_line <= line <= end_line for line in modified_lines):
            # 优先使用函数ID，如果没有则使用函数名
            matched_funcs.add(func_id if func_id else func_name)
    
    return matched_funcs


def load_groundtruth_with_index(
    jsonl_path: str,
    index_base_dir: Optional[str] = None,
    show_progress: bool = False
) -> Dict[str, Dict]:
    """加载groundtruth数据（使用函数索引和类索引）"""
    groundtruth = {}
    
    # 先统计总行数（用于进度显示）
    total_lines = 0
    if show_progress:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
    
    processed = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                inst_id = normalize_instance_id(data)
                if not inst_id:
                    continue
                
                patch = data.get('patch', '')
                files = extract_files_from_patch(patch, filter_source_only=True)
                
                # 使用函数索引和类索引提取函数
                function_index = load_function_index(inst_id, index_base_dir)
                class_index = load_class_index(inst_id, index_base_dir)
                functions = extract_functions_from_patch_with_index(
                    patch, inst_id, function_index, class_index
                )
                
                groundtruth[inst_id] = {
                    'files': files,
                    'functions': functions,
                    'patch': patch
                }
                
                processed += 1
                if show_progress and processed % 10 == 0:
                    print(f"  已处理: {processed}/{total_lines}", end='\r', file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"警告: 解析groundtruth行失败: {e}", file=sys.stderr)
                continue
    
    if show_progress:
        print(f"  完成: {processed}/{total_lines}", file=sys.stderr)
    
    return groundtruth


def load_predictions_with_index(
    jsonl_path: str,
    index_base_dir: Optional[str] = None,
    show_progress: bool = False
) -> Dict[str, Dict]:
    """加载预测数据（使用函数索引和类索引）"""
    predictions = {}
    
    # 先统计总行数（用于进度显示）
    total_lines = 0
    if show_progress:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
    
    processed = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                inst_id = normalize_instance_id(data)
                if not inst_id:
                    continue
                
                # 尝试多个可能的字段名
                patch = data.get('fix_patch', '') or data.get('patch', '') or data.get('model_patch', '') or data.get('solution', '')
                files = extract_files_from_patch(patch, filter_source_only=True)
                
                # 使用函数索引和类索引提取函数
                function_index = load_function_index(inst_id, index_base_dir)
                class_index = load_class_index(inst_id, index_base_dir)
                functions = extract_functions_from_patch_with_index(
                    patch, inst_id, function_index, class_index
                )
                
                predictions[inst_id] = {
                    'files': files,
                    'functions': functions,
                    'patch': patch
                }
                
                processed += 1
                if show_progress and processed % 10 == 0:
                    print(f"  已处理: {processed}/{total_lines}", end='\r', file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"警告: 解析预测行失败: {e}", file=sys.stderr)
                continue
    
    if show_progress:
        print(f"  完成: {processed}/{total_lines}", file=sys.stderr)
    
    return predictions


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python stat_patch_localization_with_index.py <instance_id> <patch_file>")
        sys.exit(1)
    
    instance_id = sys.argv[1]
    patch_file = sys.argv[2]
    
    with open(patch_file, 'r') as f:
        patch_text = f.read()
    
    function_index = load_function_index(instance_id)
    functions = extract_functions_from_patch_with_index(patch_text, instance_id, function_index)
    
    print(f"实例: {instance_id}")
    print(f"提取的函数:")
    for file_path, funcs in functions.items():
        print(f"  {file_path}: {funcs}")
