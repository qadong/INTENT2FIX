#!/usr/bin/env python3
"""
使用函数索引（functions.jsonl）和类索引（classes.jsonl）的批量统计脚本
统计7组实验的具体定位结果
如果patch为空，直接当作没有定位成功
"""

import json
import sys
import os
from stat_patch_localization_with_index import (
    load_groundtruth_with_index,
    load_predictions_with_index,
    calculate_file_accuracy,
    calculate_function_accuracy
)
from stat_patch_localization import extract_files_from_patch
from stat_patch_localization_hunk import (
    extract_hunks_from_patch,
    calculate_hunk_accuracy_by_overlap
)

def stat_single_group(groundtruth_path, prediction_path, group_name, index_base_dir=None):
    """统计单组patch的定位准确率（使用函数索引，允许多预测模式）"""
    print(f"\n{'='*80}")
    print(f"统计组: {group_name}")
    print(f"{'='*80}")
    
    # 加载数据
    import time
    start_time = time.time()
    
    print("加载groundtruth数据...")
    groundtruth = load_groundtruth_with_index(groundtruth_path, index_base_dir, show_progress=True)
    print(f"\n加载了 {len(groundtruth)} 条groundtruth记录 (耗时: {time.time() - start_time:.2f}秒)")
    
    start_time = time.time()
    print("加载预测数据...")
    predictions = load_predictions_with_index(prediction_path, index_base_dir, show_progress=True)
    print(f"\n加载了 {len(predictions)} 条预测记录 (耗时: {time.time() - start_time:.2f}秒)")
    
    # 统计结果
    file_correct = 0
    file_total = 0
    file_recall_scores = []
    
    function_correct = 0
    function_total = 0
    function_recall_scores = []
    
    template_function_correct = 0
    template_function_total = 0
    template_function_recall_scores = []
    
    hunk_correct = 0
    hunk_total = 0
    hunk_recall_scores = []
    
    matched_instances = 0
    unmatched_instances = []
    empty_patch_instances = []  # patch为空的实例
    
    detailed_results = []
    
    # 对每个预测进行匹配
    for inst_id, pred_data in predictions.items():
        gt_data = groundtruth.get(inst_id)
        
        if not gt_data:
            # 尝试模糊匹配
            matched = False
            for gt_id, gt_data_candidate in groundtruth.items():
                if inst_id in gt_id or gt_id in inst_id:
                    gt_data = gt_data_candidate
                    matched = True
                    break
            
            if not matched:
                unmatched_instances.append(inst_id)
                continue
        
        matched_instances += 1
        
        pred_files = pred_data['files']
        gt_files = gt_data['files']
        
        pred_functions = pred_data['functions']
        gt_functions = gt_data['functions']
        
        # 检查patch是否为空
        pred_patch = pred_data.get('patch', '')
        gt_patch = gt_data.get('patch', '')
        
        # 提取hunk
        pred_hunks = extract_hunks_from_patch(pred_patch, filter_source_only=True)
        gt_hunks = extract_hunks_from_patch(gt_patch, filter_source_only=True)
        
        # 如果预测的patch为空，直接当作没有定位成功
        if not pred_patch or not pred_patch.strip():
            empty_patch_instances.append(inst_id)
            # 文件级别：不正确
            file_total += 1
            file_recall_scores.append(0.0)
            # 函数级别：不正确
            function_total += 1
            function_recall_scores.append(0.0)
            # Hunk级别：不正确
            hunk_total += 1
            hunk_recall_scores.append(0.0)
            
            detailed_results.append({
                'instance_id': inst_id,
                'file_correct': False,
                'file_recall': 0.0,
                'function_correct': False,
                'function_recall': 0.0,
                'hunk_correct': False,
                'hunk_recall': 0.0,
                'predicted_files': [],
                'golden_files': list(gt_files),
                'predicted_functions': {},
                'golden_functions': {k: list(v) for k, v in gt_functions.items()},
                'predicted_hunks': [],
                'golden_hunks': len(gt_hunks),
                'missing_files': list(gt_files),
                'extra_files': [],
                'is_empty_patch': True,
            })
            continue
        
        # 提取模板函数/类（如果需要）
        from stat_patch_localization import is_template_function
        
        # 分离模板函数和普通函数
        gt_template_funcs = {}
        for file_path, funcs in gt_functions.items():
            gt_template_funcs[file_path] = set()
            for func_name in funcs:
                if is_template_function(func_name, gt_patch):
                    gt_template_funcs[file_path].add(func_name)
        
        pred_template_funcs = {}
        for file_path, funcs in pred_functions.items():
            pred_template_funcs[file_path] = set()
            for func_name in funcs:
                if is_template_function(func_name, pred_patch):
                    pred_template_funcs[file_path].add(func_name)
        
        # 文件级别统计（允许多预测）
        file_is_correct, file_recall = calculate_file_accuracy(pred_files, gt_files, exact_match=False)
        file_total += 1
        if file_is_correct:
            file_correct += 1
        file_recall_scores.append(file_recall)
        
        # 函数级别统计（允许多预测）
        func_is_correct, func_recall = calculate_function_accuracy(pred_functions, gt_functions, exact_match=False)
        function_total += 1
        if func_is_correct:
            function_correct += 1
        function_recall_scores.append(func_recall)
        
        # 模板函数/类统计（允许多预测）
        if any(gt_template_funcs.values()):
            template_func_is_correct, template_func_recall = calculate_function_accuracy(
                pred_template_funcs, gt_template_funcs, exact_match=False
            )
            template_function_total += 1
            if template_func_is_correct:
                template_function_correct += 1
            template_function_recall_scores.append(template_func_recall)
        
        # Hunk级别统计（允许多预测，使用重叠匹配）
        hunk_is_correct, hunk_recall = calculate_hunk_accuracy_by_overlap(
            pred_hunks, gt_hunks, overlap_threshold=0.5, exact_match=False
        )
        hunk_total += 1
        if hunk_is_correct:
            hunk_correct += 1
        hunk_recall_scores.append(hunk_recall)
        
        # 保存详细结果
        detailed_results.append({
            'instance_id': inst_id,
            'file_correct': file_is_correct,
            'file_recall': file_recall,
            'function_correct': func_is_correct,
            'function_recall': func_recall,
            'hunk_correct': hunk_is_correct,
            'hunk_recall': hunk_recall,
            'predicted_files': list(pred_files),
            'golden_files': list(gt_files),
            'predicted_functions': {k: list(v) for k, v in pred_functions.items()},
            'golden_functions': {k: list(v) for k, v in gt_functions.items()},
            'predicted_hunks': len(pred_hunks),
            'golden_hunks': len(gt_hunks),
            'missing_files': list(gt_files - pred_files),
            'extra_files': list(pred_files - gt_files),
            'is_empty_patch': False,
        })
    
    # 输出统计结果
    print(f"\n匹配的实例数: {matched_instances}")
    if unmatched_instances:
        print(f"未匹配的实例数: {len(unmatched_instances)}")
    if empty_patch_instances:
        print(f"Patch为空的实例数: {len(empty_patch_instances)}")
    
    print("\n文件级别定位准确率（允许多预测）:")
    file_acc = file_correct / file_total if file_total > 0 else 0
    print(f"  完全正确: {file_correct} / {file_total} ({file_acc*100:.2f}%)")
    if file_recall_scores:
        avg_recall = sum(file_recall_scores) / len(file_recall_scores)
        print(f"  平均召回率: {avg_recall:.4f}")
    
    print("\n函数级别定位准确率（允许多预测）:")
    func_acc = function_correct / function_total if function_total > 0 else 0
    print(f"  完全正确: {function_correct} / {function_total} ({func_acc*100:.2f}%)")
    if function_recall_scores:
        avg_recall = sum(function_recall_scores) / len(function_recall_scores)
        print(f"  平均召回率: {avg_recall:.4f}")
    
    if template_function_total > 0:
        print("\n模板函数/类定位准确率（允许多预测）:")
        template_acc = template_function_correct / template_function_total if template_function_total > 0 else 0
        print(f"  完全正确: {template_function_correct} / {template_function_total} ({template_acc*100:.2f}%)")
        if template_function_recall_scores:
            avg_recall = sum(template_function_recall_scores) / len(template_function_recall_scores)
            print(f"  平均召回率: {avg_recall:.4f}")
    
    print("\nHunk级别定位准确率（允许多预测，重叠匹配）:")
    hunk_acc = hunk_correct / hunk_total if hunk_total > 0 else 0
    print(f"  完全正确: {hunk_correct} / {hunk_total} ({hunk_acc*100:.2f}%)")
    if hunk_recall_scores:
        avg_recall = sum(hunk_recall_scores) / len(hunk_recall_scores)
        print(f"  平均召回率: {avg_recall:.4f}")
    
    return {
        'group_name': group_name,
        'matched_instances': matched_instances,
        'unmatched_instances': len(unmatched_instances),
        'empty_patch_instances': len(empty_patch_instances),
        'file_level': {
            'correct': file_correct,
            'total': file_total,
            'accuracy': file_acc,
            'avg_recall': sum(file_recall_scores) / len(file_recall_scores) if file_recall_scores else 0
        },
        'function_level': {
            'correct': function_correct,
            'total': function_total,
            'accuracy': func_acc,
            'avg_recall': sum(function_recall_scores) / len(function_recall_scores) if function_recall_scores else 0
        },
        'template_function_level': {
            'correct': template_function_correct,
            'total': template_function_total,
            'accuracy': template_function_correct / template_function_total if template_function_total > 0 else 0,
            'avg_recall': sum(template_function_recall_scores) / len(template_function_recall_scores) if template_function_recall_scores else 0
        } if template_function_total > 0 else None,
        'hunk_level': {
            'correct': hunk_correct,
            'total': hunk_total,
            'accuracy': hunk_acc,
            'avg_recall': sum(hunk_recall_scores) / len(hunk_recall_scores) if hunk_recall_scores else 0
        },
        'detailed_results': detailed_results
    }


def main():
    groundtruth_path = "INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"
    
    # 定义要统计的组
    groups = [
        ("gpt5_2_cotest_origin_patch.jsonl", "GPT5-2-CoTest"),
        ("Mopenhands_claude37.jsonl", "Mopenhands-Claude37"),
        ("Mopenhands_ds.jsonl", "Mopenhands-DS"),
        ("claude-3.7-sonnet_cotest_origin_patch.jsonl", "Claude-3.7-Sonnet-CoTest"),
        ("deepseek_v3_0324_cotest_V7_origin_patch.jsonl", "DeepSeek-V3-V7-CoTest"),
    ]
    
    results = []
    
    print("="*80)
    print("批量统计Patch定位准确率（函数索引+类索引，允许多预测）- 7组实验")
    print("注意：patch为空的实例将被视为定位失败")
    print("="*80)
    
    for prediction_file, group_name in groups:
        prediction_path = prediction_file
        if not os.path.exists(prediction_path):
            print(f"\n警告: 文件不存在: {prediction_path}")
            continue
        
        result = stat_single_group(groundtruth_path, prediction_path, group_name)
        results.append(result)
    
    # 输出对比表格
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    print(f"\n{'组名':<30} {'文件准确率':<15} {'函数准确率':<15} {'Hunk准确率':<15} {'模板函数准确率':<15}")
    print("-"*100)
    for result in results:
        group_name = result['group_name']
        file_acc = result['file_level']['accuracy'] * 100
        func_acc = result['function_level']['accuracy'] * 100
        hunk_acc = result['hunk_level']['accuracy'] * 100
        template_acc = result['template_function_level']['accuracy'] * 100 if result['template_function_level'] else 0
        
        print(f"{group_name:<30} {file_acc:>6.2f}% ({result['file_level']['correct']}/{result['file_level']['total']})  "
              f"{func_acc:>6.2f}% ({result['function_level']['correct']}/{result['function_level']['total']})  "
              f"{hunk_acc:>6.2f}% ({result['hunk_level']['correct']}/{result['hunk_level']['total']})  "
              f"{template_acc:>6.2f}% ({result['template_function_level']['correct']}/{result['template_function_level']['total']})" 
              if result['template_function_level'] else 
              f"{group_name:<30} {file_acc:>6.2f}% ({result['file_level']['correct']}/{result['file_level']['total']})  "
              f"{func_acc:>6.2f}% ({result['function_level']['correct']}/{result['function_level']['total']})  "
              f"{hunk_acc:>6.2f}% ({result['hunk_level']['correct']}/{result['hunk_level']['total']})  "
              f"{'N/A':<15}")
    
    print(f"\n{'组名':<30} {'文件召回率':<15} {'函数召回率':<15} {'Hunk召回率':<15} {'模板函数召回率':<15}")
    print("-"*100)
    for result in results:
        group_name = result['group_name']
        file_recall = result['file_level']['avg_recall'] * 100
        func_recall = result['function_level']['avg_recall'] * 100
        hunk_recall = result['hunk_level']['avg_recall'] * 100
        template_recall = result['template_function_level']['avg_recall'] * 100 if result['template_function_level'] else 0
        
        print(f"{group_name:<30} {file_recall:>6.2f}%        "
              f"{func_recall:>6.2f}%        "
              f"{hunk_recall:>6.2f}%        "
              f"{template_recall:>6.2f}%" 
              if result['template_function_level'] else 
              f"{group_name:<30} {file_recall:>6.2f}%        "
              f"{func_recall:>6.2f}%        "
              f"{hunk_recall:>6.2f}%        "
              f"{'N/A':<15}")
    
    # 显示空patch统计
    print(f"\n{'组名':<30} {'空Patch数量':<15}")
    print("-"*50)
    for result in results:
        group_name = result['group_name']
        empty_count = result['empty_patch_instances']
        print(f"{group_name:<30} {empty_count:>15}")
    
    # 保存详细结果
    output_file = "batch_localization_statistics_with_index.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'groundtruth': groundtruth_path,
            'mode': 'allow_extra_predictions',
            'method': 'function_index_and_class_index',
            'empty_patch_treated_as_failure': True,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {output_file}")
    print("="*80)
    
    # 生成详细报告
    report_file = "localization_detailed_report_with_index.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Patch定位准确率详细报告（函数索引+类索引，允许多预测）\n")
        f.write("注意：patch为空的实例被视为定位失败\n")
        f.write("="*100 + "\n\n")
        
        for result in results:
            f.write(f"\n{'='*100}\n")
            f.write(f"组: {result['group_name']}\n")
            f.write(f"{'='*100}\n\n")
            
            f.write(f"文件级别: {result['file_level']['correct']}/{result['file_level']['total']} "
                   f"({result['file_level']['accuracy']*100:.2f}%), "
                   f"平均召回率: {result['file_level']['avg_recall']:.4f}\n")
            f.write(f"函数级别: {result['function_level']['correct']}/{result['function_level']['total']} "
                   f"({result['function_level']['accuracy']*100:.2f}%), "
                   f"平均召回率: {result['function_level']['avg_recall']:.4f}\n")
            f.write(f"空Patch数量: {result['empty_patch_instances']}\n\n")
            
            # 统计失败案例
            failed_file = [r for r in result['detailed_results'] if not r['file_correct']]
            failed_func = [r for r in result['detailed_results'] if not r['function_correct']]
            empty_patches = [r for r in result['detailed_results'] if r.get('is_empty_patch', False)]
            
            f.write(f"文件级别失败案例: {len(failed_file)}/{len(result['detailed_results'])}\n")
            f.write(f"函数级别失败案例: {len(failed_func)}/{len(result['detailed_results'])}\n")
            f.write(f"空Patch案例: {len(empty_patches)}/{len(result['detailed_results'])}\n\n")
            
            # 显示前10个失败案例
            if failed_file:
                f.write("文件级别失败案例（前10个）:\n")
                for i, r in enumerate(failed_file[:10]):
                    f.write(f"\n{i+1}. {r['instance_id']}")
                    if r.get('is_empty_patch'):
                        f.write(" [空Patch]\n")
                    else:
                        f.write(f"\n   缺失文件: {r['missing_files']}\n")
                        f.write(f"   额外文件: {r['extra_files']}\n")
            
            if failed_func:
                f.write("\n函数级别失败案例（前10个）:\n")
                for i, r in enumerate(failed_func[:10]):
                    f.write(f"\n{i+1}. {r['instance_id']}")
                    if r.get('is_empty_patch'):
                        f.write(" [空Patch]\n")
                    else:
                        f.write(f"\n   召回率: {r['function_recall']:.3f}\n")
                        # 计算缺失和额外的函数
                        all_gt_funcs = set()
                        all_pred_funcs = set()
                        for file, funcs in r['golden_functions'].items():
                            all_gt_funcs.update(funcs)
                        for file, funcs in r['predicted_functions'].items():
                            all_pred_funcs.update(funcs)
                        missing_funcs = all_gt_funcs - all_pred_funcs
                        extra_funcs = all_pred_funcs - all_gt_funcs
                        if missing_funcs:
                            f.write(f"   缺失函数: {list(missing_funcs)[:10]}\n")  # 只显示前10个
                        if extra_funcs:
                            f.write(f"   额外函数: {list(extra_funcs)[:10]}\n")  # 只显示前10个
    
    print(f"详细报告已保存至: {report_file}")


if __name__ == "__main__":
    main()
