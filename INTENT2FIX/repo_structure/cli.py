import argparse
import json
from .parser import CodeParser, DirectoryAnalyzer

def main():
    parser = argparse.ArgumentParser(description='代码结构分析工具')
    parser.add_argument('path', help='要分析的目录路径')
    parser.add_argument('-o', '--output', help='输出JSON文件路径', default='code_structure.json')
    args = parser.parse_args()
    
    print(f"Analyzing directory: {args.path}")
    
    # 初始化分析器
    code_parser = CodeParser()
    analyzer = DirectoryAnalyzer(code_parser)
    
    # 执行分析
    result = analyzer.analyze(args.path)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete! Results saved to {args.output}")

if __name__ == '__main__':
    main()