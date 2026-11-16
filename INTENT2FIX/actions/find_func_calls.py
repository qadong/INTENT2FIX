from tree_sitter import Language, Parser
import os

# 加载C语言语法库
Language.build_library(
    'build/c-lang.so',
    ['vendor/tree-sitter-c']  # 需要先git clone https://github.com/tree-sitter/tree-sitter-c
)
C_LANG = Language('build/c-lang.so', 'c')

# 初始化解析器
parser = Parser()
parser.set_language(C_LANG)

def find_function_calls(code, target_function):
    # 解析代码
    tree = parser.parse(bytes(code, "utf-8"))
    root_node = tree.root_node
    
    # 收集所有调用点
    calls = []
    query = C_LANG.query("""
        (call_expression
            function: (identifier) @func_name
            arguments: (argument_list) @args
        ) @call
    """)
    
    captures = query.captures(root_node)
    for node, tag in captures:
        if tag == "func_name" and node.text.decode("utf-8") == target_function:
            calls.append({
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "code": node.parent.text.decode("utf-8")
            })
    
    return calls

# 示例用法
code = """
#include <stdio.h>
void greet() { printf("Hello"); }
int main() {
    greet();  // 调用点1
    greet();  // 调用点2
}
"""
calls = find_function_calls(code, "greet")
print(calls)