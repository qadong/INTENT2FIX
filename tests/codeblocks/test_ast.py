from tree_sitter import Language, Parser
import tree_sitter_cpp
# 加载C++语法（需要先编译tree-sitter-cpp）
CPP_LANGAUGE = Language(tree_sitter_cpp.language())

# 创建Parser并配置语言
cpp_parser = Parser(CPP_LANGAUGE)

test_code = """
class FileManager {
public:
    // 关注这个带有 virtual 关键字的析构函数定义
    virtual ~FileManager() {
        // Cleanup file handles or resources
    }

    bool isOpen = false;
};
"""

def print_ast(node, indent=0):
    print("  " * indent + f"{node.type} [start={node.start_point}, end={node.end_point}]")
    for child in node.children:
        print_ast(child, indent + 1)

# 解析并打印AST
tree = cpp_parser.parse(bytes(test_code, "utf8"))
print_ast(tree.root_node)