import sys
from INTENT2FIX.codeblocks import CodeBlockType, SpanType
from INTENT2FIX.codeblocks.parser.cpp import CppParser
from INTENT2FIX.codeblocks.codeblocks import (
    Relationship,
    RelationshipType,
    ReferenceScope,
)

def test_cpp_parser():
    """主测试函数，包含所有测试用例"""
    print("Running C++ parser tests...")
    
    # 测试基本函数解析
    # test_simple_function()
    
    # 测试类和成员方法
    # test_class_with_methods()
    
    # 测试命名空间
    # test_namespace()
    
    # # 测试模板类
    # test_template_class()
    
    # # 测试继承关系
    # test_inheritance()
    
    # # 测试Lambda表达式
    test_lambda()
    
    # # 测试控制结构
    # test_control_structures()
    
    # # 测试注释掉的代码
    # test_commented_code()
    
    # # 测试宏定义
    # test_macro()
    
    # # 测试Span识别
    # test_span_identification()
    
    # print("All C++ parser tests passed!")

# def _verify_parsing(content, assertion, debug=True):
#     """辅助函数：验证解析结果"""
#     parser = CppParser(debug=debug)
#     codeblock = parser.parse(content)
    
#     if debug:
#         print(codeblock.to_tree(include_references=True, show_spans=True, show_tokens=True))
    
#     assert codeblock.to_string() == content
#     assertion(codeblock)


def _verify_parsing(content, assertion, debug=True):
    parser = CppParser(debug=debug)
    codeblock = parser.parse(content)
    parsed = codeblock.to_string()
    
    # 标准化空白：移除空行中的尾随空格
    normalized_content = '\n'.join(line.rstrip() for line in content.split('\n'))
    normalized_parsed = '\n'.join(line.rstrip() for line in parsed.split('\n'))
    assertion(codeblock)
    assert normalized_parsed == normalized_content



def test_simple_function():
    """测试基本函数解析"""
    content = """int add(int a, int b) {
    return a + b;
}"""
    
    def assertion(codeblock):
            print(
                codeblock.to_tree(
                    include_references=True,
                    show_spans=True,
                    show_tokens=True,
                    only_identifiers=True,
                )
            )
    
    _verify_parsing(content, assertion)
    print("✓ Simple function test passed")

def test_class_with_methods():
    """测试类和成员方法"""
    content = """class Calculator {
    int add(int a, int b) {
        return a + b;
    }
    int sub(){}
    int foo(){}
};"""
    
    def assertion(codeblock):
            print(
                codeblock.to_tree(
                    include_references=True,
                    show_spans=True,
                    show_tokens=True,
                    only_identifiers=False,
                )
            )
    
    
    _verify_parsing(content, assertion)
    print("✓ Class with methods test passed")

def test_namespace():
    """测试命名空间"""
    content = """namespace Math {
    class Calculator {
    public:  // 缺少访问修饰符
        int add(int a, int b) {
            return a + b;
        }
    };  // 类定义后缺少分号

    const double PI = 3.14159;
    
    double circle_area(double radius) {
        return PI * radius * radius;
    }
}  // 命名空间结束
"""
    
    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )
    
    _verify_parsing(content, assertion)
    print("✓ Namespace test passed")

def test_template_class():
    """测试模板类"""
    content = """template<typename T>
class Vector {
private:
    T* data;
    size_t size;
    
public:
    Vector(size_t size) : size(size) {
        data = new T[size];
    }
    
    ~Vector() {
        delete[] data;
    }
};"""
    
    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )

    
    _verify_parsing(content, assertion)
    print("✓ Template class test passed")

def test_inheritance():
    """测试继承关系"""
    content = """class Shape {
public:
    virtual double area() = 0;
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    double area() override {
        return 3.14159 * radius * radius;
    }
};"""
    
    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )
    
    _verify_parsing(content, assertion)
    print("✓ Inheritance test passed")

def test_lambda():


    with open(file="/tmp/INTENT2FIX/tmp/repos/swe-bench_nlohmann__json_3073/single_include/nlohmann/json.hpp", mode="r") as f:
        lines=f.readlines()
#     """测试Lambda表达式"""
#     content = """ 
# JSON_HEDLEY_NON_NULL(1)
# inline nlohmann::json operator "" _json(const char* s, std::size_t n)
# {
#     return nlohmann::json::parse(s, s + n);
# }


# """
    content = "".join(lines[0:50])
    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )
    
    _verify_parsing(content, assertion)
    
    print("✓ Lambda test passed")

def test_control_structures():
    """测试控制结构"""
    content = """int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}"""
    
    def assertion(codeblock):
        func = codeblock.children[0]
        if_block = func.find_blocks_with_type(CodeBlockType.IF_STATEMENT)
        assert len(if_block) == 1
        
        if_children = if_block[0].children
        assert len(if_children) == 2
        assert if_children[0].type == CodeBlockType.BLOCK
        assert if_children[1].type == CodeBlockType.BLOCK
    
    _verify_parsing(content, assertion)
    print("✓ Control structures test passed")

def test_commented_code():
    """测试注释掉的代码"""
    content = """/*
int old_function() {
    return 0;
}
*/

int new_function() {
    return 1;
}"""
    
    def assertion(codeblock):
        assert len(codeblock.children) == 2
        
        commented = codeblock.children[0]
        assert commented.type == CodeBlockType.COMMENTED_OUT_CODE
        assert commented.identifier == "old_function"
        
        normal = codeblock.children[1]
        assert normal.type == CodeBlockType.FUNCTION
        assert normal.identifier == "new_function"
    
    _verify_parsing(content, assertion)
    print("✓ Commented code test passed")

def test_macro():
    """测试宏定义"""
    content = """#define VERSION_CODE 101
#if VERSION_CODE > 100
void feature_new() {
    // 这个 #if 应该关联到上面的 VERSION_CODE 定义
    std::cout << "Using new feature (Version > 100)" << std::endl;
}
#else
void feature_old() {
    std::cout << "Using old feature (Version <= 100)" << std::endl;
}
#endif // VERSION_CODE > 100"""
#     content = """
#     #define VERSION_CODE 101
# """
    
    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )
    
    _verify_parsing(content, assertion)
    print("✓ Macro test passed")

def test_span_identification():
    """测试Span识别"""
    content = """class Database {
private:
    Connection* conn;
    
public:
    Database() : conn(nullptr) {}
    
    void connect(std::string url) {
        conn = new Connection(url);
    }
    
    ~Database() {
        delete conn;
    }
};"""
    
    def assertion(codeblock):
        db_class = codeblock.children[0]
        spans = db_class.get_all_span_ids()
        
        assert "Database" in spans
        assert "Database::Database" in spans
        assert "Database::~Database" in spans
        assert "Database::connect" in spans
        
        connect_method = db_class.find_by_path(["Database", "connect"])
        assert connect_method.belongs_to_span.span_type == SpanType.IMPLEMENTATION
    
    _verify_parsing(content, assertion)
    print("✓ Span identification test passed")

if __name__ == "__main__":
    # 执行所有测试
    try:
        test_cpp_parser()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    sys.exit(0)