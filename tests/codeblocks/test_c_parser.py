from INTENT2FIX.codeblocks.codeblocks import (
    CodeBlockType,
    Relationship,
    RelationshipType,
    ReferenceScope
)
from INTENT2FIX.codeblocks.parser.c import CParser  # 假设我们有一个CParser类

def _verify_parsing(content, assertion, apply_gpt_tweaks=False, debug=True):
    parser = CParser(apply_gpt_tweaks=apply_gpt_tweaks, debug=debug)
    codeblock = parser.parse(content)
    
    if debug:
        print("\nParse Tree:")
        print(codeblock.to_tree(include_references=True, show_spans=True))
    
    assertion(codeblock)

def find_blocks_by_type(codeblock, block_type):
    """查找指定类型的所有代码块"""
    blocks = []
    for child in codeblock.children:
        if child.type == block_type:
            blocks.append(child)
        blocks.extend(find_blocks_by_type(child, block_type))
    return blocks

def test_c_struct():
    content = """struct Point {
    int x;
    int y;
};

typedef struct {
    float radius;
} Circle;
"""

    def assertion(codeblock):
        # 查找所有结构体定义
        structs = find_blocks_by_type(codeblock, CodeBlockType.STRUCT)
        print(f"Found {len(structs)} struct blocks")
        
        assert len(structs) == 2
        
        # 验证第一个结构体
        point_struct = structs[0]
        assert point_struct.identifier == "Point"
        assert "struct Point" in point_struct.content
        assert "int x;" in point_struct.content
        assert "int y;" in point_struct.content
        
        # 验证typedef结构体
        circle_struct = structs[1]
        assert circle_struct.identifier == "Circle"
        assert "typedef struct" in circle_struct.content
        assert "float radius;" in circle_struct.content

    _verify_parsing(content, assertion)
    
    
def test_c_function():
    content = """
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

void print_message(const char* message) {
    printf("%s\\n", message);
}

float calculate_average(float* values, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}
"""

    def assertion(codeblock):
        # Find all function blocks
        functions = find_blocks_by_type(codeblock, CodeBlockType.FUNCTION)
        print(f"Found {len(functions)} function blocks")
        assert len(functions) == 3
        
        # Test first function (add)
        add_func = functions[0]
        assert add_func.identifier == "add"
        assert add_func.return_type == "int"
        assert "int add(int a, int b)" in add_func.content
        assert "return a + b;" in add_func.content
        
        # Verify parameters
        params = find_blocks_by_type(add_func, CodeBlockType.PARAMETER)
        assert len(params) == 2
        assert params[0].identifier == "a"
        assert params[0].type == "int"
        assert "int a" in params[0].content
        assert params[1].identifier == "b"
        assert params[1].type == "int"
        assert "int b" in params[1].content
        
        # Test second function (print_message)
        print_func = functions[1]
        assert print_func.identifier == "print_message"
        assert print_func.return_type == "void"
        assert "void print_message(const char* message)" in print_func.content
        assert "printf(\"%s\\n\", message);" in print_func.content
        
        # Verify parameter
        params = find_blocks_by_type(print_func, CodeBlockType.PARAMETER)
        assert len(params) == 1
        assert params[0].identifier == "message"
        assert params[0].type == "const char*"
        assert "const char* message" in params[0].content
        
        # Test third function (calculate_average)
        avg_func = functions[2]
        assert avg_func.identifier == "calculate_average"
        assert avg_func.return_type == "float"
        assert "float calculate_average(float* values, int count)" in avg_func.content
        assert "return sum / count;" in avg_func.content
        
        # Verify parameters
        params = find_blocks_by_type(avg_func, CodeBlockType.PARAMETER)
        assert len(params) == 2
        assert params[0].identifier == "values"
        assert params[0].type == "float*"
        assert "float* values" in params[0].content
        assert params[1].identifier == "count"
        assert params[1].type == "int"
        assert "int count" in params[1].content
        
        # Verify function body contains statements
        body_blocks = find_blocks_by_type(avg_func, CodeBlockType.BLOCK)
        assert len(body_blocks) == 1
        body = body_blocks[0]
        assert "float sum = 0.0f;" in body.content
        assert "for (int i = 0; i < count; i++)" in body.content
        assert "sum += values[i];" in body.content
        
        # Verify for loop is properly parsed
        loops = find_blocks_by_type(avg_func, CodeBlockType.LOOP)
        assert len(loops) == 1
        assert "for (int i = 0; i < count; i++)" in loops[0].content

    _verify_parsing(content, assertion)
    
    
def test_typedef_struct():
    content = """typedef struct {
        float radius;
        char* name;
    } Circle;"""
    
    def assertion(codeblock):
        # 验证typedef
        typedefs = find_blocks_by_type(codeblock, CodeBlockType.C_TYPEDEF)
        assert len(typedefs) == 1
        assert typedefs[0].identifier == "Circle"
        assert "typedef struct" in typedefs[0].content
        
        # 验证结构体
        structs = find_blocks_by_type(typedefs[0], CodeBlockType.C_STRUCT)
        assert len(structs) == 1
        assert "struct {" in structs[0].content
        
        # 验证成员
        members = find_blocks_by_type(structs[0], CodeBlockType.MEMBER)
        assert len(members) == 2
        assert members[0].identifier == "radius"
        assert "float radius;" in members[0].content
        assert members[1].identifier == "name"
        assert "char* name;" in members[1].content
        assert len(members[0].children) == 0  # 成员不应有子节点
        
    _verify_parsing(content, assertion)
def test_c_enum_and_union():
    content = """enum Color { RED, GREEN, BLUE };

union Data {
    int i;
    float f;
    char str[20];
};
"""

    def assertion(codeblock):
        # 查找枚举定义
        enums = find_blocks_by_type(codeblock, CodeBlockType.CLASS)  # 假设枚举使用CLASS类型
        print(f"Found {len(enums)} enum blocks")
        assert len(enums) == 1
        assert enums[0].identifier == "enum_Color_"
        assert "enum Color" in enums[0].content
        assert "RED, GREEN, BLUE" in enums[0].content
        
        # 查找联合体定义
        unions = find_blocks_by_type(codeblock, CodeBlockType.STRUCT)  # 假设联合体使用STRUCT类型
        print(f"Found {len(unions)} union blocks")
        assert len(unions) == 1
        assert unions[0].identifier == "union_Data_"
        assert "union Data" in unions[0].content
        assert "int i;" in unions[0].content
        assert "float f;" in unions[0].content
        assert "char str[20];" in unions[0].content

    _verify_parsing(content, assertion)

if __name__ == "__main__":
    # test_typedef_struct()
    # test_c_struct()
    test_c_function()
    test_c_enum_and_union()
    print("All C tests passed!")