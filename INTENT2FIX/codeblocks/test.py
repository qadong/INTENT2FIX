import json
from pathlib import Path

# 找到node-types.json文件
cpp_node_types = Path.home() / ".tree-sitter/grammars/tree-sitter-cpp/src/node-types.json"

with open(cpp_node_types) as f:
    data = json.load(f)
    print("C++节点类型列表:")
    for node in data:
        print(node['type'])