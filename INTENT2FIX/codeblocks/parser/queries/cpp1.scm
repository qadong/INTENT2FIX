(preproc_if
  condition: (_) @condition_node ; 捕获条件节点, 可在 metadata 中存储
  ; 捕获 #if 块内部的第一个实际子节点 (在 condition 之后)
  ; 例如，一个 function_definition, declaration, 或甚至是另一个 preproc 指令
  (_)? @child.first ; 使用 (_)? 使其对于空 #if 块也能匹配 (尽管那时 @child.first 不会捕获)
  ; 捕获 alternative 字段指向的节点，以便后续在 Python 中处理
  alternative: (_) @meta.alternative_branch_node
) @root @definition.preproc_if

; --- #else 条件编译 (结构化, 解析其内部) ---
(preproc_else
  ; 捕获 #else 块内部的第一个实际子节点
  (_)? @child.first
) @root @definition.preproc_else

; --- #elif 条件编译 (结构化, 解析其内部) ---
(preproc_elif
  condition: (_) @condition_node
  (_)? @child.first
  alternative: (_) @meta.alternative_branch_node
) @root @definition.preproc_elif


; --- #ifdef / #ifndef (原子化/完整显示) ---
(preproc_ifdef
  name: (identifier) @identifier ; 捕获被检查的宏名
  ; 这里没有 @child.first 标签
) @root @definition.preproc_ifdef @meta.display_full