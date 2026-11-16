; --- 函数式宏定义 (preproc_function_def) ---
(preproc_function_def
  ; tree-sitter 查询通常会按顺序匹配具名子节点，
  ; 第一个子节点通常是匿名的 '#define' token，会被隐式跳过
  (identifier) @identifier                ; 捕获作为直接子节点的宏名
  (preproc_params) @parameters            ; 捕获作为直接子节点的参数列表
  (preproc_arg)? @macro_value              ; 捕获作为直接子节点的宏体 (宏体内容可选)
) @root @definition.macro @meta.display_full


; --- 对象式宏定义 (preproc_def) ---
(preproc_def
  name: (identifier) @identifier         ; 命名字段 'name'
  value: (preproc_arg)? @macro_value     ; 命名字段 'value' (内容可选)
  ; 此查询假定对象式宏的 AST 节点中没有 'parameters' 字段
) @root @definition.macro @meta.display_full


(translation_unit
  (_) @child.first ; 匹配任何第一个子节点
) @root @definition.module


(template_declaration
  [ ; 模板内部可能是类、结构体、函数等
    (class_specifier)
    (struct_specifier)
    (union_specifier)
    (function_definition)
  ] @child.first
) @root @definition.template


(union_specifier
  name: [ ; <--- 这个选择结构
    (type_identifier) @identifier
    (template_type name: (type_identifier) @identifier) ; <--- 处理模板特化
  ] @name_node
  body: (field_declaration_list) @child.first
) @root @definition.union


(class_specifier
  name: [ ; 使用 choice [...] 匹配两种可能的名称节点
    (type_identifier) @identifier         ; 匹配普通类名
    (template_type name: (type_identifier) @identifier) ; 匹配模板特化名，并捕获其基础名称
  ] @name_node ; 可选捕获整个名称节点
  (base_class_clause (type_identifier) @base_class_name @reference.is_a)?
  body: (field_declaration_list) @child.first
) @root @definition.class




(struct_specifier                                         ; 匹配根节点 struct_specifier
  name: [                                                 ; 匹配名为 'name' 的字段
    (type_identifier) @identifier                         ;   其值可以是 type_identifier (捕获为 @identifier)
    (template_type name: (type_identifier) @identifier) ;   或者是一个 template_type 节点，该节点内部有 'name' 字段，其值为 type_identifier (捕获为 @identifier)
  ] @name_node                                            ;   整个 'name' 字段的节点可以捕获为 @name_node (可选)
  (base_class_clause                                      ; 匹配一个可选的 base_class_clause 节点
    (type_identifier) @base_class_name @reference.is_a
  )?                                                      ; '?' 表示 base_class_clause 是可选的
  body: (field_declaration_list) @child.first            ; 匹配名为 'body' 的字段，其值必须是 field_declaration_list (捕获为 @child.first)
) @root @definition.struct                                ; 将 struct_specifier 节点标记为 @root 和 @definition.struct



(enum_specifier
  name: (type_identifier) @identifier
  body: (enumerator_list) @child.first ; 指向枚举成员列表
) @root @definition.enum 


(namespace_definition
  name: (namespace_identifier)? @identifier
  body: (declaration_list) @child.first
) @root @definition.namespace


; --- 函数/方法定义 (进一步增强，处理 operator name) ---
(function_definition
  (storage_class_specifier)* @storage_specifier ; static, extern
  (type_qualifier)* @function_specifier_leading ; constexpr 
  ("inline")? @inline_keyword
  ("virtual")? @virtual_keyword
  declarator: (function_declarator
    declarator: [
      (identifier) @identifier @declarator_type.identifier
      (field_identifier) @identifier @declarator_type.method_identifier ; 注意: field_identifier 通常不直接用于函数定义名称
      (destructor_name (identifier) @identifier) @declarator_type.destructor
      (operator_name) @identifier @declarator_type.operator
      (qualified_identifier
        scope: (_) @function_scope  ; 添加对 scope 节点的捕获
        name: (identifier) @identifier
      )* @declarator_node.qualified
    ]
    parameters: (parameter_list)? @parameters
    (virtual_specifier)* @function_specifier_override_final
    (type_qualifier)* @function_specifier_const
    (noexcept)? @function_specifier_noexcept
  )
  (field_initializer_list)? @initializer_list
  body: (compound_statement ("{") @child.first)
  ; 纯虚函数 (= 0;) 或默认/删除函数 (= default/delete;) 需要不同的 body 匹配或单独处理
) @root @definition.function

(field_declaration_list
  (_) @child.first
) @root @definition.implementation_block ; 或 @definition.block

(declaration_list ; 用于 namespace
  (_) @child.first
) @root @definition.implementation_block 

(enumerator_list ; 用于 enum
  (_) @child.first
) @root @definition.implementation_block 


(comment) @root @definition.comment



(preproc_include
  path: [
    (system_lib_string) @identifier @reference.imports
    (string_literal) @identifier @reference.imports
  ]
) @root @definition.import


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
  name: (identifier) @identifier 
  (_)?@child.first
  alternative: (_) @meta.alternative_branch_node
) @root @definition.preproc_ifdef 

(preproc_ifdef
  name: (identifier) @identifier 
  (_)?@child.first
) @root @definition.preproc_ifdef 

(preproc_if
  condition: (_) @condition_node ; 捕获条件节点, 可在 metadata 中存储
  ; 捕获 #if 块内部的第一个实际子节点 (在 condition 之后)
  ; 例如，一个 function_definition, declaration, 或甚至是另一个 preproc 指令
  (_)? @child.first ; 使用 (_)? 使其对于空 #if 块也能匹配 (尽管那时 @child.first 不会捕获)
) @root @definition.preproc_if


(friend_declaration
  ; friend 后面通常跟一个 declaration 或 function_definition 等
  ; 可以简单地将其视为原子块，或尝试捕获内部声明
   (_) @declaration_node ; 捕获 friend 后的内容
) @root @definition.friend @meta.display_full ; 需要 FRIEND 类型

(access_specifier) @root @definition.access_modifier

(linkage_specification
  value: (string_literal) @linkage_string ; Capture to check its value in Python
  body: (function_definition
    declarator: (function_declarator
      declarator: (identifier) @identifier
    )
  ) @function_node @child.first
) @root @definition.linkage