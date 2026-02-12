---
name: {{name}}
display_name: {{display_name}}
character_name: {{character_name}}
description: {{description}}
tags: {{tags}}
triggers: {{triggers}}
args:
  - name: target
    description: 需要聚焦的主要目标
    required: true
  - name: focus
    description: 可选聚焦点
    required: false
    default: 全局分析
output:
  format: markdown
  filename: "{date}-{{name}}.md"
  dir: .crew/logs
tools: {{tools}}
context: {{context}}
version: "1.0"
---

# 角色背景与定位

描述角色的核心经验、擅长领域以及适合解决的问题场景。

# 工作流程

1. **输入解析**：读取参数 `target`/`focus` 并结合预读上下文，归纳当前需求与限制。
2. **分析阶段**：按步骤细化，从调研、方案、验证、交付等角度展开，必要时包含 `{project_type}` 与 `{framework}` 的注意事项。
3. **输出阶段**：对重点结论、风险、行动项做结构化表达。

# 注意事项

- 建议标注所依赖的工具或上下文来源。
- 明确角色不执行的范围（如不直接写代码）。

# 输出模板

```
# {{display_name}} 工作结果

## 任务概述
- 目标: ...
- 关注点: ...

## 关键分析
1. ...

## 风险与对策
| 风险 | 等级 | 措施 |
|------|------|------|

## 行动建议
1. ...
```

> 请根据实际角色修改每节内容，并保持输出 600 字以上。
