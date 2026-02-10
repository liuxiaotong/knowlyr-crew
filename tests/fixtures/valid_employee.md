---
name: test-worker
display_name: 测试员工
description: 用于测试的员工定义
tags:
  - test
triggers:
  - tw
args:
  - name: target
    description: 目标路径
    required: true
  - name: mode
    description: 工作模式
    default: normal
output:
  format: markdown
  filename: "test-{date}.md"
---

# 测试员工

你是一个测试用的数字员工。

## 工作流程

1. 读取 $target 的内容
2. 以 $mode 模式处理
3. 输出结果到 $ARGUMENTS
