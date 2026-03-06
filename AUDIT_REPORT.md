# Crew webhook_handlers.py 全量审查报告

> 审查人：林锐 | 日期：2026-03-06

## 统计摘要
- 总端点数：约 105 个 handler 函数（含辅助函数）
- 发现问题数：32（P0: 3, P1: 12, P2: 17）

---

## P0 问题（必须立即修复）

### [P0-1] `_run_and_callback` 引用未定义的 `request` 变量
- **端点**：后台回调函数（被 `POST /api/employees/{name}/run` 和 `POST /api/chat` 调用）
- **行号**：L3397, L3432
- **问题**：`_run_and_callback` 是一个独立的 async 函数（L3337），参数列表中没有 `request`。但在 L3397 调用了 `_tenant_id_for_config(request)`，在 L3432 也调用了 `_tenant_id_for_config(request)`。这两处 `request` 引用会在运行时触发 `NameError`。由于此函数是后台任务，异常会被 `_task_done_callback` 捕获并记日志，但员工执行和蚁聚回调会完全失败——用户发消息但永远收不到回复。
- **建议修复**：在 `_run_and_callback` 的参数列表中添加 `tenant_id: str | None` 参数，调用者在创建 task 时传入已解析的 tenant_id，函数体内直接使用该值，不再依赖 request 对象。

### [P0-2] `_handle_chat` 异步回调后台任务未加入 `_background_tasks` 引用集合
- **端点**：POST /api/chat（异步模式）
- **行号**：L5146-L5162
- **问题**：`asyncio.create_task(...)` 创建了后台任务但未调用 `_background_tasks.add(task)` 和 `task.add_done_callback(_task_done_callback)`。Python 的 asyncio 不对 Task 持有强引用，如果没有其他引用，Task 对象会被垃圾回收器回收，导致异步回调被静默取消——用户的异步请求消失无响应。对比 L3720-3722（`_handle_run_employee` 中正确的实现）。
- **建议修复**：
  ```python
  task = asyncio.create_task(...)
  _background_tasks.add(task)
  task.add_done_callback(_task_done_callback)
  ```

### [P0-3] `_handle_memory_dashboard` 全局统计无分页限制，可 OOM
- **端点**：GET /api/memory/dashboard（不传 employee 时）
- **行号**：L2276-L2319
- **问题**：不传 `employee` 参数时，遍历所有员工，每人 `store.query(emp, limit=1000)` 全量加载。假设有 20 个员工各 1000 条记忆，共 20000 条全加载到内存做统计。如果记忆数增长（或攻击者创建大量员工+记忆），将导致内存溢出或极长响应时间。类似问题也存在于 L4308-4310 的 `_handle_org_memories`（`limit=0` 不限）。
- **建议修复**：改为 SQL 聚合查询（DB 版），或限制每员工查询上限并在文档中注明。`_handle_org_memories` 的 `limit=0` 应改为有合理上限（如 10000）。

---

## P1 问题（应该尽快修复）

### [P1-1] 大量端点的 `limit` 参数无上限校验
- **端点**：GET /api/memory/query, GET /api/memory/archive, GET /api/memory/drafts, GET /api/memory/shared 等
- **行号**：L1332, L1595, L1707, L1919, L2059, L2838, L5758, L5873
- **问题**：用户可传 `limit=999999999`，导致后端加载大量数据到内存。虽然 `_safe_int` 处理了非数字输入，但没有对合法整数做上限约束。
- **建议修复**：添加统一的 limit 校验函数，如 `limit = min(limit, MAX_LIMIT)` 其中 MAX_LIMIT 设为合理值（如 1000）。

### [P1-2] 错误响应泄露内部异常信息
- **端点**：几乎所有 handler 的 catch-all `except Exception as e`
- **行号**：L223, L273, L312, L587, L665, L967, L982, L1632, L1676, L5377, L5965, L6004 等（共 40+ 处）
- **问题**：`str(e)` 或 `f"内部错误: {e}"` 直接返回给客户端，可能泄露文件路径、数据库表名、堆栈片段等内部实现细节。例如 L967 `f"Write failed: {e}"` 可能暴露服务器文件系统路径，L982 `f"DB update failed: {e}"` 可能暴露数据库连接信息。
- **建议修复**：对外统一返回 "内部错误，请联系管理员"，将 `str(e)` 仅写入日志（已有 `logger.exception` 在做）。可添加请求 ID 方便排查。

### [P1-3] `_handle_memory_update` 调用内部方法 `store._resolve_to_character_name`
- **端点**：PUT /api/memory/{entry_id}
- **行号**：L1404, L1448
- **问题**：直接调用以 `_` 开头的私有方法，这是不稳定的 API 调用。如果 `MemoryStore` 的实现变更，这里会静默 break。L1449 同样调用了 `store._employee_file(employee)`。
- **建议修复**：将 `_resolve_to_character_name` 提升为公共方法，或在 handler 层做名称解析。

### [P1-4] `_handle_memory_similar` 从 URL path 提取 memory_id 而非 path_params
- **端点**：GET /api/memory/semantic/similar/{memory_id}
- **行号**：L2834-L2835
- **问题**：`memory_id = path.split("/")[-1]` 从 URL 手动解析，而非使用 `request.path_params["memory_id"]`。如果 URL 末尾有查询参数或路径变化，会解析错误。同样的问题在 L2950-2951（feedback get）和 L2994-2995（usage stats）。
- **建议修复**：统一使用 `request.path_params["memory_id"]`。

### [P1-5] `_handle_memory_archive_restore` 缺少管理员权限校验
- **端点**：POST /api/memory/archive/restore
- **行号**：L1954-L2002
- **问题**：恢复归档记忆是写操作（将记忆从归档状态恢复到活跃），但没有 `_require_admin_token` 校验。对比同模块的 `_handle_memory_batch_update`（L2346）和 `_handle_memory_batch_delete`（L2478）都有 admin 校验。
- **建议修复**：添加 `admin_err = _require_admin_token(request)` 校验。

### [P1-6] `_handle_memory_feedback_submit` 缺少管理员权限校验
- **端点**：POST /api/memory/feedback
- **行号**：L2859-L2927
- **问题**：任何 Bearer token 持有者都可以提交反馈，`submitted_by` 字段完全由客户端控制可伪造。攻击者可以批量提交 "incorrect" 反馈来操纵记忆质量评分。
- **建议修复**：至少从认证上下文获取 `submitted_by`（类似 L2797-2798 drafts_approve 的做法），或要求 admin token。

### [P1-7] `_handle_memory_usage_record` 缺少权限校验
- **端点**：POST /api/memory/usage/record
- **行号**：L3023-L3068
- **问题**：任何 Bearer token 持有者都可以记录记忆使用统计，没有任何权限校验。攻击者可以伪造使用数据，影响热门/低质量记忆的排名。
- **建议修复**：添加 admin token 校验或调用者身份验证。

### [P1-8] `_handle_openclaw` 和 `_handle_generic` 缺少输入验证
- **端点**：POST /webhooks/openclaw, POST /webhooks/generic
- **行号**：L3252-L3303
- **问题**：`target_type` 参数直接传给 `_dispatch_task`，没有白名单校验。攻击者可以传入任意 `target_type`。`args` 也没有任何结构校验。虽然有 Bearer token 认证，但对于多租户场景，非 admin 租户不应能调度任意 pipeline/employee。
- **建议修复**：对 `target_type` 做白名单校验（允许值：employee, pipeline, discussion, chain），对 `target_name` 做存在性校验。

### [P1-9] `_handle_employee_state` 的 `sort_by` 参数未校验
- **端点**：GET /api/employees/{identifier}/state
- **行号**：L813
- **问题**：`sort_by = request.query_params.get("sort_by", "created_at")` 直接传给 `store.query()`。如果 store 实现是 SQL 后端，未经校验的 `sort_by` 可能构成 SQL 注入向量（取决于 store 实现的防护）。
- **建议修复**：添加白名单 `if sort_by not in ("created_at", "importance", "updated_at"): sort_by = "created_at"`。

### [P1-10] 响应格式不一致
- **端点**：全局
- **行号**：多处
- **问题**：部分端点使用 `_ok_response`/`_error_response`（返回 `{"ok": true/false, ...}`），部分直接用 `JSONResponse({"error": ...})`（无 `ok` 字段）。例如：
  - L784: `JSONResponse({"error": "Employee not found"}, status_code=404)` — 无 `ok` 字段
  - L1660: `JSONResponse({"ok": False, "error": "entry_id is required"}, ...` — 有 `ok` 字段
  - L4988: `_error_response` 使用 `{"ok": False, "error": ...}`
  - L346-347: `JSONResponse({"error": admin_err}, status_code=403)` — 无 `ok` 字段

  客户端需要同时处理两种格式，增加复杂度。
- **建议修复**：逐步迁移所有端点使用 `_ok_response`/`_error_response`（已定义在 L111-125）。

### [P1-11] `_handle_chat` 函数过长（450+ 行），分支路径复杂
- **端点**：POST /api/chat
- **行号**：L4984-L5420
- **问题**：单个函数超过 430 行，包含 Skills 触发、SG Bridge、SSH Bridge、agent tools、stream/non-stream、context_only 等 6+ 条执行路径。极难维护和测试。各路径的错误处理不一致，部分路径没有轨迹录制。
- **建议修复**：将各执行路径提取为独立函数（如 `_chat_via_sg_bridge`, `_chat_via_agent_tools`, `_chat_via_engine`），在 `_handle_chat` 中只做路由判断。

### [P1-12] `_handle_run_employee` 函数过长（330+ 行），与 `_handle_chat` 大量重复
- **端点**：POST /api/employees/{name}/run
- **行号**：L3546-L3877
- **问题**：Skills 触发逻辑（L3571-L3651）与 `_handle_chat`（L5050-L5124）几乎完全重复。fast path/full path 判断逻辑也重复。修改一处容易忘记另一处。
- **建议修复**：提取共享逻辑到公共函数（如 `_execute_with_skills`, `_choose_execution_path`）。

---

## P2 问题（建议改进）

### [P2-1] 大量静默吞掉异常的 `except Exception: pass`
- **行号**：L388-389, L722-723, L734-735, L831-832, L1302-1303, L1436-1437, L1510-1511, L3473-3474, L4517-4518, L4648-4649, L6217-6218, L6338-6339
- **问题**：共 12 处 `except Exception: pass`。虽然多数是非关键路径（缓存失效、embedding 更新等），但完全没有日志输出，排查问题时无迹可循。
- **建议修复**：改为 `except Exception: logger.debug(...)` 或 `logger.warning(...)`。

### [P2-2] 大量函数内 lazy import 模式
- **行号**：几乎每个 handler 的函数体开头
- **问题**：`from starlette.responses import JSONResponse` 在 70+ 个函数中重复导入。`from crew.memory import get_memory_store` 在 20+ 个函数中重复导入。这导致文件可读性下降，且（微弱的）导入性能开销。
- **建议修复**：将高频导入移到模块级别（`JSONResponse` 尤其应该是模块级导入），仅对真正有循环依赖风险的保留 lazy import。

### [P2-3] `_handle_memory_add` 幂等检查有性能问题
- **端点**：POST /api/memory/add
- **行号**：L1253-L1272
- **问题**：幂等检查通过 `store.query(employee, limit=50)` 加载最近 50 条记忆到内存遍历，而非数据库级别的唯一约束。随着记忆增长，这个方案不可扩展。
- **建议修复**：在 DB 版使用 `SELECT EXISTS(... WHERE source_session = ? AND category = ?)` 查询。

### [P2-4] `_handle_kv_list` 使用 `rglob("*")` 遍历所有文件
- **端点**：GET /api/kv/
- **行号**：L5570
- **问题**：`scan_dir.rglob("*")` 会递归遍历所有子目录的所有文件。如果 KV 存储中文件数量增长，这个操作会变慢。没有分页支持。
- **建议修复**：添加 `limit` 参数限制返回数量，或改为浅层 `iterdir()`。

### [P2-5] `_handle_audit_trends` 遍历全量任务快照
- **端点**：GET /api/audit/trends
- **行号**：L4930
- **问题**：`ctx.registry.snapshot()` 返回所有任务记录，然后在 Python 中做日期过滤。随着任务累积，这个操作会越来越慢。
- **建议修复**：给 `registry.snapshot()` 添加 `since` 参数，在存储层做过滤。

### [P2-6] `_handle_project_status` 做了 3 次外部 HTTP 请求
- **端点**：GET /api/project/status
- **行号**：L4720-4747
- **问题**：串行调用 `fetch_aiberm_billing`、`fetch_aiberm_balance`、`fetch_moonshot_balance`。如果任一外部服务超时，整个端点响应时间会叠加到 45s+。没有整体超时控制。
- **建议修复**：用 `asyncio.gather` 并发执行外部请求，或添加整体超时上限。

### [P2-7] `_handle_trajectory_report` 索引文件存在并发写入风险
- **端点**：POST /api/trajectory/report
- **行号**：L4642-L4666
- **问题**：`index.json` 的读-改-写操作没有文件锁。并发上报轨迹时可能出现数据丢失（后写覆盖前写）。
- **建议修复**：使用 `fcntl.flock` 或 `crew.paths.file_lock` 保护 index.json 的写操作。

### [P2-8] `_handle_employee_delete` 使用 `shutil.rmtree` 不检查符号链接
- **端点**：DELETE /api/employees/{identifier}
- **行号**：L1031-L1032
- **问题**：`if source.is_dir(): shutil.rmtree(source)` 不检查 `source` 是否为符号链接。根据 CLAUDE.md 硬规则，`rm -rf` 前必须检查符号链接。`shutil.rmtree` 会跟随符号链接删除目标内容。
- **建议修复**：在 `rmtree` 之前添加 `if source.is_symlink(): source.unlink()` 分支。

### [P2-9] `_handle_team_agents` 在非 admin 时仍做 discover_employees
- **端点**：GET /api/team/agents
- **行号**：L704
- **问题**：`discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))` 在匿名访问（无 admin token）时调用了 `_tenant_id_for_config(request)`。匿名请求经过 middleware skip_paths 跳过认证，`get_current_tenant(request)` 可能返回默认值，导致返回 admin 租户的员工列表而非公开列表。
- **建议修复**：匿名访问时硬编码使用 admin 租户 ID（公开展示数据来源应固定），避免依赖请求上下文。

### [P2-10] `_handle_memory_drafts_reject` 的 Content-Type 检查不完整
- **端点**：POST /api/memory/drafts/{draft_id}/reject
- **行号**：L1868
- **问题**：`request.headers.get("content-type") == "application/json"` 做精确匹配。如果客户端发送 `application/json; charset=utf-8`，这个条件为 false，body 会被解析为空字典。同样的模式出现在 L1971-1972（archive restore）、L2105-2106（shared usage）、L2352-2353（batch update）、L2484-2485（batch delete）、L3316-3317（run pipeline）、L3551-3552（run employee）、L3894-3895（run route）。
- **建议修复**：改为 `if "application/json" in (request.headers.get("content-type") or "")`。

### [P2-11] 多处重复的 Skills 触发逻辑
- **行号**：L3571-L3651（`_handle_run_employee`）、L5050-L5124（`_handle_chat`）
- **问题**：两段 Skills 触发代码几乎完全一致（约 70 行重复），包括 check_triggers → execute_skill → record_trigger → enhanced_context 合并。修改一处容易忘记另一处。
- **建议修复**：提取为 `async def _execute_skills(ctx, employee_name, message, args, channel, ...)` 公共函数。

### [P2-12] `_handle_employee_prompt` 返回的 `tool_schemas` 可能很大
- **端点**：GET /api/employees/{identifier}/prompt
- **行号**：L425
- **问题**：`tool_schemas` 包含所有工具的完整 JSON Schema 定义，可能达到几十 KB。而此接口同时返回 `system_prompt`（也可能很长），整体响应可能超过 100KB。
- **建议修复**：考虑添加 `?fields=` 参数允许客户端按需获取字段，或分离 tool_schemas 到独立端点。

### [P2-13] `_handle_memory_shared_list` 在查询后再做 category 过滤
- **端点**：GET /api/memory/shared
- **行号**：L2071-L2073
- **问题**：注释说 "query_shared 不支持 category 参数"，所以先查出所有数据再在 Python 中过滤。如果共享记忆数量大，这是浪费。
- **建议修复**：给 `query_shared` 添加 `category` 参数支持。

### [P2-14] `_write_yaml_field` 的 tempfile 不够安全
- **行号**：L128-L153
- **问题**：使用 `tempfile.mkstemp` + `os.replace` 是正确的原子写入模式，但 `os.write(fd, content.encode("utf-8"))` 一次性写入全部内容，如果内容很大且磁盘满，可能写入不完整但 `os.replace` 仍然执行。
- **建议修复**：写入后检查字节数是否与预期一致，或在 replace 前 `os.fsync(fd)` 确保落盘。

### [P2-15] `_handle_employee_state` 笔记读取逻辑硬编码路径
- **端点**：GET /api/employees/{identifier}/state
- **行号**：L855-L864
- **问题**：笔记目录 `.crew/notes/*.md` 是硬编码路径，且只读取最新 5 个文件。文件内容匹配用简单的字符串包含检查（`employee.character_name in text`），可能误匹配。
- **建议修复**：可接受，但应加注释说明设计意图。

### [P2-16] `_handle_memory_add` 的相似度检测是异步操作但使用 await
- **端点**：POST /api/memory/add
- **行号**：L1219-L1225
- **问题**：`find_similar_memories` 是 async 函数，在记忆写入的热路径上做 embedding 计算和相似度比较，增加了响应延迟。如果 embedding 服务不可用，会导致记忆写入超时。
- **建议修复**：可以考虑将相似度检测改为后台异步检查（先写入再检查），或添加超时保护。

### [P2-17] 文件 6372 行，应该拆分
- **行号**：全文件
- **问题**：单文件 6372 行、105+ 个函数，远超可维护范围。按功能域可拆分为：
  - `webhook_tenant.py` — 租户 CRUD（~130 行）
  - `webhook_employee.py` — 员工管理（~560 行）
  - `webhook_memory.py` — 记忆相关（~2500 行）
  - `webhook_trajectory.py` — 轨迹相关（~400 行）
  - `webhook_config_api.py` — 配置存储（~400 行）
  - `webhook_chat.py` — 对话接口（~500 行）
  - `webhook_handlers.py` — 其他（~500 行）
- **建议修复**：逐步拆分，优先拆出最大的记忆模块。

---

## 按维度汇总

### 输入验证
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P1-1 | P1 | limit 参数无上限校验（12+ 处） |
| P1-8 | P1 | openclaw/generic 的 target_type 无白名单 |
| P1-9 | P1 | sort_by 参数未校验（潜在注入） |
| P2-10 | P2 | Content-Type 精确匹配导致请求解析失败（9 处） |

总体评价：必填参数校验覆盖较好，但数值范围校验（limit、offset）和枚举值校验（sort_by、target_type）普遍缺失。

### 错误处理
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P1-2 | P1 | 40+ 处 `str(e)` 直接返回客户端 |
| P2-1 | P2 | 12 处 `except Exception: pass` 静默吞掉错误 |
| P2-14 | P2 | _write_yaml_field 写入不完整风险 |

总体评价：异常捕获覆盖率高（没有 bare except），但错误信息外泄是系统性问题。

### 数据泄露
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P1-2 | P1 | 错误响应泄露内部异常（文件路径、DB 信息） |
| P2-9 | P2 | team_agents 匿名访问可能返回非公开员工列表 |

总体评价：安全加固后数据泄露风险大幅降低，员工列表已做 admin/非 admin 字段分级，soul/prompt 等敏感内容已限 admin。主要残余风险在错误信息。

### 性能隐患
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P0-3 | P0 | dashboard 全量加载所有员工记忆 |
| P1-1 | P1 | limit 无上限可触发大量数据加载 |
| P2-3 | P2 | 幂等检查用内存遍历而非 DB 查询 |
| P2-4 | P2 | KV list 无分页的文件遍历 |
| P2-5 | P2 | audit trends 全量快照遍历 |
| P2-6 | P2 | project status 串行外部 HTTP 请求 |
| P2-13 | P2 | shared list 查询后内存过滤 |
| P2-16 | P2 | 记忆写入热路径上的 embedding 计算 |

总体评价：多个端点在数据量增长后存在性能退化风险。核心问题是缺乏 DB 层聚合查询，过多依赖 Python 内存操作。

### 一致性
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P1-10 | P1 | 错误响应格式不统一（有/无 ok 字段） |
| P2-2 | P2 | 大量重复的 lazy import |
| P1-4 | P1 | path 解析方式不一致（path_params vs split） |

总体评价：已定义了 `_ok_response`/`_error_response` 统一格式函数，但仅部分端点使用。迁移进度约 30%。

### 死代码 & 冗余
| 编号 | 严重度 | 描述 |
|------|--------|------|
| P2-11 | P2 | Skills 触发逻辑在两个 handler 中完全重复 |
| P1-11 | P1 | _handle_chat 430+ 行过长 |
| P1-12 | P1 | _handle_run_employee 与 _handle_chat 大量重复 |
| P2-17 | P2 | 整个文件 6372 行需要拆分 |

总体评价：无明显死代码（路由注册和 handler 对应关系完整），但冗余和重复是最突出的可维护性问题。

---

## 修复优先级建议

1. **立即修复**（P0）：`_run_and_callback` 的 request 引用 bug 是运行时错误，异步回调模式目前实际上是坏的（如果触发了 fast/full path 判断的 `_tenant_id_for_config` 分支）。后台任务 GC 风险也需要立即修复。
2. **本周修复**（P1-5, P1-6, P1-7）：缺失的权限校验应在下一次部署前补上。
3. **近期改进**（P1-1, P1-2, P1-10）：limit 上限、错误信息脱敏、响应格式统一可作为一个批次修复。
4. **中期重构**（P2-11, P2-17, P1-11, P1-12）：Skills 去重、文件拆分、chat/run_employee 合并是中期架构改进目标。
