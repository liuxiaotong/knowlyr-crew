SERVER := knowlyr-web-1
REMOTE_DIR := /opt/knowlyr-crew/project
LOCAL_CREW := .crew
LOCAL_EMPLOYEES := private/employees

# ============================================================
# 训练循环: pull → 本地分析/改进 → push → test
# ============================================================

## 第1步: 从服务器拉取训练数据
pull:
	@echo "=== 拉取轨迹数据 ==="
	rsync -avz $(SERVER):$(REMOTE_DIR)/$(LOCAL_CREW)/trajectories/ $(LOCAL_CREW)/trajectories/
	@echo ""
	@echo "=== 拉取会话日志 ==="
	rsync -avz $(SERVER):$(REMOTE_DIR)/$(LOCAL_CREW)/sessions/ $(LOCAL_CREW)/sessions/
	@echo ""
	@echo "=== 拉取记忆 ==="
	rsync -avz $(SERVER):$(REMOTE_DIR)/$(LOCAL_CREW)/memory/ $(LOCAL_CREW)/memory/
	@echo ""
	@wc -l $(LOCAL_CREW)/trajectories/*.jsonl 2>/dev/null || true
	@ls $(LOCAL_CREW)/sessions/*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "Sessions: {} 个"
	@echo "拉取完成。用 Claude Code 分析后修改 prompt，然后 make push"

## 第2步: 一键部署员工（推送 + 同步 project-dir + 重启 + 同步到 knowlyr-id）
push:
	@echo "=== 推送员工配置到 private/ ==="
	rsync -avz $(LOCAL_EMPLOYEES)/ $(SERVER):/opt/knowlyr-crew/private/employees/
	@echo ""
	@echo "=== 同步 private/ → project-dir ==="
	ssh $(SERVER) "mkdir -p $(REMOTE_DIR)/$(LOCAL_EMPLOYEES) && \
		cp /opt/knowlyr-crew/private/organization.yaml $(REMOTE_DIR)/private/organization.yaml && \
		rsync -a /opt/knowlyr-crew/private/employees/ $(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/"
	@echo ""
	@echo "=== 重启 crew 服务 ==="
	ssh $(SERVER) "sudo systemctl restart knowlyr-crew"
	@sleep 2
	@echo "=== 同步到 knowlyr-id ==="
	ssh $(SERVER) "set -a && source /opt/knowlyr-crew/.env && set +a && \
		/opt/knowlyr-crew/venv/bin/knowlyr-crew agents sync-all \
		--dir $(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/ --force"
	@echo ""
	@echo "=== 部署完成 ==="

## 只推送文件，不重启不同步
push-only:
	rsync -avz $(LOCAL_EMPLOYEES)/ $(SERVER):/opt/knowlyr-crew/private/employees/
	@echo "文件已推送到 private/（未同步 project-dir、未重启、未同步 knowlyr-id）"

## 第3步: 在服务器上测试某个员工（走真实 webhook，调真实工具）
test-employee:
ifndef NAME
	$(error 用法: make test-employee NAME=ceo-assistant TASK="今天心情不错")
endif
ifndef TASK
	$(eval TASK := 你好)
endif
	@echo "=== 测试 $(NAME)（webhook 真实链路） ==="
	@ssh $(SERVER) 'set -a && source /opt/knowlyr-crew/.env && set +a && \
		curl -s -X POST http://localhost:8765/run/employee/$(NAME) \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$CREW_API_TOKEN" \
		-d "{\"args\": {\"task\": \"$(TASK)\"}, \"sync\": true, \"agent_id\": 3073}" \
		| python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get(\"result\",d); print(r.get(\"output\",\"(no output)\")); print(\"  [model]\", r.get(\"model\",\"?\")); print(\"  [tokens]\", r.get(\"input_tokens\",0), \"in /\", r.get(\"output_tokens\",0), \"out\"); print(\"  [rounds]\", r.get(\"tool_rounds\",0))"'

# ============================================================
# 快捷命令
# ============================================================

## 推送单个员工
push-employee:
ifndef NAME
	$(error 用法: make push-employee NAME=姜墨言-3073)
endif
	rsync -avz $(LOCAL_EMPLOYEES)/$(NAME)/ $(SERVER):/opt/knowlyr-crew/private/employees/$(NAME)/
	ssh $(SERVER) "mkdir -p $(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/$(NAME) && \
		rsync -a /opt/knowlyr-crew/private/employees/$(NAME)/ $(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/$(NAME)/"
	@echo "已推送 $(NAME)（private/ + project-dir）"

## 清理服务器上的脏数据
clean-memory:
ifndef NAME
	$(error 用法: make clean-memory NAME=ceo-assistant)
endif
	ssh $(SERVER) '\
		cp $(REMOTE_DIR)/$(LOCAL_CREW)/memory/$(NAME).jsonl \
		   $(REMOTE_DIR)/$(LOCAL_CREW)/memory/$(NAME).jsonl.bak 2>/dev/null; \
		echo "" > $(REMOTE_DIR)/$(LOCAL_CREW)/memory/$(NAME).jsonl'
	@echo "已清理 $(NAME) 的记忆 (备份已保存)"

clean-trajectories:
	ssh $(SERVER) '\
		cp $(REMOTE_DIR)/$(LOCAL_CREW)/trajectories/trajectories.jsonl \
		   $(REMOTE_DIR)/$(LOCAL_CREW)/trajectories/trajectories.jsonl.bak 2>/dev/null; \
		echo "" > $(REMOTE_DIR)/$(LOCAL_CREW)/trajectories/trajectories.jsonl'
	@echo "已清理轨迹数据 (备份已保存)"

## 批量跑任务积累轨迹
batch:
	ssh $(SERVER) 'cd $(REMOTE_DIR) && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		bash scripts/batch_trajectory.sh'

## 用 conversation domain 对轨迹打分
score:
	ssh $(SERVER) 'cd $(REMOTE_DIR) && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		knowlyr-crew trajectory score --all'

## 查看轨迹列表
trajectories:
	ssh $(SERVER) 'cd $(REMOTE_DIR) && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		knowlyr-crew trajectory list'

## 部署 crew 引擎代码 (触发 GitHub Actions)
deploy-engine:
	@echo "推送代码到 GitHub，自动触发部署..."
	git push

## 升级服务器上的 agent 包 (从 git 安装最新版)
upgrade-agent:
	ssh $(SERVER) 'source /opt/knowlyr-crew/venv/bin/activate && \
		pip install --force-reinstall --no-deps \
		"knowlyr-core @ git+https://github.com/liuxiaotong/knowlyr-agent.git@main#subdirectory=packages/core" \
		"knowlyr-reward @ git+https://github.com/liuxiaotong/knowlyr-agent.git@main#subdirectory=packages/reward"'
	@echo "agent 包已升级"

# ============================================================
# 完整训练循环 (一键)
# ============================================================

## 完整循环: 拉数据 → 批跑 → 打分 → 拉回结果
train-cycle:
	@echo "=== 1/4 批跑任务 ==="
	$(MAKE) batch
	@echo ""
	@echo "=== 2/4 打分 ==="
	$(MAKE) score
	@echo ""
	@echo "=== 3/4 拉取数据到本地 ==="
	$(MAKE) pull
	@echo ""
	@echo "=== 4/4 完成 ==="
	@echo "数据已就绪。现在用 Claude Code 分析 .crew/trajectories/ 和 .crew/sessions/"
	@echo "改进 prompt 后运行: make push && make test-employee NAME=ceo-assistant TASK='你的测试任务'"

.PHONY: pull push test-employee push-employee clean-memory clean-trajectories \
        batch score trajectories deploy-engine upgrade-agent train-cycle
