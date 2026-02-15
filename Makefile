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

## 第2步: 将改进后的员工配置推送到服务器
push:
	@echo "=== 推送员工配置 ==="
	rsync -avz $(LOCAL_EMPLOYEES)/ $(SERVER):$(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/
	@echo ""
	@echo "=== 推送完成 ==="
	@echo "员工配置已更新。运行 make test-employee NAME=ceo-assistant 验证"

## 第3步: 在服务器上测试某个员工
test-employee:
ifndef NAME
	$(error 用法: make test-employee NAME=ceo-assistant TASK="今天心情不错")
endif
ifndef TASK
	$(eval TASK := 你好)
endif
	@echo "=== 测试 $(NAME) ==="
	ssh $(SERVER) 'cd $(REMOTE_DIR) && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		knowlyr-crew run $(NAME) --execute --arg task="$(TASK)"'

# ============================================================
# 快捷命令
# ============================================================

## 推送单个员工
push-employee:
ifndef NAME
	$(error 用法: make push-employee NAME=姜墨言-3073)
endif
	rsync -avz $(LOCAL_EMPLOYEES)/$(NAME)/ $(SERVER):$(REMOTE_DIR)/$(LOCAL_EMPLOYEES)/$(NAME)/
	@echo "已推送 $(NAME)"

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
