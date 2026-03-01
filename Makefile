SERVER := knowlyr-web-1

# ============================================================
# 部署（常规路径：git push → GitHub Actions 自动部署）
# ============================================================

## 推送代码触发 CI 部署（推前自动拉取远程变更）
deploy:
	git pull --ff-only origin main
	@echo "推送代码到 GitHub，自动触发部署..."
	git push origin main

## 紧急旁路：跳过 CI 直接更新引擎
emergency-engine:
	bash deploy/deploy.sh engine

## 紧急旁路：只重启服务
emergency-restart:
	bash deploy/deploy.sh restart

## 查看服务器状态
status:
	bash deploy/deploy.sh status

# ============================================================
# 运维
# ============================================================

## 注册新员工到 knowlyr-id（需指定 NAME）
register:
ifndef NAME
	$(error 用法: make register NAME=employee-name)
endif
	ssh $(SERVER) "set -a && source /opt/knowlyr-crew/.env && set +a && \
		/opt/knowlyr-crew/venv/bin/knowlyr-crew register $(NAME)"

## 同步员工信息到 knowlyr-id
id-sync:
	bash deploy/deploy.sh id-sync

## 在服务器上测试某个员工（走真实 webhook，调真实工具）
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

## 升级服务器上的 gym 包（从 git 安装最新版）
upgrade-agent:
	ssh $(SERVER) 'source /opt/knowlyr-crew/venv/bin/activate && \
		pip install --force-reinstall --no-deps \
		"knowlyr-core @ git+https://github.com/liuxiaotong/knowlyr-gym.git@main#subdirectory=packages/core" \
		"knowlyr-reward @ git+https://github.com/liuxiaotong/knowlyr-gym.git@main#subdirectory=packages/reward"'
	@echo "agent 包已升级"

# ============================================================
# 训练循环
# ============================================================

## 从服务器拉取轨迹数据到本地分析
pull:
	@echo "=== 拉取轨迹数据 ==="
	rsync -avz $(SERVER):/opt/knowlyr-crew/project/.crew/trajectories/ .crew/trajectories/
	@echo ""
	@echo "=== 拉取会话日志 ==="
	rsync -avz $(SERVER):/opt/knowlyr-crew/project/.crew/sessions/ .crew/sessions/
	@echo ""
	@wc -l .crew/trajectories/*.jsonl 2>/dev/null || true
	@ls .crew/sessions/*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "Sessions: {} 个"
	@echo "拉取完成。"

## 批量跑任务积累轨迹
batch:
	ssh $(SERVER) 'cd /opt/knowlyr-crew/project && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		bash scripts/batch_trajectory.sh'

## 用 conversation domain 对轨迹打分
score:
	ssh $(SERVER) 'cd /opt/knowlyr-crew/project && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		knowlyr-crew trajectory score --all'

## 查看轨迹列表
trajectories:
	ssh $(SERVER) 'cd /opt/knowlyr-crew/project && source /opt/knowlyr-crew/venv/bin/activate && \
		set -a && source /opt/knowlyr-crew/.env && set +a && \
		knowlyr-crew trajectory list'

## 完整训练循环: 批跑 → 打分 → 拉回结果
train-cycle:
	@echo "=== 1/3 批跑任务 ==="
	$(MAKE) batch
	@echo ""
	@echo "=== 2/3 打分 ==="
	$(MAKE) score
	@echo ""
	@echo "=== 3/3 拉取数据到本地 ==="
	$(MAKE) pull

# ============================================================
# 开发
# ============================================================

## 安装 git hooks（拦截 private/ 误提交）
install-hooks:
	git config core.hooksPath hooks
	@echo "已配置 git hooks 目录: hooks/"

.PHONY: deploy emergency-engine emergency-restart status \
        register id-sync test-employee upgrade-agent \
        pull batch score trajectories train-cycle install-hooks
