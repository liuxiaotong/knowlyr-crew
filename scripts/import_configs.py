#!/usr/bin/env python3
"""配置导入脚本 — 从本地文件系统导入到数据库.

将 knowlyr-crew-private 中的配置一次性导入到数据库：
- employees/*/soul.md → employee_souls 表
- discussions/*.yaml → discussions 表
- pipelines/*.yaml → pipelines 表

用法:
    python scripts/import_configs.py /path/to/knowlyr-crew-private
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def import_souls(private_dir: Path) -> int:
    """导入员工灵魂配置."""
    from crew.config_store import update_soul

    employees_dir = private_dir / "employees"
    if not employees_dir.is_dir():
        logger.warning("未找到 employees 目录: %s", employees_dir)
        return 0

    count = 0
    for emp_dir in sorted(employees_dir.iterdir()):
        if not emp_dir.is_dir():
            continue

        soul_file = emp_dir / "soul.md"
        if not soul_file.is_file():
            logger.debug("跳过（无 soul.md）: %s", emp_dir.name)
            continue

        # 从目录名提取员工名（格式：姓名-ID）
        dir_name = emp_dir.name
        parts = dir_name.rsplit("-", 1)
        if len(parts) == 2:
            employee_name = parts[0]
        else:
            employee_name = dir_name

        content = soul_file.read_text(encoding="utf-8")
        try:
            result = update_soul(employee_name, content, updated_by="import_script")
            logger.info(
                "✓ 导入 soul: %s (version=%s)",
                employee_name,
                result["version"],
            )
            count += 1
        except Exception as exc:
            logger.error("✗ 导入 soul 失败: %s - %s", employee_name, exc)

    return count


def import_discussions(private_dir: Path) -> int:
    """导入讨论会配置."""
    from crew.config_store import create_discussion

    discussions_dir = private_dir / "discussions"
    if not discussions_dir.is_dir():
        logger.warning("未找到 discussions 目录: %s", discussions_dir)
        return 0

    count = 0
    for yaml_file in sorted(discussions_dir.glob("*.yaml")):
        name = yaml_file.stem
        yaml_content = yaml_file.read_text(encoding="utf-8")

        # 从 YAML 中提取 description（如果有）
        description = ""
        try:
            import yaml

            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict):
                description = data.get("description", "")
        except Exception:
            pass

        try:
            result = create_discussion(name, yaml_content, description)
            logger.info("✓ 导入 discussion: %s", name)
            count += 1
        except Exception as exc:
            # 如果已存在，尝试更新
            if "duplicate" in str(exc).lower() or "already exists" in str(exc).lower():
                try:
                    from crew.config_store import update_discussion

                    result = update_discussion(name, yaml_content, description)
                    logger.info("✓ 更新 discussion: %s", name)
                    count += 1
                except Exception as exc2:
                    logger.error("✗ 导入 discussion 失败: %s - %s", name, exc2)
            else:
                logger.error("✗ 导入 discussion 失败: %s - %s", name, exc)

    return count


def import_pipelines(private_dir: Path) -> int:
    """导入流水线配置."""
    from crew.config_store import create_pipeline

    pipelines_dir = private_dir / "pipelines"
    if not pipelines_dir.is_dir():
        logger.warning("未找到 pipelines 目录: %s", pipelines_dir)
        return 0

    count = 0
    for yaml_file in sorted(pipelines_dir.glob("*.yaml")):
        name = yaml_file.stem
        yaml_content = yaml_file.read_text(encoding="utf-8")

        # 从 YAML 中提取 description（如果有）
        description = ""
        try:
            import yaml

            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict):
                description = data.get("description", "")
        except Exception:
            pass

        try:
            result = create_pipeline(name, yaml_content, description)
            logger.info("✓ 导入 pipeline: %s", name)
            count += 1
        except Exception as exc:
            # 如果已存在，尝试更新
            if "duplicate" in str(exc).lower() or "already exists" in str(exc).lower():
                try:
                    from crew.config_store import update_pipeline

                    result = update_pipeline(name, yaml_content, description)
                    logger.info("✓ 更新 pipeline: %s", name)
                    count += 1
                except Exception as exc2:
                    logger.error("✗ 导入 pipeline 失败: %s - %s", name, exc2)
            else:
                logger.error("✗ 导入 pipeline 失败: %s - %s", name, exc)

    return count


def main():
    parser = argparse.ArgumentParser(description="导入配置到数据库")
    parser.add_argument(
        "private_dir",
        type=Path,
        help="knowlyr-crew-private 目录路径",
    )
    parser.add_argument(
        "--souls-only",
        action="store_true",
        help="仅导入员工灵魂配置",
    )
    parser.add_argument(
        "--discussions-only",
        action="store_true",
        help="仅导入讨论会配置",
    )
    parser.add_argument(
        "--pipelines-only",
        action="store_true",
        help="仅导入流水线配置",
    )

    args = parser.parse_args()

    if not args.private_dir.is_dir():
        logger.error("目录不存在: %s", args.private_dir)
        sys.exit(1)

    # 检查数据库连接
    from crew.database import is_pg

    if not is_pg():
        logger.error("配置存储仅支持 PostgreSQL，请设置 CREW_DATABASE_URL 环境变量")
        sys.exit(1)

    # 初始化数据库表
    from crew.config_store import init_config_tables

    try:
        init_config_tables()
        logger.info("数据库表初始化完成")
    except Exception as exc:
        logger.error("数据库表初始化失败: %s", exc)
        sys.exit(1)

    # 执行导入
    total = 0

    if args.souls_only:
        total += import_souls(args.private_dir)
    elif args.discussions_only:
        total += import_discussions(args.private_dir)
    elif args.pipelines_only:
        total += import_pipelines(args.private_dir)
    else:
        # 全部导入
        total += import_souls(args.private_dir)
        total += import_discussions(args.private_dir)
        total += import_pipelines(args.private_dir)

    logger.info("=" * 60)
    logger.info("导入完成，共 %d 条配置", total)


if __name__ == "__main__":
    main()
