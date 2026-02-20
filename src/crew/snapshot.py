"""数据快照管理 — 记录某一刻的 prompt 状态 + 数据清单.

快照 = prompt 硬拷贝 + 数据计数，用于评估版本对比。

用法:
    from crew.snapshot import SnapshotManager
    sm = SnapshotManager(crew_root)
    version = sm.create("墨言 prompt 优化前基线")
    sm.list_snapshots()
    sm.diff("v20260216a", "v20260217a")
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SnapshotManager:
    """管理 .crew/snapshots/ 下的数据快照."""

    def __init__(self, crew_root: Path):
        self.crew_root = crew_root
        self.snapshots_dir = crew_root / ".crew" / "snapshots"
        self.global_dir = crew_root / ".crew" / "global"
        self.sessions_dir = crew_root / ".crew" / "sessions"
        self.meetings_dir = crew_root / ".crew" / "meetings"
        self.memory_dir = crew_root / ".crew" / "memory"
        self.logs_dir = crew_root / ".crew" / "logs"

    # ------------------------------------------------------------------
    # 版本号生成
    # ------------------------------------------------------------------

    def _next_version(self) -> str:
        """生成下一个版本号: v + 日期 + 序号字母."""
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"v{today}"

        existing = (
            sorted(
                d.name
                for d in self.snapshots_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            )
            if self.snapshots_dir.exists()
            else []
        )

        if not existing:
            return f"{prefix}a"

        # 取最后一个的末尾字母 +1
        last_letter = existing[-1][-1]
        next_letter = chr(ord(last_letter) + 1)
        return f"{prefix}{next_letter}"

    # ------------------------------------------------------------------
    # 创建快照
    # ------------------------------------------------------------------

    def create(self, description: str = "") -> str:
        """创建一个快照.

        Args:
            description: 快照说明 (如 "墨言 prompt 优化前基线")

        Returns:
            版本号 (如 "v20260216a")
        """
        version = self._next_version()
        snap_dir = self.snapshots_dir / version
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 1. 拷贝所有员工的 prompt 文件
        prompts_dir = snap_dir / "prompts"
        employees_info = self._copy_prompts(prompts_dir)

        # 2. 收集数据计数
        data_counts = self._count_data()

        # 3. 写 manifest
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": description,
            "employees": employees_info,
            "data_counts": data_counts,
        }

        manifest_path = snap_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("快照创建: %s (%d 个员工)", version, len(employees_info))
        return version

    def _copy_prompts(self, dest_dir: Path) -> dict:
        """拷贝所有员工的 prompt 文件，返回员工信息 dict."""
        import yaml

        employees_info = {}

        if not self.global_dir.exists():
            return employees_info

        for emp_dir in sorted(self.global_dir.iterdir()):
            if not emp_dir.is_dir():
                continue

            yaml_path = emp_dir / "employee.yaml"
            if not yaml_path.exists():
                continue

            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                continue

            emp_name = config.get("name", emp_dir.name)
            char_name = config.get("character_name", "")
            version = str(config.get("version", "1.0"))
            content_hash = config.get("_content_hash", "")
            model = config.get("model", "")

            # 拷贝文件
            emp_dest = dest_dir / emp_dir.name
            emp_dest.mkdir(parents=True, exist_ok=True)

            # 必拷: employee.yaml, prompt.md, soul.md
            for fname in ["employee.yaml", "prompt.md", "soul.md"]:
                src = emp_dir / fname
                if src.exists():
                    shutil.copy2(src, emp_dest / fname)

            # 拷贝 workflows/ 和 adaptors/ 目录
            for subdir_name in ["workflows", "adaptors"]:
                subdir = emp_dir / subdir_name
                if subdir.is_dir():
                    dest_sub = emp_dest / subdir_name
                    if dest_sub.exists():
                        shutil.rmtree(dest_sub)
                    shutil.copytree(subdir, dest_sub)

            employees_info[emp_name] = {
                "display_name": char_name,
                "prompt_version": version,
                "content_hash": content_hash,
                "model": model,
            }

        return employees_info

    def _count_data(self) -> dict:
        """统计各类数据的数量."""
        counts = {}

        # sessions
        if self.sessions_dir.exists():
            all_sessions = list(self.sessions_dir.glob("*.jsonl"))
            counts["sessions_total"] = len(all_sessions)

            # 区分 organic/synthetic
            organic = 0
            synthetic = 0
            for f in all_sessions:
                first_line = f.read_text("utf-8").split("\n")[0]
                import json as _json

                start = _json.loads(first_line)
                source = start.get("metadata", {}).get("source", "")
                if source.startswith("cli."):
                    synthetic += 1
                else:
                    organic += 1
            counts["sessions_organic"] = organic
            counts["sessions_synthetic"] = synthetic

        # meetings
        if self.meetings_dir.exists():
            counts["meetings"] = len(list(self.meetings_dir.glob("*.md")))

        # memory (从各员工的 memory/ 子目录中计数)
        memory_count = 0
        if self.global_dir.exists():
            for emp_dir in self.global_dir.iterdir():
                mem_dir = emp_dir / "memory"
                if mem_dir.is_dir():
                    for mem_file in mem_dir.glob("*.jsonl"):
                        with open(mem_file, encoding="utf-8") as mf:
                            memory_count += sum(1 for line in mf if line.strip())
        counts["memory_entries"] = memory_count

        # logs
        if self.logs_dir.exists():
            counts["logs"] = len(list(self.logs_dir.glob("*")))

        return counts

    # ------------------------------------------------------------------
    # 列出快照
    # ------------------------------------------------------------------

    def list_snapshots(self) -> list[dict]:
        """列出所有快照，返回 manifest 列表（按时间排序）."""
        snapshots = []

        if not self.snapshots_dir.exists():
            return snapshots

        for snap_dir in sorted(self.snapshots_dir.iterdir()):
            if not snap_dir.is_dir():
                continue
            manifest_path = snap_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshots.append(manifest)

        return snapshots

    # ------------------------------------------------------------------
    # 对比快照
    # ------------------------------------------------------------------

    def diff(self, version_a: str, version_b: str) -> dict:
        """对比两个快照，返回变更信息.

        Returns:
            {
                "prompt_changes": [{"employee": ..., "version_a": ..., "version_b": ..., "hash_changed": bool}],
                "data_changes": {"sessions_organic": (a, b), ...},
                "new_employees": [...],
                "removed_employees": [...],
            }
        """
        manifest_a = self._load_manifest(version_a)
        manifest_b = self._load_manifest(version_b)

        if not manifest_a or not manifest_b:
            raise FileNotFoundError(f"找不到快照: {version_a if not manifest_a else version_b}")

        emps_a = manifest_a.get("employees", {})
        emps_b = manifest_b.get("employees", {})
        all_emps = set(emps_a.keys()) | set(emps_b.keys())

        prompt_changes = []
        for emp in sorted(all_emps):
            info_a = emps_a.get(emp, {})
            info_b = emps_b.get(emp, {})
            hash_a = info_a.get("content_hash", "")
            hash_b = info_b.get("content_hash", "")

            if hash_a != hash_b:
                prompt_changes.append(
                    {
                        "employee": emp,
                        "display_name": info_b.get("display_name")
                        or info_a.get("display_name", ""),
                        "version_a": info_a.get("prompt_version", "—"),
                        "version_b": info_b.get("prompt_version", "—"),
                        "hash_a": hash_a,
                        "hash_b": hash_b,
                    }
                )

        # 数据变化
        data_a = manifest_a.get("data_counts", {})
        data_b = manifest_b.get("data_counts", {})
        data_changes = {}
        for key in set(data_a.keys()) | set(data_b.keys()):
            va = data_a.get(key, 0)
            vb = data_b.get(key, 0)
            if va != vb:
                data_changes[key] = (va, vb)

        return {
            "version_a": version_a,
            "version_b": version_b,
            "prompt_changes": prompt_changes,
            "data_changes": data_changes,
            "new_employees": sorted(set(emps_b.keys()) - set(emps_a.keys())),
            "removed_employees": sorted(set(emps_a.keys()) - set(emps_b.keys())),
        }

    def _load_manifest(self, version: str) -> dict | None:
        """加载指定版本的 manifest."""
        manifest_path = self.snapshots_dir / version / "manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 获取当前状态的 content hash 指纹
    # ------------------------------------------------------------------

    def current_fingerprint(self) -> str:
        """计算当前所有员工的组合 content hash，用于检测是否需要新快照."""
        import hashlib

        from crew.versioning import compute_content_hash

        hasher = hashlib.sha256()
        if not self.global_dir.exists():
            return hasher.hexdigest()[:12]

        for emp_dir in sorted(self.global_dir.iterdir()):
            if not emp_dir.is_dir():
                continue
            emp_hash = compute_content_hash(emp_dir)
            hasher.update(emp_hash.encode())

        return hasher.hexdigest()[:12]

    def find_matching_snapshot(self) -> str | None:
        """找到与当前 prompt 状态匹配的快照版本（如果有）."""
        current_fp = self.current_fingerprint()

        for snap in reversed(self.list_snapshots()):
            # 重算该快照的指纹
            snap_emps = snap.get("employees", {})
            import hashlib

            hasher = hashlib.sha256()
            for emp_name in sorted(snap_emps.keys()):
                h = snap_emps[emp_name].get("content_hash", "")
                hasher.update(h.encode())
            snap_fp = hasher.hexdigest()[:12]

            if snap_fp == current_fp:
                return snap["version"]

        return None
