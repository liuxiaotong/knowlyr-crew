"""飞书对话记忆 — 维护每个 chat_id 的对话历史 + 长期知识沉淀."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 每个聊天保留的最大消息数（超出后截断旧消息）
_MAX_MESSAGES_PER_CHAT = 200


class FeishuChatStore:
    """基于文件的飞书对话历史存储.

    存储结构: {store_dir}/{chat_id}.jsonl
    每行: {"role": "user"|"assistant", "content": "...", "ts": "..."}
    """

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir

    def _ensure_dir(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _chat_file(self, chat_id: str) -> Path:
        safe_id = chat_id.replace("/", "_").replace("..", "_")
        return self.store_dir / f"{safe_id}.jsonl"

    def append(
        self,
        chat_id: str,
        role: str,
        content: str,
        sender_name: str = "",
        **extra: str,
    ) -> None:
        """追加一条消息.

        额外字段通过 ``**extra`` 传入，如 ``path="full"`` 记录使用的路径。
        """
        self._ensure_dir()
        entry: dict[str, str] = {
            "role": role,
            "content": content,
            "ts": datetime.now().isoformat(),
        }
        if sender_name:
            entry["sender_name"] = sender_name
        for k, v in extra.items():
            if v is not None:
                entry[k] = v
        with self._chat_file(chat_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent(self, chat_id: str, limit: int = 20) -> list[dict]:
        """获取最近 N 条消息（按时间正序）."""
        path = self._chat_file(chat_id)
        if not path.exists():
            return []

        lines = path.read_text(encoding="utf-8").splitlines()
        entries: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= limit:
                break
        entries.reverse()
        return entries

    def format_for_prompt(self, chat_id: str, limit: int = 20) -> str:
        """格式化对话历史为 prompt 文本."""
        entries = self.get_recent(chat_id, limit=limit)
        if not entries:
            return ""
        lines = []
        for entry in entries:
            if entry["role"] == "user":
                role_label = entry.get("sender_name") or "Kai"
            else:
                role_label = "你"
            lines.append(f"{role_label}: {entry['content']}")
        return "\n".join(lines)


def capture_feishu_memory(
    *,
    project_dir: Path,
    employee_name: str,
    chat_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """记录飞书交互到 session 并尝试沉淀为长期记忆."""
    try:
        from crew.session_recorder import SessionRecorder

        recorder = SessionRecorder(project_dir=project_dir)
        sid = recorder.start(
            session_type="feishu",
            subject=employee_name,
            metadata={"chat_id": chat_id, "employee": employee_name},
        )
        recorder.record_message(sid, "user", user_text, metadata={"employee": employee_name})
        recorder.record_message(
            sid, "assistant", assistant_text, metadata={"employee": employee_name}
        )
        recorder.finish(sid)

        # 尝试沉淀长期记忆
        from crew.session_summary import SessionMemoryWriter

        writer = SessionMemoryWriter(project_dir=project_dir)
        writer.capture(employee=employee_name, session_id=sid)
    except Exception as e:
        logger.debug("飞书记忆沉淀失败: %s", e)
