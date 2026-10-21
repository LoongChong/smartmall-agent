"""会话存储（SQLite）。

数据隔离：每条会话绑定 ``user_id``，读取时强制校验归属——
知道 session_id 也不能读别人的会话。
历史轮数有上限（``session_max_turns``），超出后丢弃最旧的轮次，防止无界增长。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from ..errors import ForbiddenError, SessionNotFoundError
from ..logging_setup import get_logger
from ..models import ChatTurn, utcnow

logger = get_logger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, seq);
"""


class SessionStore:
    def __init__(self, conn: aiosqlite.Connection, *, max_turns: int = 10) -> None:
        self._conn = conn
        self._lock = asyncio.Lock()
        self._max_turns = max_turns

    @classmethod
    async def open(cls, db_path: Path, *, max_turns: int = 10) -> SessionStore:
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await conn.executescript(SCHEMA)
        await conn.commit()
        return cls(conn, max_turns=max_turns)

    async def close(self) -> None:
        await self._conn.close()

    async def ensure_session(self, session_id: str, user_id: str | None) -> None:
        """创建或校验会话。已存在但 user_id 不匹配时抛 ForbiddenError。"""
        async with self._lock:
            cur = await self._conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
            row = await cur.fetchone()
            now = utcnow().isoformat()
            if row is None:
                await self._conn.execute(
                    "INSERT INTO sessions (session_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, user_id, now, now),
                )
            else:
                owner = row["user_id"]
                if owner != user_id:
                    logger.warning(
                        "session_owner_mismatch",
                        extra={"session": session_id, "owner": owner, "caller": user_id},
                    )
                    raise ForbiddenError("该会话不属于当前用户")
                await self._conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            await self._conn.commit()

    async def append_turn(self, session_id: str, turn: ChatTurn) -> None:
        async with self._lock:
            cur = await self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM turns WHERE session_id = ?", (session_id,)
            )
            row = await cur.fetchone()
            seq = (row["m"] if row else -1) + 1
            await self._conn.execute(
                "INSERT INTO turns (session_id, seq, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, turn.role, turn.content, turn.created_at.isoformat()),
            )
            # 裁剪历史：只保留最近 max_turns*2 条（user + assistant 各一条算一轮）
            keep = self._max_turns * 2
            await self._conn.execute(
                """DELETE FROM turns WHERE session_id = ? AND seq <= ?""",
                (session_id, seq - keep),
            )
            await self._conn.commit()

    async def history(self, session_id: str, user_id: str | None) -> list[ChatTurn]:
        """读取历史。会话归属不符直接拒绝，而不是返回空列表。"""
        cur = await self._conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = await cur.fetchone()
        if row is None:
            raise SessionNotFoundError(f"会话不存在: {session_id}")
        if row["user_id"] != user_id:
            raise ForbiddenError("该会话不属于当前用户")

        cur = await self._conn.execute(
            "SELECT role, content, created_at FROM turns WHERE session_id = ? ORDER BY seq", (session_id,)
        )
        from datetime import datetime

        return [
            ChatTurn(role=r["role"], content=r["content"], created_at=datetime.fromisoformat(r["created_at"]))
            for r in await cur.fetchall()
        ]

    async def turn_count(self, session_id: str) -> int:
        cur = await self._conn.execute("SELECT COUNT(*) AS c FROM turns WHERE session_id = ?", (session_id,))
        row = await cur.fetchone()
        return row["c"] if row else 0
