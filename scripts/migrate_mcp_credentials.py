#!/usr/bin/env python3
"""MCP credentials table migration.

Upgrades old table (server_name/encrypted_value) to new schema
(mcp_server/access_token/refresh_token/tenant_id/scopes).
Also adds tenant_id to mcp_audit_log.

Usage:
    python scripts/migrate_mcp_credentials.py --dry-run
    python scripts/migrate_mcp_credentials.py
"""
from __future__ import annotations
import argparse
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SAFE_TABLE_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _table_exists(cur, table_name, is_pg):
    if is_pg:
        cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = %s LIMIT 1", (table_name,))
    else:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,))
    return cur.fetchone() is not None


def _column_exists(cur, table_name, column_name, is_pg):
    if not _SAFE_TABLE_NAME.match(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    if is_pg:
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s LIMIT 1", (table_name, column_name))
        return cur.fetchone() is not None
    else:
        cur.execute(f"PRAGMA table_info({table_name})")
        return column_name in [row[1] for row in cur.fetchall()]


def migrate(dry_run=False):
    from crew.database import get_connection, is_pg
    pg = is_pg()
    logger.info("DB type: %s", "PostgreSQL" if pg else "SQLite")

    with get_connection() as conn:
        cur = conn.cursor() if pg else conn

        if not _table_exists(cur, "user_mcp_credentials", pg):
            logger.info("Table not found, creating new")
            if not dry_run:
                from crew.mcp_gateway.credentials import _init_credentials_table
                _init_credentials_table()
            return

        has_mcp_server = _column_exists(cur, "user_mcp_credentials", "mcp_server", pg)
        has_tenant_id = _column_exists(cur, "user_mcp_credentials", "tenant_id", pg)

        if has_mcp_server and has_tenant_id:
            logger.info("Table already up to date")
            return

        has_server_name = _column_exists(cur, "user_mcp_credentials", "server_name", pg)

        if has_server_name and not has_mcp_server:
            logger.info("Old schema detected, upgrading...")
            if dry_run:
                logger.info("[DRY RUN] Would rename columns and add new ones")
                return

            if pg:
                for sql in [
                    "ALTER TABLE user_mcp_credentials RENAME COLUMN server_name TO mcp_server",
                    "ALTER TABLE user_mcp_credentials RENAME COLUMN encrypted_value TO access_token",
                    "ALTER TABLE user_mcp_credentials ADD COLUMN IF NOT EXISTS refresh_token TEXT NOT NULL DEFAULT ''",
                    "ALTER TABLE user_mcp_credentials ADD COLUMN IF NOT EXISTS token_expires_at TIMESTAMP",
                    "ALTER TABLE user_mcp_credentials ADD COLUMN IF NOT EXISTS scopes TEXT NOT NULL DEFAULT ''",
                    "ALTER TABLE user_mcp_credentials ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''",
                ]:
                    logger.info("Running: %s", sql)
                    cur.execute(sql)
            else:
                # SQLite: rebuild table
                cur.execute("SELECT user_id, server_name, encrypted_value, created_at, updated_at FROM user_mcp_credentials")
                rows = cur.fetchall()
                cur.execute("DROP TABLE user_mcp_credentials")
                from crew.mcp_gateway.credentials import _init_credentials_table
                _init_credentials_table()
                for row in rows:
                    cur.execute(
                        "INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (row[0], row[1], row[2], row[3], row[4]),
                    )
                conn.commit()
            logger.info("Schema upgrade complete")
            return

        # Add missing columns
        if has_mcp_server and not has_tenant_id:
            if not dry_run:
                sql = "ALTER TABLE user_mcp_credentials ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''"
                if pg:
                    sql = "ALTER TABLE user_mcp_credentials ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''"
                logger.info("Running: %s", sql)
                cur.execute(sql)
                if not pg:
                    conn.commit()
            logger.info("Added tenant_id column")


def migrate_audit_table(dry_run=False):
    from crew.database import get_connection, is_pg
    pg = is_pg()

    with get_connection() as conn:
        cur = conn.cursor() if pg else conn
        if not _table_exists(cur, "mcp_audit_log", pg):
            return
        if _column_exists(cur, "mcp_audit_log", "tenant_id", pg):
            logger.info("mcp_audit_log already has tenant_id")
            return
        if dry_run:
            logger.info("[DRY RUN] Would add tenant_id to mcp_audit_log")
            return
        sql = "ALTER TABLE mcp_audit_log ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''"
        if pg:
            sql = "ALTER TABLE mcp_audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''"
        logger.info("Running: %s", sql)
        cur.execute(sql)
        if not pg:
            conn.commit()
        logger.info("mcp_audit_log tenant_id column added")


def main():
    parser = argparse.ArgumentParser(description="MCP credentials migration")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        migrate(dry_run=args.dry_run)
        migrate_audit_table(dry_run=args.dry_run)
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
