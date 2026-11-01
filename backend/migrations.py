"""
Database Migration Management Module for DocuMind AI (Alembic Integration).

Provides programmatic helpers for executing, verifying, and testing Alembic migrations
across development, test, and production environments.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

logger = logging.getLogger("documind.migrations")

# Locate alembic.ini relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI_PATH = PROJECT_ROOT / "alembic.ini"


def get_alembic_config(db_url: Optional[str] = None, connection=None) -> Config:
    """
    Constructs an Alembic Config instance loaded from alembic.ini.
    Dynamically overrides sqlalchemy.url or passes an existing active connection.
    Never logs or exposes credentials.
    """
    ini_path = str(ALEMBIC_INI_PATH) if ALEMBIC_INI_PATH.exists() else "alembic.ini"
    cfg = Config(ini_path)

    # Set script location relative to project root
    script_loc = PROJECT_ROOT / "backend" / "alembic"
    cfg.set_main_option("script_location", str(script_loc))

    if connection is not None:
        cfg.attributes["connection"] = connection

    if db_url:
        cfg.attributes["db_url"] = db_url

    return cfg


def get_current_revision(engine: Engine) -> Optional[str]:
    """
    Returns the current Alembic revision hash stored in the target database,
    or None if the database is unversioned / unmigrated.
    """
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    except Exception as err:
        logger.debug("Failed to read alembic_version (table may not exist): %s", err)
        return None


def get_head_revision() -> Optional[str]:
    """
    Returns the latest migration revision defined in the versions directory.
    """
    cfg = get_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head()


def check_migration_status(engine: Engine) -> Dict[str, Any]:
    """
    Inspects migration state of the target database against Alembic definitions.
    """
    current_rev = get_current_revision(engine)
    head_rev = get_head_revision()
    is_up_to_date = (current_rev == head_rev) and (head_rev is not None)

    return {
        "current_revision": current_rev,
        "head_revision": head_rev,
        "is_up_to_date": is_up_to_date,
        "is_baselined": current_rev is not None,
    }


def stamp_database(db_url: Optional[str] = None, connection=None, revision: str = "head") -> None:
    """
    Marks the target database at the given revision without running DDL statements.
    Used exclusively for baselining existing databases with matching schemas.
    """
    cfg = get_alembic_config(db_url=db_url, connection=connection)
    command.stamp(cfg, revision)
    logger.info("Database successfully stamped at revision: %s", revision)


def upgrade_database(db_url: Optional[str] = None, connection=None, revision: str = "head") -> None:
    """
    Applies pending Alembic migrations forward to the target revision (default: 'head').
    Safe and idempotent.
    """
    cfg = get_alembic_config(db_url=db_url, connection=connection)
    command.upgrade(cfg, revision)
    logger.info("Database successfully upgraded to revision: %s", revision)


def downgrade_database(db_url: Optional[str] = None, connection=None, revision: str = "-1") -> None:
    """
    Rolls back migrations to the target revision.
    """
    cfg = get_alembic_config(db_url=db_url, connection=connection)
    command.downgrade(cfg, revision)
    logger.info("Database successfully downgraded to revision: %s", revision)


def ensure_database_migrated(engine: Engine) -> Dict[str, Any]:
    """
    Idempotent migration management for application bootstrapping.

    Behavior:
    1. If target database has existing tables ('users') but no alembic_version,
       safely baselines the database by stamping it to '0001_baseline_schema'.
    2. If target database is completely fresh (no tables),
       executes 'upgrade head' to construct schema cleanly.
    3. If target database already has alembic_version,
       runs 'upgrade head' to apply any pending incremental revisions.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    current_rev = get_current_revision(engine)
    head_rev = get_head_revision()

    if current_rev is None:
        if "users" in existing_tables:
            logger.info("Existing schema detected without Alembic tracking. Baselining to head...")
            stamp_database(connection=None, db_url=str(engine.url), revision="head")
        else:
            logger.info("Fresh database detected. Applying baseline migrations...")
            upgrade_database(connection=None, db_url=str(engine.url), revision="head")
    else:
        if current_rev != head_rev:
            logger.info("Pending migrations detected (%s -> %s). Upgrading...", current_rev, head_rev)
            upgrade_database(connection=None, db_url=str(engine.url), revision="head")
        else:
            logger.debug("Database schema is up to date at revision: %s", current_rev)

    return check_migration_status(engine)
