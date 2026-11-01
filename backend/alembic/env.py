import os
import sys
from logging.config import fileConfig
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from alembic import context
from pgvector.sqlalchemy import Vector

# Load environment configuration safely
load_dotenv()

# Prepend project root to sys.path so backend modules are cleanly discoverable
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Target SQLAlchemy metadata
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Safely retrieves the database URL without exposing credentials in stdout/logs.
    Defaults to the application DATABASE_URL or standard connection string.
    """
    # 1. Direct attribute passed programmatically (e.g. test harnesses)
    if "db_url" in config.attributes:
        return config.attributes["db_url"]

    # 2. Main option set in alembic config
    main_url = config.get_main_option("sqlalchemy.url")
    if main_url:
        return main_url

    # 3. Allow explicit override from CLI argument (-x db_url=...)
    x_args = context.get_x_argument(as_dictionary=True)
    if "db_url" in x_args:
        return x_args["db_url"]

    # 4. Read from application environment
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://documind:documind_password@localhost:5432/documind_db",
    )
    return url


def include_object(object_, name, type_, reflected, compare_to):
    """
    Filters out system tables, extensions, and unmanaged objects from autogeneration.
    """
    # Exclude internal PostgreSQL and pgvector system constructs
    if type_ == "table" and name in ("alembic_version", "spatial_ref_sys"):
        return False
    return True


def run_migrations_offline() -> None:
    """
    Runs migrations in 'offline' mode.
    Configures the context with just a URL and not an Engine.
    """
    url = get_database_url()
    dialect_name = "postgresql" if "postgres" in url else "sqlite"
    render_as_batch = (dialect_name == "sqlite")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=render_as_batch,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Runs migrations in 'online' mode.
    Creates an Engine and associates a connection with the context.
    """
    # Check if a connection was directly passed in config (e.g. during test harnesses)
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        db_url = get_database_url()
        configuration = config.get_section(config.config_ini_section, {})
        configuration["sqlalchemy.url"] = db_url

        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        dialect_name = connection.dialect.name
        render_as_batch = (dialect_name == "sqlite")

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=render_as_batch,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
