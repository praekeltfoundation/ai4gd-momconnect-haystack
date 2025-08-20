from alembic import context
from sqlalchemy import create_engine, pool

from ai4gd_momconnect_haystack.database import DATABASE_URL
from ai4gd_momconnect_haystack.sqlalchemy_models import Base

config = context.config
# config.set_main_option("sqlalchemy.url", DATABASE_URL)


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    sync_db_url = DATABASE_URL.replace("sqlite+aiosqlite", "sqlite")
    connectable = create_engine(sync_db_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
