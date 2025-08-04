import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

# Set the environment variable for the test database BEFORE any app code is imported.
# This ensures that all modules use the correct test database URL from the start.
TEST_DB_FILE = "./test.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_FILE}"

# Now that the environment is configured, we can safely import app and database code.
from ai4gd_momconnect_haystack.database import Base  # noqa: E402
from ai4gd_momconnect_haystack.api import app  # noqa: E402

# NEW AND CRITICAL: Import the models module. This registers your table definitions
# (UserJourneyState, ChatHistory, etc.) with SQLAlchemy's Base metadata,
# so that Base.metadata.create_all() knows what tables to build.
from ai4gd_momconnect_haystack import sqlalchemy_models  # noqa: F401, E402


@pytest.fixture(scope="session")
def event_loop():
    """Creates an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_database():
    """
    This fixture runs once per test session. It creates an async engine and
    all database tables directly from the SQLAlchemy models. It then tears
    down the database file after the tests are complete.
    """
    # Ensure any old test database from a previous failed run is removed
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)

    # Create an engine that points to our test database file
    engine = create_async_engine(os.environ["DATABASE_URL"])

    # Create all tables in the test database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield  # This is where the tests will run

    # Dispose of the engine and remove the test database file
    await engine.dispose()
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)


@pytest.fixture
def client() -> TestClient:
    """Provides a TestClient instance for making API requests."""
    return TestClient(app)


def pytest_configure(config):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_FILE}"
