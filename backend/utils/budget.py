import sqlite3
from contextlib import closing

from utils.config import settings


class BudgetExceeded(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.ledger_path, timeout=30, isolation_level=None)
    connection.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK (id = 1), spent REAL NOT NULL)")
    connection.execute("INSERT OR IGNORE INTO budget (id, spent) VALUES (1, 0)")
    return connection


def spent() -> float:
    with closing(_connect()) as connection:
        return connection.execute("SELECT spent FROM budget WHERE id = 1").fetchone()[0]


def reserve(amount: float) -> None:
    """Atomically set aside a call's worst-case cost, or refuse if it could pass the cap."""
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE budget SET spent = spent + ? WHERE id = 1 AND spent + ? <= ?",
            (amount, amount, settings.LLM_BUDGET_USD),
        )
        if cursor.rowcount == 0:
            raise BudgetExceeded(
                f"The ${settings.LLM_BUDGET_USD:.2f} AI spending limit has been reached, "
                "so no more AI requests will be sent."
            )


def settle(reserved: float, actual: float) -> None:
    with closing(_connect()) as connection:
        connection.execute("UPDATE budget SET spent = MAX(0, spent + ? - ?) WHERE id = 1", (actual, reserved))
