"""Apply additive database migrations: python -m api.migrate."""

from pathlib import Path

from api.db import get_conn

MIGRATION_DIR = Path(__file__).with_name("migrations")


def migrate(conn) -> None:
    with conn.cursor() as cursor:
        for path in sorted(MIGRATION_DIR.glob("*.sql")):
            cursor.execute(path.read_text(encoding="utf-8"))


def main() -> None:
    with get_conn() as conn:
        migrate(conn)
    print("Database migrations applied.")


if __name__ == "__main__":
    main()
