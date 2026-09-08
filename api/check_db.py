"""Check configured Postgres connectivity and TLS: python -m api.check_db."""

from api.db import close_pool, database_url, get_conn


def main() -> None:
    try:
        database_url()  # Report missing TLS explicitly before connecting.
        with get_conn() as conn:
            row = conn.execute("SELECT version()").fetchone()
            if row is None:
                raise RuntimeError("Server version query returned no row")
            print(f"Server version: {row[0]}")
            print(f"TLS active: {'yes' if conn.pgconn.ssl_in_use else 'no'}")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
