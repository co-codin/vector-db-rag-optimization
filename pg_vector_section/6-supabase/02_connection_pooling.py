
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

SUPABASE_POOLER_URL = os.getenv("SUPABASE_POOLER_URL")

class DatabasePool:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, min_conn: int = 1, max_conn: int = 10):
        if self._pool is None:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn=min_conn,
                max_conn=max_conn,
                dsn=SUPABASE_POOLER_URL or DATABASE_URL,
            )
            print(f"✅ Pool initialized: {min_conn}-{max_conn} connections")


    @contextmanager
    def get_connection(self):
        if self._pool is None:
            self.initialize()
        conn = self._pool.getconn()

        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None
            print("✅ All connections closed")


db_pool = DatabasePool()

def search_with_pooling(query_embedding: list, limit: int = 5):
    with db_pool.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit)
        )
        results = cur.fetchall()
        cur.close()
        return results


def demonstrate_pooling():
    from openai import OpenAI
    client = OpenAI()

    print("\n🔄 Demonstrating connection pooling...")

    db_pool.initialize(min_conn=2, max_conn=5)

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test query"
    )

    query_embedding = response.data[0].embedding

    for i in range(5):
        with db_pool.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            cur.close()
            print(f"   Query {i + 1}: {count} documents (connection reused)")

    print("\n✅ All queries used pooled connections!")

def main():
    print("=" * 60)
    print("Connection Pooling for Production")
    print("=" * 60)

    print("\n📝 Why Connection Pooling?")
    print("   - PostgreSQL connections are expensive to create")
    print("   - Each connection uses ~10MB memory")
    print("   - Connection limits exist (Supabase free: ~20)")
    print("   - Pooling reuses connections efficiently")

    print("\n📝 Supabase Pooler:")
    print("   - Direct:  port 5432 (limited connections)")
    print("   - Pooler:  port 6543 (use this in production!)")

    if SUPABASE_POOLER_URL:
        print(f"\n✅ Using Supabase pooler")
    else:
        print("\n⚠️  Using local database (set SUPABASE_POOLER_URL for production)")

    try:
        demonstrate_pooling()
    except Exception as e:
        print(f"\n⚠️  Could not demonstrate (need documents table): {e}")
        print("   Run lecture-03 code first to create sample data")

    # Cleanup
    db_pool.close_all()


if __name__ == "__main__":
    main()