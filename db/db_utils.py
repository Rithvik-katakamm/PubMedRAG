# in db/db_utils.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv("env_variables.env")

def get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", 5432),
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )

def store_interaction(
    email: str,
    topic: str,
    query: str,
    retrieved_chunks: str,
    answer: str
):
    """
    Insert a new user_data row or update the retrieved_chunks+answer
    if that (email, topic, query) already exists.
    """
    sql = """
    INSERT INTO user_data (
        email, topic, query, retrieved_chunks, answer
    )
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (email, topic, query)
    DO UPDATE SET
      retrieved_chunks = EXCLUDED.retrieved_chunks,
      answer = EXCLUDED.answer;
    """
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (email, topic, query, retrieved_chunks, answer))
    conn.close()
