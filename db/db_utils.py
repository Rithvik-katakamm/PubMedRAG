import os, json, logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import SimpleConnectionPool

load_dotenv("env_variables.env")
logger = logging.getLogger(__name__)

_POOL: SimpleConnectionPool | None = None

def _get_pool() -> SimpleConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.environ["DB_HOST"],
            port=os.environ.get("DB_PORT", 5432),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
        )
    return _POOL

def log_conversation(email: str, topic: str, convo_dict: dict) -> None:
    """Upsert a whole conversation (JSON) into Postgres."""
    sql = """
    INSERT INTO conversations (convo_id, email, topic, conversation)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (convo_id)
      DO UPDATE SET conversation = EXCLUDED.conversation;
    """
    # convo_id sits inside convo_dict["id"]
    convo_id = convo_dict["id"]
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (convo_id, email, topic, json.dumps(convo_dict)))
    except Exception as e:
        logger.exception("DB write failed")
        raise
    finally:
        pool.putconn(conn)
