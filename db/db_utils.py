import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="rag_logs",
        user="postgres",
        password="madshailu",  # use os.getenv() with .env in real use
        host="localhost",
        port="5432"
    )

def store_query(email, topic, data, query):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_data (email, topic, abstracts, query)
            VALUES (%s, %s, %s, %s)
        """, (email, topic, data, query))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Query stored.")
    except Exception as e:
        print("❌ Error:", e)

# Optional: add fetch_all(), get_by_email(), etc.
