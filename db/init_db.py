import duckdb
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'analytical_db.duckdb')
LOG_DB_PATH = os.path.join(os.path.dirname(__file__), 'query_logs.duckdb')

def initialize_analytical_db():
    """Initializes the analytical DuckDB database with some sample data."""
    conn = duckdb.connect(database=DB_PATH, read_only=False)
    conn.execute("SET GLOBAL pandas_analyze_sample=100000;")
    conn.execute("DROP TABLE IF EXISTS sales;")
    conn.execute("DROP TABLE IF EXISTS products;")
    conn.execute("DROP TABLE IF EXISTS customers;")

    # sample tables
    conn.execute("""
        CREATE TABLE products (
            product_id INTEGER,
            product_name VARCHAR,
            category VARCHAR,
            price DECIMAL(10, 2)
        );
    """)

    conn.execute("""
        CREATE TABLE customers (
            customer_id INTEGER,
            customer_name VARCHAR,
            city VARCHAR,
            country VARCHAR
        );
    """)

    conn.execute("""
        CREATE TABLE sales (
            sale_id INTEGER,
            product_id INTEGER,
            customer_id INTEGER,
            sale_date DATE,
            quantity INTEGER,
            revenue DECIMAL(10, 2)
        );
    """)

    # sample data (few rows for testing)
    conn.execute("INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 1200.00);")
    conn.execute("INSERT INTO products VALUES (2, 'Mouse', 'Electronics', 25.00);")
    conn.execute("INSERT INTO products VALUES (3, 'Keyboard', 'Electronics', 75.00);")
    conn.execute("INSERT INTO products VALUES (4, 'Monitor', 'Electronics', 300.00);")
    conn.execute("INSERT INTO products VALUES (5, 'Desk', 'Furniture', 150.00);")


    conn.execute("INSERT INTO customers VALUES (101, 'Alice Smith', 'New York', 'USA');")
    conn.execute("INSERT INTO customers VALUES (102, 'Bob Johnson', 'London', 'UK');")
    conn.execute("INSERT INTO customers VALUES (103, 'Charlie Brown', 'Paris', 'France');")

    conn.execute("INSERT INTO sales VALUES (1, 1, 101, '2024-01-05', 1, 1200.00);")
    conn.execute("INSERT INTO sales VALUES (2, 2, 101, '2024-01-05', 2, 50.00);")
    conn.execute("INSERT INTO sales VALUES (3, 3, 102, '2024-01-10', 1, 75.00);")
    conn.execute("INSERT INTO sales VALUES (4, 1, 103, '2024-01-15', 1, 1200.00);")
    conn.execute("INSERT INTO sales VALUES (5, 5, 102, '2024-01-20', 1, 150.00);")
    conn.execute("INSERT INTO sales VALUES (6, 4, 101, '2024-01-22', 1, 300.00);")
    conn.execute("INSERT INTO sales VALUES (7, 1, 103, '2024-01-25', 2, 2400.00);")


    conn.close()
    print(f"Analytical database initialized at {DB_PATH}")

def initialize_query_logs_db():
    """Initializes the query logs DuckDB database."""
    conn = duckdb.connect(database=LOG_DB_PATH, read_only=False)

    conn.execute("DROP TABLE IF EXISTS query_logs;")
    conn.execute("""
        CREATE TABLE query_logs (
            log_id UUID PRIMARY KEY DEFAULT uuid(),
            query_text VARCHAR,
            execution_time_ms DOUBLE, -- In milliseconds for better precision
            result_rows INTEGER,
            query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Features extracted from query, to be added later
            has_select_all BOOLEAN,
            has_where BOOLEAN,
            has_join BOOLEAN,
            has_groupby BOOLEAN,
            query_length INTEGER,
            num_tables INTEGER,
            estimated_cost DOUBLE -- Placeholder for ML prediction
            -- Add more features as needed based on your sqlglot parsing
        );
    """)
    conn.close()
    print(f"Query logs database initialized at {LOG_DB_PATH}")

if __name__ == "__main__":
    initialize_analytical_db()
    initialize_query_logs_db()

    conn_analytic = duckdb.connect(database=DB_PATH, read_only=True)
    print("\nTables in analytical_db:", conn_analytic.execute("SHOW TABLES;").fetchall())
    conn_analytic.close()

    conn_logs = duckdb.connect(database=LOG_DB_PATH, read_only=True)
    print("Tables in query_logs_db:", conn_logs.execute("SHOW TABLES;").fetchall())
    conn_logs.close()