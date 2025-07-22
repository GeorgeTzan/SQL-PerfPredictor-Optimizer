from sqlglot import exp, parse_one


def extract_query_features(query_text: str) -> dict:
    """
    Extracts basic features from a SQL query using sqlglot.
    This is a preliminary version and can be significantly expanded.
    """
    features = {
        "query_text": query_text,
        "query_length": len(query_text),
        "has_select_all": False,
        "has_where": False,
        "has_join": False,
        "has_groupby": False,
        "num_tables": 0,
    }

    try:
        parsed_expression = parse_one(query_text, read="duckdb")

        if parsed_expression.find(exp.Star):
            features["has_select_all"] = True

        if parsed_expression.find(exp.Where):
            features["has_where"] = True

        if parsed_expression.find(exp.Join):
            features["has_join"] = True

        if parsed_expression.find(exp.Group):
            features["has_groupby"] = True

        tables = set()
        for table_exp in parsed_expression.find_all(exp.Table):
            tables.add(table_exp.name)
        features["num_tables"] = len(tables)

    except Exception as e:
        print(f"Warning: Could not parse query '{query_text[:50]}...'. Error: {e}")
        pass

    return features


if __name__ == "__main__":
    # Test cases for the parser
    print("Testing SQL Parser:")
    test_queries = [
        "SELECT * FROM sales WHERE quantity > 10;",
        "SELECT product_id, SUM(revenue) FROM sales GROUP BY product_id;",
        "SELECT c.customer_name, p.product_name FROM customers c JOIN sales s ON c.customer_id = s.customer_id JOIN products p ON s.product_id = p.product_id WHERE s.revenue > 100;",
        "SELECT count(*) FROM products;",
        "SELECT a + b FROM some_table;",
        "INVALID SQL SYNTAX",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        feats = extract_query_features(query)
        for k, v in feats.items():
            if k != "query_text":
                print(f"  {k}: {v}")
