from sqlglot import parse_one, exp
from sqlglot.errors import ParseError
import re


def extract_query_features(query_text: str) -> dict:
    features = {
        "query_text": query_text,
        "query_length": len(query_text),
        "num_lines": query_text.count("\n") + 1,
        "has_select_all": False,
        "has_where": False,
        "has_join": False,
        "has_groupby": False,
        "has_orderby": False,
        "has_limit": False,
        "has_distinct": False,
        "has_union": False,
        "has_intersect": False,
        "has_except": False,
        "has_cte": False,
        "has_subquery": False,
        "is_dml": False,
        "is_ddl": False,
        "num_tables": 0,
        "num_columns_selected": 0,
        "num_joins": 0,
        "num_predicates": 0,
        "num_aggregations": 0,
        "num_unions": 0,
        "join_types": {},
    }

    try:
        query_upper = query_text.upper().strip()

        if re.search(r"\bSELECT\s+\*", query_upper):
            features["has_select_all"] = True

        if "WHERE" in query_upper:
            features["has_where"] = True

        if any(
            join_type in query_upper
            for join_type in [
                "JOIN",
                "INNER JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "FULL JOIN",
            ]
        ):
            features["has_join"] = True
            features["num_joins"] = len(re.findall(r"\bJOIN\b", query_upper))

        if "GROUP BY" in query_upper:
            features["has_groupby"] = True

        if "ORDER BY" in query_upper:
            features["has_orderby"] = True

        if "LIMIT" in query_upper:
            features["has_limit"] = True

        if "DISTINCT" in query_upper:
            features["has_distinct"] = True

        if "UNION" in query_upper:
            features["has_union"] = True
            features["num_unions"] = len(re.findall(r"\bUNION\b", query_upper))

        if "INTERSECT" in query_upper:
            features["has_intersect"] = True

        if "EXCEPT" in query_upper:
            features["has_except"] = True

        if "WITH" in query_upper:
            features["has_cte"] = True

        if query_upper.count("SELECT") > 1:
            features["has_subquery"] = True

        if any(dml in query_upper for dml in ["INSERT", "UPDATE", "DELETE"]):
            features["is_dml"] = True

        if any(ddl in query_upper for ddl in ["CREATE", "DROP", "ALTER"]):
            features["is_ddl"] = True

        table_matches = re.findall(r"\bFROM\s+(\w+)", query_upper)
        join_matches = re.findall(r"\bJOIN\s+(\w+)", query_upper)
        features["num_tables"] = len(set(table_matches + join_matches))

        and_or_count = len(re.findall(r"\b(AND|OR)\b", query_upper))
        comparison_count = len(re.findall(r"[<>=!]+", query_text))
        features["num_predicates"] = max(
            and_or_count + 1 if features["has_where"] else 0, comparison_count
        )

        agg_functions = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
        agg_count = 0
        for func in agg_functions:
            agg_count += len(re.findall(rf"\b{func}\s*\(", query_upper))
        features["num_aggregations"] = agg_count

        try:
            parsed_expression = parse_one(query_text, read="duckdb")
            if not parsed_expression:
                parsed_expression = parse_one(query_text, read="sql")

            if parsed_expression:
                if isinstance(parsed_expression, (exp.Insert, exp.Update, exp.Delete)):
                    features["is_dml"] = True
                elif isinstance(parsed_expression, (exp.Create, exp.Drop, exp.Alter)):
                    features["is_ddl"] = True

                if parsed_expression.find_all(exp.Star):
                    features["has_select_all"] = True

                joins = list(parsed_expression.find_all(exp.Join))
                if joins:
                    features["has_join"] = True
                    features["num_joins"] = len(joins)
                    for join in joins:
                        join_type = join.args.get(
                            "kind", exp.Join.Kind.INNER
                        ).name.upper()
                        features["join_types"][join_type] = (
                            features["join_types"].get(join_type, 0) + 1
                        )

                tables = set()
                for table_exp in parsed_expression.find_all(exp.Table):
                    tables.add(table_exp.name)
                if tables:
                    features["num_tables"] = len(tables)

                if (
                    isinstance(parsed_expression, exp.Select)
                    and not features["has_select_all"]
                ):
                    columns = list(parsed_expression.find_all(exp.Column))
                    features["num_columns_selected"] = len(columns)

                predicates = []
                for clause in parsed_expression.find_all(
                    exp.Where, exp.Join, exp.Having
                ):
                    condition = (
                        clause.on if isinstance(clause, exp.Join) else clause.this
                    )
                    if condition and isinstance(condition, exp.Expression):
                        predicates.extend(
                            list(condition.find_all(exp.Binary, exp.Condition))
                        )

                if predicates:
                    features["num_predicates"] = max(
                        features["num_predicates"], len(set(predicates))
                    )

        except (ValueError, AttributeError, ParseError):
            pass

    except Exception as e:
        print(f"Warning: Error extracting features from query: {e}")

    return features
