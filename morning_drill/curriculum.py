"""
The curriculum from PROJECT.md, as data.

The progress agent reads this to know what exists and what order it comes in.
"""

CURRICULUM = {
    "python": {
        "dataframe_filtering": 1,
        "groupby": 2,
        "merge_join": 3,
        "pivot_reshape": 4,
        "missing_duplicates": 5,
        "string_ops": 6,
        "datetime_ops": 7,
        "apply_map": 8,
        "advanced_indexing": 9,
        "vectorization": 10,
    },
    "sql": {
        "select_where": 1,
        "joins": 2,
        "group_by": 3,
        "subqueries": 4,
        "ctes": 5,
        "window_functions": 6,
        "case": 7,
        "set_operations": 8,
        "indexes": 9,
        "analytical_queries": 10,
    },
    "cli": {
        "file_operations": 1,
        "text_processing": 2,
        "process_management": 3,
        "pipes_redirection": 4,
        "find": 5,
        "permissions": 6,
        "environment_variables": 7,
        "bash_scripting": 8,
        "network_utilities": 9,
        "archives": 10,
    },
}

# Only these topics have a spec vocabulary behind them so far. The trainer
# agent is confined to them; the rest of the curriculum needs new generators.
IMPLEMENTED = {
    "python": [
        "dataframe_filtering",
        "groupby",
        # cleaning topics: the student is handed the raw table (morning_drill.mess)
        "missing_duplicates",
        "string_ops",
    ]
}


def curriculum_text():
    lines = []

    for domain, topics in CURRICULUM.items():
        ordered = sorted(topics, key=topics.get)
        implemented = IMPLEMENTED.get(domain, [])

        lines.append(f"{domain}:")

        for topic in ordered:
            mark = "" if topic in implemented else "   (no generator yet)"
            lines.append(f"  {topics[topic]:>2}. {topic}{mark}")

    return "\n".join(lines)


def next_topic(progress, history=None):
    """
    The active topic, by curriculum order, among those we can generate for.
    """
    for domain, topics in IMPLEMENTED.items():
        for topic in topics:
            state = progress.get(domain, {}).get(topic, {})

            if state.get("status") != "mastered":
                return topic, state.get("level", 1)

    return "groupby", 4
