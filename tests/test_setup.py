from conftest import ROOT


def test_project_structure():

    required = [
        "datasets",
        "scripts",
        "morning_drill",
        "morning_drill/agents",
        "notebooks",
        "tests",
        "progress.json",
        "PROJECT.md",
        "ARCHITECTURE.md",
        ".gitignore",
        ".env.example",
        "requirements.txt",
    ]

    for name in required:
        assert (ROOT / name).exists(), f"Missing: {name}"


def test_secrets_are_not_committed():
    assert ".env" in (ROOT / ".gitignore").read_text()


def test_env_example_carries_no_real_key():
    """
    .env.example is committed, so it must only ever hold empty placeholders.
    """
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "API_KEY" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            assert value.strip() == "", f"{key} has a value in .env.example"


def test_tests_do_not_write_to_the_real_record_files():
    """
    A Session built without explicit paths reads and writes the project's
    progress.json and drill_records.json. Every test must redirect all three
    paths, or the suite both pollutes the records and makes its own
    assertions depend on them.
    """
    source = (ROOT / "tests" / "test_session.py").read_text()

    for construction in source.split("Session(")[1:]:
        head = construction[: construction.index(")")]
        for required in ("progress_path", "log_path", "records_path"):
            assert required in head, f"Session(...) missing {required}"
