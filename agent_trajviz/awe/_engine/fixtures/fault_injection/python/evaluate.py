"""Fixed evaluator: failed assertions are grades; crashes are ungraded."""

import json


def main() -> None:
    # This fixture is also a packaged Python module. Only import the toy
    # workspace's module when executing its evaluator, never on package import.
    import core

    checks = [
        ("lower_bound", lambda: core.clamp(-2, 0, 10) == 0),
        ("upper_bound", lambda: core.clamp(12, 0, 10) == 10),
        ("inside_interval", lambda: core.clamp(5, 0, 10) == 5),
        ("addition", lambda: core.add(2, 3) == 5),
    ]
    tests = [{"id": name, "passed": check()} for name, check in checks]
    print(json.dumps({"tests": tests, "resolved": all(t["passed"] for t in tests)}))


if __name__ == "__main__":
    main()
