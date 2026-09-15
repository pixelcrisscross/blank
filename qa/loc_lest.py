"""
Location resolver QA.

Covers: coastal cities, inland cities, states, ambiguous names,
coordinates, non-Indian names, empty input, malformed input.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.location import resolve_location_query


def _fmt(result):
    if result.get("status") != "FOUND":
        return f"FAIL ({result.get('status')})"
    return (
        f"{result.get('name'):25s} "
        f"({result.get('latitude'):>8.4f}, {result.get('longitude'):>9.4f})  "
        f"[{result.get('source')}]"
    )


CASES = [
    # (query, expected_status, expected_name_contains, description)
    ("Goa",                       "FOUND", "Goa",         "state, not village"),
    ("Panaji",                    "FOUND", "Panaji",      "Goa capital"),
    ("Mumbai",                    "FOUND", "Mumbai",      "west coast"),
    ("Chennai",                   "FOUND", "Chennai",     "east coast"),
    ("Kochi",                     "FOUND", "Kochi",       "Kerala"),
    ("Visakhapatnam",             "FOUND", "Visakhapatnam", "Andhra"),
    ("Lakshadweep",               "FOUND", "Lakshadweep", "UT"),
    ("Andaman",                   "FOUND", None,          "UT"),
    ("Koramangala",               "FOUND", "Koramangala", "Bengaluru area"),
    ("Bangalore",                 "FOUND", "Bangalore",   "inland city"),
    ("off Goa",                   "FOUND", "Goa",         "prefix stripping"),
    ("near Mumbai",               "FOUND", "Mumbai",      "prefix stripping"),
    ("17.69, 83.29",              "FOUND", "coordinates", "explicit coords"),
    ("17.69 83.29",               "FOUND", "coordinates", "coords, space sep"),
    ("xyzabcnotaplace",           "NOT_FOUND", None,      "garbage input"),
    ("",                          "NOT_FOUND", None,      "empty string"),
    ("Sydney",                    "FOUND", None,          "non-Indian"),
    ("Genoa",                     "FOUND", None,          "Italian city"),
]


def main():
    print("=" * 80)
    print("LOCATION RESOLVER QA")
    print("=" * 80)
    passed = 0
    failed = 0

    for query, expected_status, expect_name, desc in CASES:
        r = resolve_location_query(query)
        status = r.get("status")
        status_ok = status == expected_status

        name_ok = True
        if expect_name and status == "FOUND":
            name_ok = expect_name.lower() in str(r.get("name", "")).lower()

        ok = status_ok and name_ok
        mark = "✅" if ok else "❌"
        print(f"{mark}  {desc:35s}  q={query!r:25s}")
        print(f"     -> {_fmt(r)}")
        if not ok:
            print(f"     expected status={expected_status}, "
                  f"name contains {expect_name!r}")
            failed += 1
        else:
            passed += 1

    print()
    print(f"Passed: {passed}  Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)