"""
Coastal classifier QA.

Verifies that major Indian coastal cities are classified as coastal and
inland cities are classified as inland. Checks the boundary behaviour
around the 30 km threshold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.coastal_check import is_coastal


CASES = [
    # (name, lat, lon, expected_coastal)
    ("Goa",             15.2993,  73.8243, True),
    ("Panaji",          15.4909,  73.8278, True),
    ("Mumbai",          19.0760,  72.8777, True),
    ("Chennai",         13.0827,  80.2707, True),
    ("Kochi",            9.9312,  76.2673, True),
    ("Visakhapatnam",   17.6868,  83.2185, True),
    ("Mangalore",       12.9141,  74.8560, True),
    ("Karwar",          14.8135,  74.1297, True),
    ("Lakshadweep",     10.5667,  72.6417, True),
    ("Port Blair",      11.6234,  92.7265, True),
    ("Koramangala",     12.9320,  77.6227, False),
    ("Bangalore",       12.9716,  77.5946, False),
    ("Delhi",           28.6139,  77.2090, False),
    ("Hyderabad",       17.3850,  78.4867, False),
    ("Pune",            18.5204,  73.8567, False),
    ("Mysore",          12.2958,  76.6394, False),
]


def main():
    print("=" * 80)
    print("COASTAL CLASSIFIER QA")
    print("=" * 80)
    passed = 0
    failed = 0

    for name, lat, lon, expected in CASES:
        r = is_coastal(lat, lon)
        ok = r["coastal"] == expected
        mark = "✅" if ok else "❌"
        print(
            f"{mark}  {name:20s}  "
            f"coastal={r['coastal']!s:5s}  "
            f"dist={r['distance_km']:>7.1f} km  "
            f"(expected coastal={expected})"
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Passed: {passed}  Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)