"""
Diagnostic: dump the raw HTML and stripped text from the IMD bulletin
so we can see why the parser fails.
"""
import httpx
import re

URL = "https://mausam.imd.gov.in/Forecast/coastal_bulletin_new.php?id=4"

r = httpx.get(
    URL,
    timeout=25,
    follow_redirects=True,
    headers={
        "User-Agent": (
            "Mozilla/5.0 (compatible; ORCA-Marine-Intelligence/1.0; "
            "academic, non-commercial)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    },
)
r.raise_for_status()

# Save raw HTML
with open("imd_raw.html", "w", encoding="utf-8") as f:
    f.write(r.text)
print(f"Raw HTML saved to imd_raw.html ({len(r.text)} chars)")

# Strip tags (same logic as the service)
text = re.sub(r"<script[^>]*>.*?</script>", " ", r.text, flags=re.DOTALL | re.I)
text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.I)
text = re.sub(r"<[^>]+>", " ", text)
text = text.replace("&nbsp;", " ").replace("&amp;", "&")
text = re.sub(r"\s+", " ", text).strip()

with open("imd_stripped.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Stripped text saved to imd_stripped.txt ({len(text)} chars)")

print("\n" + "=" * 70)
print("First 3000 characters of stripped text:")
print("=" * 70)
print(text[:3000])
print("\n" + "=" * 70)
print("Looking for known field labels:")
print("=" * 70)

for label in [
    "Synoptic", "Wind", "Weather", "Visibility",
    "Sea Condition", "Port Signal", "Storm Surge", "Tidal Wave",
]:
    idx = text.lower().find(label.lower())
    if idx >= 0:
        # Show the 100 chars around each hit
        snippet = text[max(0, idx - 20):idx + 150]
        print(f"\n[{label}] found at char {idx}")
        print(f"  ...{snippet}...")
    else:
        print(f"\n[{label}] NOT FOUND")