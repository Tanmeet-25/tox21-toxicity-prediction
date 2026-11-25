"""Assemble the single-file web app from its sources and the exported data payload."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app"

tpl = (APP / "template.html").read_text()
css = (APP / "style.css").read_text()
js = (APP / "app.js").read_text()
data = (ROOT / "src" / "app_data.json").read_text()
meta = json.loads(data)["meta"]

out = (tpl.replace("__CSS__", css)
          .replace("__JS__", js)
          .replace("__DATA__", data)
          .replace("__NCOMP__", f"{meta['compounds']:,}")
          .replace("__NSCAF__", f"{meta['scaffolds']:,}")
          .replace("__MACRO__", str(meta["testMacroRoc"]))
          .replace("__NSHOWN__", str(meta["shown"])))

# the animated counters need raw integers, not comma-formatted strings
out = out.replace(f'data-count="{meta["compounds"]:,}"', f'data-count="{meta["compounds"]}"')
out = out.replace(f'data-count="{meta["scaffolds"]:,}"', f'data-count="{meta["scaffolds"]}"')

for token in ["__CSS__", "__JS__", "__DATA__", "__NCOMP__", "__NSCAF__", "__MACRO__", "__NSHOWN__"]:
    assert token not in out, f"unreplaced placeholder: {token}"

target = ROOT / "index.html"
target.write_text(out)
print(f"built {target} ({target.stat().st_size / 1024:.1f} KB)")
