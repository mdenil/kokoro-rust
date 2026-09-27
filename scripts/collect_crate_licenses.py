"""Copy the license files of every Rust crate used to build the binary into a release package.

Reads `cargo metadata --format-version 1 --filter-platform <target>` JSON on stdin and writes, under
the directory given as the only argument:
  crates/<name>-<version>/<license files of that crate>
  CRATES.tsv  (name, version, license expression, source, copied files)
Only normal and build dependencies reachable from the workspace package count (not dev-dependencies).
"""
import json
import pathlib
import re
import shutil
import sys

LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|UNLICENSE)", re.IGNORECASE)


def shipped_packages(meta: dict) -> list[dict]:
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen: set[str] = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(k["kind"] in (None, "build") for k in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(root)
    return sorted((packages[p] for p in seen), key=lambda p: (p["name"], p["version"]))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: collect_crate_licenses.py OUT_DIR < cargo-metadata.json")
    out = pathlib.Path(sys.argv[1])
    meta = json.load(sys.stdin)
    rows = ["name\tversion\tlicense\tsource\tfiles"]
    missing = []
    for pkg in shipped_packages(meta):
        src = pathlib.Path(pkg["manifest_path"]).parent
        files = sorted(f for f in src.iterdir() if f.is_file() and LICENSE_FILE.match(f.name))
        dest = out / "crates" / f"{pkg['name']}-{pkg['version']}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(f, dest / f.name)
        if not files:
            missing.append(pkg["name"])
        rows.append("\t".join([pkg["name"], pkg["version"], pkg.get("license") or "", pkg.get("source") or "",
                               ",".join(f.name for f in files) or "(no license file in the published crate)"]))
    (out / "CRATES.tsv").write_text("\n".join(rows) + "\n")
    print(f"collected licenses of {len(rows) - 1} crates into {out}", file=sys.stderr)
    if missing:
        print(f"crates without a license file in their package: {', '.join(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
