#!/usr/bin/env python3
"""Pre-push content audit: fail if any tracked or staged file contains private-fixture content.

Ignore rules only protect paths; this checks CONTENT. For every private fixture text file under
$KOKORO_DATA/bench/private/**.txt (never printed), it builds fingerprints of every line of >= 24
characters plus sliding 48-character windows, then scans all tracked + staged repository files
(working-tree and index versions). Also refuses tracked audio / binary-array artifacts outright.
Exit 0 = clean; exit 1 = possible leak (reports repo path and fixture line NUMBER only).
Usage: python3 scripts/check_private_leaks.py [--repo DIR]
"""
import hashlib
import os
import pathlib
import subprocess
import sys

WINDOW = 48
MIN_LINE = 24
FORBIDDEN_EXT = {".wav", ".flac", ".mp3", ".m4a", ".m4b", ".aac", ".opus", ".ogg", ".npy", ".npz", ".safetensors", ".pt", ".pth"}


def norm(s):
    return " ".join(s.split())


def main():
    repo = pathlib.Path(sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else pathlib.Path(__file__).resolve().parent.parent)
    if not os.environ.get("KOKORO_DATA"):
        sys.exit("KOKORO_DATA is not set: the private corpus to audit against lives under it (refusing to audit nothing)")
    data = pathlib.Path(os.environ["KOKORO_DATA"])
    private = sorted((data / "bench/private").glob("**/*.txt"))
    if not private:
        sys.exit(f"no private corpus files under {data / 'bench/private'}: refusing to report clean against nothing")
    lines, windows = {}, {}
    for f in private:
        for no, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            t = norm(line)
            if len(t) >= MIN_LINE:
                lines[hashlib.sha256(t.encode()).digest()] = (f.name, no)
            for i in range(0, max(0, len(t) - WINDOW + 1), 8):
                windows[t[i:i + WINDOW]] = (f.name, no)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True).stdout

    paths = set(git("ls-files", "-z").decode().split("\0")) | set(git("diff", "--cached", "--name-only", "-z").decode().split("\0"))
    paths.discard("")
    problems = []
    for p in sorted(paths):
        if pathlib.Path(p).suffix.lower() in FORBIDDEN_EXT:
            problems.append(f"{p}: forbidden artifact type tracked/staged")
            continue
        blobs = []
        wt = repo / p
        if wt.is_file():
            blobs.append(("worktree", wt.read_bytes()))
        try:
            blobs.append(("index", git("show", f":{p}")))
        except subprocess.CalledProcessError:
            pass
        for where, b in blobs:
            text = norm(b.decode("utf-8", errors="replace"))
            hit = None
            for line in b.decode("utf-8", errors="replace").splitlines():
                t = norm(line)
                if len(t) >= MIN_LINE and hashlib.sha256(t.encode()).digest() in lines:
                    hit = lines[hashlib.sha256(t.encode()).digest()]
                    break
            if hit is None and windows:
                for i in range(0, max(0, len(text) - WINDOW + 1)):
                    w = text[i:i + WINDOW]
                    if w in windows:
                        hit = windows[w]
                        break
            if hit:
                problems.append(f"{p} ({where}): matches private fixture {hit[0]} line {hit[1]}")
                break
    print(f"audited {len(paths)} tracked/staged files against {len(private)} private fixture file(s), "
          f"{len(lines)} line fingerprints, {len(windows)} windows")
    if problems:
        print("POSSIBLE PRIVATE CONTENT LEAK — do not push:")
        for x in problems:
            print("  " + x)
        sys.exit(1)
    print("clean")


if __name__ == "__main__":
    main()
