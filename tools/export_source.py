"""Deterministically concatenate tracked project text files, excluding the export itself."""

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(output):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = sorted(n for n in names if n and n != "CNGE_SOURCE.txt" and (ROOT / n).is_file())
    chunks = [
        "CNGE — complete source archive (UTF-8)\n",
        "Each FILE section contains the original text of one tracked project file.\n",
    ]
    for name in names:
        data = (ROOT / name).read_bytes()
        text = data.decode("utf-8")
        chunks.append(
            f"\n{'=' * 80}\nFILE: {name}\nSHA256: {hashlib.sha256(data).hexdigest()}\n{'=' * 80}\n{text}"
        )
        if not text.endswith("\n"):
            chunks.append("\n")
    output.write_text("".join(chunks), encoding="utf-8")
    return len(names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "CNGE_SOURCE.txt")
    print(f"Exported {export(parser.parse_args().output)} files")
