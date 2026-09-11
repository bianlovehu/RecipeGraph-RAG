"""Restore recipe cover images from Git LFS, verifying size and SHA-256."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MEDIA = "https://media.githubusercontent.com/media/FutureUnreal/HowToCook/master/"


def restore(path):
    pointer = path.read_text(encoding="utf-8")
    fields = dict(line.split(" ", 1) for line in pointer.splitlines())
    expected_hash = fields["oid"].removeprefix("sha256:")
    relative = path.relative_to(ROOT / "data").as_posix()
    for attempt in range(3):
        try:
            with urlopen(MEDIA + quote(relative, safe="/"), timeout=30) as response:
                content = response.read(int(fields["size"]) + 1)
            if len(content) != int(fields["size"]) or hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("LFS size or SHA-256 mismatch")
            temporary = path.with_name(path.name + ".download")
            temporary.write_bytes(content)
            temporary.replace(path)
            return True
        except Exception as exc:
            if attempt == 2:
                print(f"FAILED {relative}: {exc}", flush=True)
    return False


def main():
    recipes = json.loads((ROOT / "data/recipes_with_images.json").read_text(encoding="utf-8"))
    paths = set()
    for recipe in recipes:
        path = (ROOT / recipe["file_path"]).parent / recipe.get("image_url", "")
        path = path.resolve()
        if not path.is_relative_to(ROOT / "data"):
            continue
        if path.is_file() and path.stat().st_size < 1024:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                paths.add(path)
    print(f"Restoring {len(paths)} recipe images...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(restore, sorted(paths)))
    print(f"Restored {sum(results)}/{len(paths)} images", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
