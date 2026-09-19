from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOTS = {
    "shenzhen": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html",
    "montgomery": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html",
}
ROW = re.compile(r"<a href='([^']+)'[^>]*>.*?</a></td><td class='size'>([^<]*)</td>")


def inventory(index_url: str) -> list[tuple[str, int]]:
    queue = [index_url]
    seen = set()
    files = []
    while queue:
        index = queue.pop()
        if index in seen:
            continue
        seen.add(index)
        with urllib.request.urlopen(index, timeout=60) as response:
            content = response.read().decode("utf-8")
        for href, size_text in ROW.findall(content):
            url = urllib.parse.urljoin(index, href)
            if href.endswith("index.html"):
                queue.append(url)
            elif size_text.strip().isdigit():
                files.append((url, int(size_text)))
    return files


def download_one(item: tuple[str, int], root_url: str, destination: Path) -> str:
    url, expected_size = item
    root_dir = root_url.rsplit("/", 1)[0] + "/"
    relative = urllib.parse.unquote(url.removeprefix(root_dir))
    path = destination / Path(relative)
    if path.is_file() and path.stat().st_size == expected_size:
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise OSError(f"Size mismatch for {url}: {partial.stat().st_size} != {expected_size}")
    partial.replace(path)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire public NLM TB CXR releases")
    parser.add_argument("dataset", choices=ROOTS)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    files = inventory(ROOTS[args.dataset])
    total = sum(size for _, size in files)
    print(f"Verified inventory: {len(files)} files, {total / 2**30:.3f} GiB")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(
            executor.map(
                lambda item: download_one(item, ROOTS[args.dataset], args.destination), files
            )
        )
    print(f"Downloaded: {outcomes.count('downloaded')}; resumed: {outcomes.count('skipped')}")
    inventory_text = "\n".join(f"{url}\t{size}" for url, size in sorted(files))
    provenance = {
        "source_index_url": ROOTS[args.dataset],
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "indexed_file_count": len(files),
        "indexed_bytes": total,
        "inventory_sha256": hashlib.sha256(inventory_text.encode("utf-8")).hexdigest(),
        "downloaded_this_run": outcomes.count("downloaded"),
        "already_complete": outcomes.count("skipped"),
        "note": "Individual official files; no source archive was provided, so archive checksum is not applicable.",
    }
    (args.destination / "acquisition_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
