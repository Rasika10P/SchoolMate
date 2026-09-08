"""Read pasted CDE ELD text until EOF, preview, then confirm appending to data/eld.csv."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
import sys

if __package__:
    from .paste_to_csv import confirm, web_line
else:
    from paste_to_csv import confirm, web_line

ROOT = Path(__file__).resolve().parents[1]
CODE = re.compile(r"ELD\.(PII?)\.(K|\d+)\.(\d+[a-z]?)\.(Br|Em|Ex)")
LEVELS = {"Br": "Bridging", "Em": "Emerging", "Ex": "Expanding"}
LABELS = ("Grade", "Critical Principle", "Cluster", "Proficiency Level", "Content Strand", "Standard")


@dataclass(frozen=True)
class EldRow:
    framework_id: str
    code: str
    grade: int
    critical_principle: str
    cluster: str
    proficiency_level: str
    content_strand: str
    text: str
    source_url: str
    page: int | None


HEADER = [field.name for field in fields(EldRow)]


def parse_text(text: str, *, framework: str, source_url: str, page: int | None = None) -> list[EldRow]:
    """Preserve all supplied fields; reject malformed records before any writing."""
    cleaned = "\n".join(web_line(line) for line in text.splitlines())
    blocks = re.split(r"(?m)^Standard Identifier:\s*", cleaned)
    rows = []
    for block in blocks[1:]:
        identifier, separator, body = block.partition("\n")
        link = re.fullmatch(r"\[([^]]+)\]\((https?://[^)]+)\)", identifier.strip())
        code = link.group(1) if link else identifier.strip()
        match = CODE.fullmatch(code)
        if not match or not separator:
            raise ValueError(f"Invalid ELD identifier or record: {code}")
        # Copied CDE pages sometimes concatenate adjacent metadata labels.
        metadata, standard_label, standard_text = body.partition("Standard:")
        metadata = re.sub(r"(?<!\n)(Cluster:|Content Strand:)", r"\n\1", metadata)
        body = metadata + standard_label + standard_text
        labels = list(re.finditer(r"(?m)^(" + "|".join(LABELS) + r"):[ \t]*", body))
        values = {}
        for i, label in enumerate(labels):
            name = label.group(1)
            if name in values:
                raise ValueError(f"{code}: repeated {name} field")
            end = labels[i + 1].start() if i + 1 < len(labels) else len(body)
            value = body[label.end():end]
            value = "\n".join(line for line in value.splitlines() if line.strip() != "---")
            values[name] = " ".join(value.split())
        missing = [name for name in LABELS if not values.get(name)]
        if missing:
            raise ValueError(f"{code}: missing {', '.join(missing)}")
        if values["Grade"] != match.group(2) or values["Proficiency Level"] != LEVELS[match.group(4)]:
            raise ValueError(f"{code}: grade or proficiency disagrees with identifier")
        rows.append(EldRow(framework, code, 0 if values["Grade"] == "K" else int(values["Grade"]),
                           values["Critical Principle"], values["Cluster"], values["Proficiency Level"],
                           values["Content Strand"], values["Standard"], link.group(2) if link else source_url, page))
    return rows


def pending_rows(path: Path, rows: list[EldRow]) -> tuple[list[dict], int]:
    seen = set()
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != HEADER:
                raise ValueError(f"{path}: expected header {','.join(HEADER)}")
            for record in reader:
                if None in record or any(value is None for value in record.values()):
                    raise ValueError(f"{path}:{reader.line_num}: malformed CSV row")
                seen.add((record["framework_id"], record["code"]))
    pending = []
    for row in rows:
        identity = (row.framework_id, row.code)
        if identity not in seen:
            pending.append(asdict(row))
            seen.add(identity)
    return pending, len(rows) - len(pending)


def preview(rows: list[EldRow], output) -> None:
    print(" | ".join(HEADER), file=output)
    print(" | ".join("---" for _ in HEADER), file=output)
    for row in rows:
        print(" | ".join("" if value is None else str(value).replace("|", "\\|")
                         for value in asdict(row).values()), file=output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--page", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.page is not None and args.page < 1:
        parser.error("--page must be positive")
    print("Paste ELD text, then press Ctrl-D on an empty line to preview.")
    try:
        rows = parse_text(sys.stdin.read(), framework=args.framework, source_url=args.source_url, page=args.page)
        path = ROOT / "data" / "eld.csv"
        pending, skipped = pending_rows(path, rows)
        preview(rows, sys.stdout)
        placeholders = sum(row.text.startswith("No standard") for row in rows)
        print(f"Ready to append: {len(pending)}; duplicates skipped: {skipped}; 'No standard' entries retained: {placeholders}.")
        appended = 0
        if pending and not args.dry_run and confirm():
            pending, skipped = pending_rows(path, rows)
            if pending:
                path.parent.mkdir(parents=True, exist_ok=True)
                empty = not path.exists() or path.stat().st_size == 0
                needs_newline = False
                if not empty:
                    with path.open("rb") as existing:
                        existing.seek(-1, 2)
                        needs_newline = existing.read(1) not in (b"\n", b"\r")
                with path.open("a", newline="", encoding="utf-8") as handle:
                    if needs_newline:
                        handle.write("\n")
                    writer = csv.DictWriter(handle, fieldnames=HEADER)
                    if empty:
                        writer.writeheader()
                    writer.writerows(pending)
                appended = len(pending)
        print(f"Rows parsed: {len(rows)}; rows appended: {appended}; duplicates skipped: {skipped}.")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
