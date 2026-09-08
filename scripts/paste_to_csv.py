"""Preview pasted PDF or CDE web text and append confirmed CSV rows."""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
STANDARD_HEADER = ["framework_id", "code", "grade", "domain", "cluster", "text", "source_url", "page"]
ALD_HEADER = ["framework_id", "grade", "subject", "level", "text"]
CODE = re.compile(r"^(?P<code>(?P<grade>\d+|K)\.(?P<domain>[A-Z]{2,3})\.(?P<cluster>[A-Z])\.(?P<number>\d+[a-z]?))(?=\s|$)\s*(?P<text>.*)$")
LEVEL = re.compile(r"^Level\s+([1-4])\s*(?::\s*(.*))?$", re.IGNORECASE)
CODE_LIKE = re.compile(r"^(?:\d+|K)\.[A-Z]|^Level\s+\d", re.IGNORECASE)
WEB_IDENTIFIER = re.compile(r"^Standard Identifier:\s*(.*)$")
WEB_CODE = re.compile(r"(?P<grade>K|\d+)\.(?P<domain>[A-Z]{1,3})\.(?:[A-Z]\.)?\d+(?:\.\d+)*(?:\.?[a-z])?")
RUNNING_HEADER = re.compile(
    r"^(?:California Department of Education(?:\s*[|—–-]\s*.*)?|"
    r"California Common Core State Standards(?:\s*:\s*Mathematics)?|"
    r"Mathematics(?:\s*\|\s*(?:Grade\s+\d+|Kindergarten))?|"
    r"Grade\s+\d+|Kindergarten|Page\s+\d+(?:\s+of\s+\d+)?|\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StandardRow:
    framework_id: str
    code: str
    grade: int
    domain: str
    cluster: str
    text: str
    source_url: str
    page: int | None


@dataclass(frozen=True)
class AldRow:
    framework_id: str
    grade: int
    subject: str
    level: int
    text: str


@dataclass
class Parsed:
    rows: list[StandardRow | AldRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unparsed: list[tuple[int, str]] = field(default_factory=list)
    dropped: list[tuple[int, str]] = field(default_factory=list)


def repair_whitespace(lines: list[str]) -> str:
    """Join wrapped lines; remove a line-end hyphen only before a lower-case letter."""
    result = ""
    for line in lines:
        line = " ".join(line.split())
        if not line:
            continue
        if result.endswith(("-", "\u00ad")) and line[0].islower():
            result = result[:-1] + line
        else:
            result += (" " if result else "") + line
    return result


def web_line(line: str) -> str:
    """Remove copied Markdown headings, bold markup, and hard line breaks."""
    line = re.sub(r"^[*•-](?:\s+|$)", "", html.unescape(line).strip())
    return re.sub(r"^#{1,6}\s+", "", line).replace("**", "").rstrip("\\").strip()


def is_web_text(text: str) -> bool:
    return any(WEB_IDENTIFIER.match(web_line(line)) for line in text.splitlines())


def parse_web_text(text: str, *, framework: str, grade: int, source_url: str, page: int | None) -> Parsed:
    """Read labeled CDE records, preserving their identifiers and cluster wording."""
    result = Parsed()
    identifier: re.Match[str] | None = None
    fields: dict[str, list[str]] = {}
    active = ""
    url = source_url
    start = 0

    def finish() -> None:
        if identifier is None:
            return
        values = {name: " ".join(" ".join(parts).split()) for name, parts in fields.items()}
        code = identifier.group()
        missing = [name for name in ("Grade", "Domain", "Cluster", "Standard") if not values.get(name)]
        if missing:
            result.warnings.append(f"Line {start}: {code} missing {', '.join(missing)}; not appendable.")
            result.unparsed.append((start, code))
            return
        actual_grade = 0 if identifier.group("grade") == "K" else int(identifier.group("grade"))
        if values["Grade"] != identifier.group("grade"):
            result.warnings.append(f"{code}: Grade field disagrees with identifier; not appendable.")
            result.unparsed.append((start, code))
            return
        if actual_grade != grade:
            result.warnings.append(f"{code}: code grade {actual_grade} disagrees with --grade {grade}; using {actual_grade}.")
        outcome = values["Standard"]
        if values.get("Footnote"):
            outcome += " Footnote: " + values["Footnote"]
        result.rows.append(StandardRow(framework, code, actual_grade, identifier.group("domain"),
                                      values["Cluster"], outcome, url, page))
        ending = re.sub(r"\s+CA$", "", outcome)
        if not re.search(r'[.!?][\)\]"”’]*$', ending) or ending.endswith(("...", "…")):
            result.warnings.append(f"{code}: possibly truncated text; inspect the original source.")

    for number, original in enumerate(text.splitlines(), 1):
        line = web_line(original)
        heading = WEB_IDENTIFIER.match(line)
        if heading:
            finish()
            fields, active, start = {}, "", number
            value = heading.group(1)
            link = re.fullmatch(r"\[([^]]+)\]\((https?://[^)]+)\)", value)
            identifier = WEB_CODE.fullmatch(link.group(1) if link else value)
            url = link.group(2) if link else source_url
            if identifier is None:
                result.unparsed.append((number, original))
            continue
        if not line:
            continue
        if re.fullmatch(r"(?:\[(?:Next |Previous )?Page of Results\]\([^)]*\)|(?:Next |Previous )?Page of Results)", line):
            finish()
            identifier, active = None, ""
            result.dropped.append((number, original))
            continue
        if re.fullmatch(r"-{3,}", line):
            finish()
            identifier, active = None, ""
            continue
        if identifier is None:
            result.dropped.append((number, original))
            continue
        label = re.match(r"^(Grade|Domain|Cluster|Standard|Footnote):\s*(.*)$", line)
        if label:
            active = label.group(1)
            fields.setdefault(active, []).append(label.group(2))
        elif active:
            fields[active].append(line)
        else:
            result.unparsed.append((number, original))
    finish()
    return result


def parse_text(
    text: str, *, framework: str, grade: int, source_url: str = "", page: int | None = None,
    ald: bool = False, subject: str = "mathematics", headers: tuple[str, ...] = (),
) -> Parsed:
    """Recognize blocks, retaining unfamiliar lines and flagging uncertain endings."""
    if not ald and is_web_text(text):
        return parse_web_text(text, framework=framework, grade=grade, source_url=source_url, page=page)
    result = Parsed()
    current: re.Match[str] | None = None
    chunks: list[str] = []
    start = 0

    def finish() -> None:
        nonlocal current, chunks
        if current is None:
            return
        outcome = repair_whitespace(chunks)
        label = f"Level {current.group(1)}" if ald else current.group("code")
        if not outcome:
            result.warnings.append(f"Line {start}: {label} has no text; not appendable.")
            result.unparsed.append((start, current.string))
        else:
            if ald:
                result.rows.append(AldRow(framework, grade, subject, int(current.group(1)), outcome))
            else:
                actual_grade = 0 if current.group("grade") == "K" else int(current.group("grade"))
                if actual_grade != grade:
                    result.warnings.append(f"{label}: code grade {actual_grade} disagrees with --grade {grade}; using {actual_grade}.")
                result.rows.append(StandardRow(framework, label, actual_grade, current.group("domain"),
                                               current.group("cluster"), outcome, source_url, page))
            if outcome.endswith(("...", "…", "-", "\u00ad")) or not re.search(r'[.!?][\)\]"”’]*$', outcome):
                result.warnings.append(f"{label}: possibly truncated text; inspect the original PDF (nothing completed or corrected).")
        current, chunks = None, []

    for number, original in enumerate(text.splitlines(), 1):
        line = original.strip()
        if not line:
            continue
        if RUNNING_HEADER.fullmatch(line) or line in headers:
            result.dropped.append((number, original))
            continue
        match = (LEVEL if ald else CODE).match(line)
        if match:
            finish()
            current, start = match, number
            first = (match.group(2) or "") if ald else match.group("text")
            chunks = [first] if first else []
        elif CODE_LIKE.match(line):
            finish()
            result.unparsed.append((number, original))
        elif current is None:
            result.unparsed.append((number, original))
        else:
            chunks.append(original)
    finish()
    return result


def _identity(row: dict[str, Any], ald: bool) -> tuple[str, ...]:
    keys = ("framework_id", "grade", "subject", "level") if ald else ("framework_id", "code")
    return tuple(str(row[key]) for key in keys)


def pending_rows(path: Path, rows: list[StandardRow | AldRow], ald: bool) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate both against the file and within this paste, scoped by framework."""
    expected = ALD_HEADER if ald else STANDARD_HEADER
    seen: set[tuple[str, ...]] = set()
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected:
                raise ValueError(f"{path}: expected exact header {','.join(expected)}")
            for record in reader:
                if None in record or any(value is None for value in record.values()):
                    raise ValueError(f"{path}:{reader.line_num}: malformed existing CSV row")
                seen.add(_identity(record, ald))
    pending = []
    skipped = 0
    for row in rows:
        record = asdict(row)
        identity = _identity(record, ald)
        if identity in seen:
            skipped += 1
        else:
            pending.append(record)
            seen.add(identity)
    return pending, skipped


def preview(rows: list[StandardRow | AldRow], output: TextIO) -> None:
    if not rows:
        print("No rows parsed.", file=output)
        return
    records = [asdict(row) for row in rows]
    names = list(records[0])
    print(" | ".join(names), file=output)
    print(" | ".join("---" for _ in names), file=output)
    for row in records:
        print(" | ".join("" if row[name] is None else str(row[name]).replace("|", "\\|") for name in names), file=output)


def confirm() -> bool:
    """After redirected stdin reaches EOF, obtain confirmation from the terminal."""
    try:
        if sys.stdin.isatty():
            return input("Append these rows? [y/N] ").strip().lower() == "y"
        # Text update mode (r+) requires seeking, which terminals do not support.
        sys.stdout.flush()
        with open("/dev/tty", "r") as terminal_input, open("/dev/tty", "w") as terminal_output:
            terminal_output.write("Append these rows? [y/N] ")
            terminal_output.flush()
            return terminal_input.readline().strip().lower() == "y"
    except (EOFError, OSError):
        print("No interactive confirmation available; nothing appended.")
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--grade", required=True, type=int)
    parser.add_argument("--source-url")
    parser.add_argument("--page", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ald", action="store_true")
    parser.add_argument("--subject", default="mathematics", help="ALD subject column (default: mathematics)")
    parser.add_argument("--header", action="append", default=[], help="Exact extra running-header line to drop; repeatable")
    args = parser.parse_args(argv)
    if args.grade < 0:
        parser.error("--grade must be non-negative; kindergarten is 0")
    if not args.ald and not args.source_url:
        parser.error("standards mode requires --source-url")
    if args.page is not None and args.page < 1:
        parser.error("--page must be positive")
    print("Paste PDF or CDE web text, then press Ctrl-D on an empty line to preview.")
    text = sys.stdin.read()
    if not args.ald and args.page is None and not is_web_text(text):
        parser.error("PDF standards mode requires --page; CDE web text may omit it")
    parsed = parse_text(text, framework=args.framework, grade=args.grade,
                        source_url=args.source_url or "", page=args.page,
                        ald=args.ald, subject=args.subject, headers=tuple(args.header))
    preview(parsed.rows, sys.stdout)
    for warning in parsed.warnings:
        print(f"WARNING: {warning}")
    for label, lines in (("Dropped header/footer", parsed.dropped), ("Unparsed", parsed.unparsed)):
        for number, original in lines:
            print(f"{label} line {number}: {original}")
    path = ROOT / "data" / ("ald.csv" if args.ald else "maths.csv")
    pending, skipped = pending_rows(path, parsed.rows, args.ald)
    print(f"Ready to append: {len(pending)}; duplicates skipped: {skipped}.")
    if args.ald:
        print("ALD uses its existing header; source URL and page are not stored by that schema.")
    appended = 0
    if pending and not args.dry_run and confirm():
        # Recheck after confirmation in case the file changed while reviewing.
        pending, skipped = pending_rows(path, parsed.rows, args.ald)
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
            writer = csv.DictWriter(handle, fieldnames=ALD_HEADER if args.ald else STANDARD_HEADER)
            if empty:
                writer.writeheader()
            writer.writerows(pending)
        appended = len(pending)
    print(f"Rows parsed: {len(parsed.rows)}; rows appended: {appended}; rows skipped as duplicates: {skipped}; unparsed lines: {len(parsed.unparsed)}.")


if __name__ == "__main__":
    main()
