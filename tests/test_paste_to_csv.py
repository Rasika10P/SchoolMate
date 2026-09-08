import csv
import io
import os
from pathlib import Path
import pty

import pytest

from scripts import paste_to_csv as paste


@pytest.mark.parametrize("answer,expected", [("y\n", True), ("Y\n", True), ("\n", False), ("n\n", False)])
def test_redirected_stdin_confirmation_uses_terminal(
    monkeypatch: pytest.MonkeyPatch, answer: str, expected: bool,
) -> None:
    master, slave = pty.openpty()
    terminal_path = os.ttyname(slave)
    real_open = open

    def open_terminal(path: str, mode: str):
        assert path == "/dev/tty"
        return real_open(
            terminal_path, mode,
            opener=lambda name, flags: os.open(name, flags | os.O_NOCTTY),
        )

    monkeypatch.setattr(paste.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr("builtins.open", open_terminal)
    try:
        os.write(master, answer.encode())
        assert paste.confirm() is expected
    finally:
        os.close(master)
        os.close(slave)


MESSY = """Unrecognized introduction
4.NF.A.1 Explain why a fraction a/b is equiv-
alent to a fraction (n × a)/(n × b) by using visual
California Department of Education
27
fraction models, with attention to how the number and size
of the parts differ even though the two fractions themselves are the same size.
4.NF.A.2 Compare two fractions with different numerators and
different denominators.
"""


def parse(text: str = MESSY, **kw: object) -> paste.Parsed:
    return paste.parse_text(text, framework="CA-CCSSM-2013", grade=4,
                            source_url="https://example.com/source.pdf", page=27, **kw)


def test_messy_pdf_preserves_words_and_repairs_wrapping() -> None:
    result = parse()
    assert len(result.rows) == 2
    assert result.rows[0].text == (
        "Explain why a fraction a/b is equivalent to a fraction (n × a)/(n × b) by using visual "
        "fraction models, with attention to how the number and size of the parts differ even though "
        "the two fractions themselves are the same size."
    )
    assert result.rows[1].text == "Compare two fractions with different numerators and different denominators."
    assert len(result.dropped) == 2
    assert result.unparsed == [(1, "Unrecognized introduction")]
    assert not result.warnings


def test_code_metadata_and_truncation() -> None:
    result = parse("3.NF.A.1a Explain why the numerator\n")
    row = result.rows[0]
    assert (row.grade, row.domain, row.cluster, row.code) == (3, "NF", "A", "3.NF.A.1a")
    assert row.text == "Explain why the numerator"
    assert any("disagrees" in warning for warning in result.warnings)
    assert any("truncated" in warning for warning in result.warnings)


def test_ald_blocks() -> None:
    result = parse("Level 1\nRecognizes frac-\ntions with support.\nPage 2\nLevel 2:\nCompares simple fractions.", ald=True)
    assert [row.level for row in result.rows] == [1, 2]
    assert result.rows[0].text == "Recognizes fractions with support."
    assert result.rows[0].subject == "mathematics"


def test_duplicates_are_scoped_by_framework(tmp_path: Path) -> None:
    path = tmp_path / "maths.csv"
    rows = parse().rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paste.STANDARD_HEADER)
        writer.writeheader()
        writer.writerow(paste.asdict(rows[0]))
        writer.writerow({**paste.asdict(rows[1]), "framework_id": "other-framework"})
    pending, skipped = paste.pending_rows(path, rows + [rows[1]], False)
    assert skipped == 2
    assert [row["code"] for row in pending] == ["4.NF.A.2"]


@pytest.mark.parametrize("dry_run,approved,expected", [(True, True, 0), (False, False, 0), (False, True, 2)])
def test_cli_confirmation_and_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    dry_run: bool, approved: bool, expected: int,
) -> None:
    monkeypatch.setattr(paste, "ROOT", tmp_path)
    monkeypatch.setattr(paste.sys, "stdin", io.StringIO(MESSY))
    confirmations: list[bool] = []
    def confirm() -> bool:
        confirmations.append(True)
        return approved
    monkeypatch.setattr(paste, "confirm", confirm)
    args = ["--framework", "CA-CCSSM-2013", "--grade", "4", "--source-url", "https://example.com", "--page", "27"]
    paste.main(args + (["--dry-run"] if dry_run else []))
    output = capsys.readouterr().out
    assert f"rows appended: {expected}" in output
    assert "Unrecognized introduction" in output
    path = tmp_path / "data" / "maths.csv"
    assert path.exists() == bool(expected)
    assert bool(confirmations) == (not dry_run)
    if expected:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == paste.STANDARD_HEADER
            assert len(list(reader)) == expected


def test_malformed_code_is_reported_in_full() -> None:
    text = "4.NF.A.1 Some valid text.\n4.NF.??? Broken code here\nUnattached continuation"
    result = parse(text)
    assert result.rows[0].text == "Some valid text."
    assert result.unparsed == [(2, "4.NF.??? Broken code here"), (3, "Unattached continuation")]


WEB_PASTE = (Path(__file__).parent / "fixtures" / "cde_kindergarten.md").read_text()


def test_bulleted_web_records_navigation_and_california_substandard() -> None:
    text = """* Standard Identifier: 2.NBT.7.1
* Grade: 2
* Domain: Number and Operations in Base Ten
*
* Cluster:
* Use place value understanding.
* Standard:
* Use estimation strategies to make reasonable estimates in problem solving. CA
* Page of Results
Standard Identifier: 2.NBT.8
Grade: 2
Domain: Number and Operations in Base Ten
Cluster: Use place value understanding.
Standard: Mentally add 10.
* Next Page of Results
* 
"""
    result = paste.parse_text(text, framework="test", grade=2, source_url="https://www2.cde.ca.gov/cacs/math")
    assert [row.code for row in result.rows] == ["2.NBT.7.1", "2.NBT.8"]
    assert result.rows[0].text == "Use estimation strategies to make reasonable estimates in problem solving. CA"
    assert result.rows[1].text == "Mentally add 10."
    assert not result.warnings and not result.unparsed
    assert len(result.dropped) == 2


def test_cde_web_paste_preserves_identifiers_text_and_sources() -> None:
    result = paste.parse_text(WEB_PASTE, framework="CA-CCSSM-2013", grade=0)
    assert [row.code for row in result.rows] == [
        "K.CC.1", "K.CC.2", "K.CC.3", "K.CC.4.a", "K.CC.4.b",
        "K.CC.4.c", "K.CC.5", "K.CC.6", "K.CC.7", "K.G.1",
    ]
    assert not result.warnings and not result.unparsed
    assert all(row.grade == 0 and row.page is None for row in result.rows)
    assert result.rows[0].text == "Count to 100 by ones and by tens."
    assert result.rows[0].cluster == "Know number names and the count sequence."
    assert result.rows[0].source_url == "https://www2.cde.ca.gov/cacs/id/web/256"
    assert result.rows[7].text.endswith(" Footnote: Includes groups with up to ten objects.")
    assert result.rows[-1].domain == "G"
    assert result.rows[-1].text.endswith("beside, in front of, behind, and next to.")
    assert all("**" not in row.text and "\\" not in row.text for row in result.rows)


def test_web_incomplete_record_is_not_appendable() -> None:
    result = paste.parse_text("Standard Identifier: K.CC.1\nGrade: K\nStandard: Count.",
                             framework="test", grade=0)
    assert not result.rows
    assert "missing Domain, Cluster" in result.warnings[0]


def test_web_cli_without_pdf_page(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(paste, "ROOT", tmp_path)
    monkeypatch.setattr(paste.sys, "stdin", io.StringIO(WEB_PASTE))
    monkeypatch.setattr(paste, "confirm", lambda: True)
    paste.main(["--framework", "CA-CCSSM-2013", "--grade", "0", "--source-url", "https://www2.cde.ca.gov/cacs/math"])
    with (tmp_path / "data" / "maths.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 10
    assert all(row["page"] == "" for row in records)
    assert "rows appended: 10" in capsys.readouterr().out
