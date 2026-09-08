import csv
import io

import pytest

from scripts import paste_eld_to_csv as eld


PASTE = """Standard Identifier: ELD.PI.K.1.Br
Grade: K
Critical Principle: Part I: Interacting in Meaningful WaysCluster: A. Collaborative
Proficiency Level: BridgingContent Strand: Exchanging information and ideas

Standard:
Contribute to discussions.
Standard Identifier: ELD.PI.1.4.Em
Grade: 1
Critical Principle: Part I: Interacting in Meaningful WaysCluster: A. Collaborative
Proficiency Level: EmergingContent Strand: Adapting language choices
Standard:
No standard for grade 1.
"""


def test_all_eld_fields_and_no_standard_entries_are_preserved():
    rows = eld.parse_text(PASTE, framework="CA-ELD-2012", source_url="https://www2.cde.ca.gov/cacs/eld")
    assert len(rows) == 2
    assert rows[0].grade == 0
    assert rows[0].critical_principle == "Part I: Interacting in Meaningful Ways"
    assert rows[0].cluster == "A. Collaborative"
    assert rows[0].proficiency_level == "Bridging"
    assert rows[0].content_strand == "Exchanging information and ideas"
    assert rows[0].text == "Contribute to discussions."
    assert rows[1].grade == 1
    assert rows[1].text == "No standard for grade 1."


@pytest.mark.parametrize("replacement", ["Grade: 2", "Grade:"])
def test_invalid_record_rejected(replacement):
    with pytest.raises(ValueError):
        eld.parse_text(PASTE.replace("Grade: K", replacement), framework="test", source_url="source")


@pytest.mark.parametrize("approved,dry_run", [(False, False), (True, True), (True, False)])
def test_confirmation_append_and_repeat_are_safe(monkeypatch, tmp_path, approved, dry_run):
    monkeypatch.setattr(eld, "ROOT", tmp_path)
    confirmations = []
    def confirm():
        confirmations.append(True)
        return approved
    monkeypatch.setattr(eld, "confirm", confirm)
    args = ["--framework", "CA-ELD-2012", "--source-url", "https://www2.cde.ca.gov/cacs/eld"]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(eld.sys, "stdin", io.StringIO(PASTE))
    eld.main(args)
    path = tmp_path / "data" / "eld.csv"
    assert path.exists() == (approved and not dry_run)
    assert len(confirmations) == (0 if dry_run else 1)
    if path.exists():
        before = path.read_bytes()
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == eld.HEADER
            records = list(reader)
        assert len(records) == 2 and all(row["page"] == "" for row in records)
        monkeypatch.setattr(eld.sys, "stdin", io.StringIO(PASTE))
        eld.main(args)
        assert len(confirmations) == 1
        assert path.read_bytes() == before
