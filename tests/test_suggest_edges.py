import csv

import pytest

from scripts.suggest_edges import main, read_standards, suggest

F = 'CA-CCSS-ELA-2013'


def row(code, grade, **extra):
    return dict(framework_id=F, code=code, grade=str(grade), text='Official text, with punctuation.', **extra)


def test_adjacent_strand_and_number_only():
    rows = [row('RL.K.1', 0), row('RL.1.1', 1), row('RL.2.1', 2),
            row('RI.2.1', 2), row('RL.2.2', 2), row('RL.4.1', 4)]
    edges, unmatched = suggest(rows)
    assert [(e['from_code'], e['to_code']) for e in edges] == [('RL.K.1', 'RL.1.1'), ('RL.1.1', 'RL.2.1')]
    assert {r['code'] for r in unmatched} == {'RI.2.1', 'RL.2.2', 'RL.4.1'}
    assert all(e['relation'] == '' and 'unverified' in e['confidence_note'] for e in edges)


def test_subparts_proficiency_and_unsupported_codes():
    rows = [row('ELD.PI.1.1.Em', 1), row('ELD.PI.2.1.Em', 2), row('ELD.PI.2.1.Br', 2),
            row('1.OA.1.a', 1), row('2.OA.1.b', 2), row('K-2-ETS1-1', 0), row('RL.2.1', 1)]
    edges, unmatched = suggest(rows)
    assert len(edges) == 1
    assert edges[0]['to_code'] == 'ELD.PI.2.1.Em'
    assert len(unmatched) == 5


def test_math_cluster_letter_does_not_change_number():
    edges, _ = suggest([row('3.NF.A.1', 3), row('4.NF.1', 4)])
    assert len(edges) == 1


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_cli_separate_outputs_preserves_progression(tmp_path):
    write_rows(tmp_path / 'ela.csv', [row('RL.1.1', 1), row('RL.2.1', 2), row('W.2.2', 2)])
    progression = tmp_path / 'progression.csv'
    progression.write_text('untouched\n')
    output = tmp_path / 'review.csv'
    argv = ['--framework', F, '--data-dir', str(tmp_path), '--output', str(output)]
    assert main(argv) == 0
    with output.open() as handle:
        edges = list(csv.DictReader(handle))
    assert edges[0]['from_text'] == 'Official text, with punctuation.'
    assert (tmp_path / 'review-unmatched.csv').exists()
    assert progression.read_text() == 'untouched\n'
    with pytest.raises(SystemExit):
        main(argv)
    with pytest.raises(SystemExit):
        main(argv[:-1] + [str(progression)])
    assert progression.read_text() == 'untouched\n'


def test_wrong_framework_rejected(tmp_path):
    path = tmp_path / 'ela.csv'
    write_rows(path, [row('RL.1.1', 1)])
    with pytest.raises(ValueError, match='row 2: expected'):
        read_standards(path, 'CA-CCSSM-2013')
