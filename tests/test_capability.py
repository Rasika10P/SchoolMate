from api import capability


def test_capability_matrix() -> None:
    expected = {
        "math": (True, True, True, True, True),
        "ela": (True, True, True, True, True),
        "eld": (True, True, True, True, False),
        "sci": (True, True, False, True, True),
        "hss": (True, False, False, True, False),
        "vapa": (True, False, False, True, True),
        "pe": (True, False, False, True, True),
    }
    tabs = ("standards", "next_steps", "standing", "activities", "programs")
    assert set(capability.SUBJECTS) == set(expected) == set(capability.CAPABILITY)
    for subject, values in expected.items():
        assert tuple(capability.supports(subject, tab, 3) for tab in tabs) == values
        for tab, value in zip(tabs, values):
            if not value:
                reason = capability.NOT_PUBLISHED_REASON[subject, tab]
                assert reason.endswith(".")


def test_science_standing_grade_five_only() -> None:
    assert capability.supports("sci", "standing", 5)
    assert not capability.supports("sci", "standing", 3)
    assert not capability.supports("sci", "standing")
