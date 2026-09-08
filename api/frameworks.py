"""Shared framework routing for ingestion and curriculum tools."""

FRAMEWORK_BY_SUBJECT: dict[str, str] = {
    "math": "CA-CCSSM-2013",
    "ela": "CA-CCSS-ELA-2013",
    "eld": "CA-ELD-2012",
    "sci": "CA-NGSS-2013",
    "hss": "CA-HSS-2016",
    "vapa": "CA-VAPA-2019",
    "pe": "CA-PE-2005",
}

SUBJECT_BY_FILE = {
    "maths.csv": "math", "ela.csv": "ela", "eld.csv": "eld",
    "science.csv": "sci", "history_social_science.csv": "hss",
    "vapa.csv": "vapa", "pe.csv": "pe",
}
SUBJECT_FILES = {filename: FRAMEWORK_BY_SUBJECT[subject]
                 for filename, subject in SUBJECT_BY_FILE.items()}
