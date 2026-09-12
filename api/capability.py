"""Shared publication capabilities for tools and UI; kindergarten is grade zero."""

from typing import Literal


SUBJECTS: tuple[str, ...] = ("math", "ela", "eld", "sci", "hss", "vapa", "pe")
CAPABILITY: dict[str, dict[str, bool | Literal["grade_5_only"]]] = {
    "math": dict(standards=True, next_steps=True, standing=True, activities=True, programs=True),
    "ela": dict(standards=True, next_steps=True, standing=True, activities=True, programs=True),
    "eld": dict(standards=True, next_steps=True, standing=True, activities=True, programs=False),
    "sci": dict(standards=True, next_steps=True, standing="grade_5_only", activities=True, programs=True),
    "hss": dict(standards=True, next_steps=False, standing=False, activities=True, programs=False),
    "vapa": dict(standards=True, next_steps=False, standing=False, activities=True, programs=True),
    "pe": dict(standards=True, next_steps=False, standing=False, activities=True, programs=True),
}

NOT_PUBLISHED_REASON: dict[tuple[str, str], str] = {
    ("sci", "standing"): "California science achievement descriptors are available here for grade 5 only, so we cannot show a state-published standing description for this grade.",
    ("hss", "standing"): "California does not publish achievement descriptors for history and social science in this guide, so standards describe what children learn rather than a statewide achievement level.",
    ("vapa", "standing"): "California does not publish achievement descriptors for visual and performing arts in this guide, so we cannot assign a state-published achievement level.",
    ("pe", "standing"): "California does not publish achievement descriptors for physical education in this guide, so fitness activities should not be treated as a state-published achievement level.",
    ("hss", "next_steps"): "California does not publish a prerequisite progression for history and social science in this guide, so we cannot label earlier or later standards as official next steps.",
    ("vapa", "next_steps"): "California does not publish a prerequisite progression for visual and performing arts in this guide, so we cannot label earlier or later standards as official next steps.",
    ("pe", "next_steps"): "California does not publish a prerequisite progression for physical education in this guide, so we cannot label earlier or later standards as official next steps.",
    ("hss", "programs"): "No history and social science programs are published in this guide's catalog; this does not mean that no local opportunities exist.",
    ("eld", "programs"): "There are no separate English language development programs in this catalog, so we show English language arts programs that may interest your child instead.",
}
# External programs are not California publications. A grade-filtered empty
# catalog must not imply that California says no such programs exist anywhere.
for _subject in SUBJECTS:
    NOT_PUBLISHED_REASON.setdefault(
        (_subject, "programs"),
        "No program in this guide's catalog matches your child's grade; this does not mean that no local opportunities exist.",
    )


def supports(subject: str, tab: str, grade: int | None = None) -> bool:
    """Resolve publication support, including the science grade restriction."""
    if subject not in CAPABILITY:
        raise ValueError(f"Unsupported subject {subject!r}")
    if tab not in CAPABILITY[subject]:
        raise ValueError(f"Unsupported tab {tab!r}")
    value = CAPABILITY[subject][tab]
    return grade == 5 if value == "grade_5_only" else bool(value)


# Parent-facing heading and one-line description for each stored domain.
_DOMAIN_DETAILS: dict[str, tuple[str, str]] = {
    "CC": ("Counting", "Naming numbers and counting objects."),
    "OA": ("Adding and subtracting", "Putting amounts together, taking them apart, and noticing number patterns."),
    "NBT": ("Understanding numbers", "Understanding how digits represent amounts and using numbers in calculations."),
    "NF": ("Parts of a whole", "Working with equal parts, fractions, and their relationships."),
    "G": ("Shapes and space", "Exploring shapes, their features, and where things are."),
    "MD": ("Measuring and comparing", "Measuring amounts and using information to make comparisons."),
    "Language": ("Using words and sentences", "Choosing words and building sentences to communicate clearly."),
    "Reading: Foundational Skills": ("Learning how reading works", "Connecting letters, sounds, and words when reading."),
    "Reading: Informational Text": ("Reading to learn", "Finding and discussing information in books and other texts."),
    "Reading: Literature": ("Stories and poems", "Exploring characters, events, and ideas in stories and poems."),
    "Speaking and Listening": ("Talking and listening", "Sharing ideas and understanding what other people say."),
    "Writing": ("Putting ideas into writing", "Expressing ideas and information in written words."),
    "Part I: Interacting in Meaningful Ways": ("Communicating in English", "Using English to exchange ideas and understand spoken and written messages."),
    "Part II: Learning About How English Works": ("Building English sentences", "Exploring how English words and sentences fit together to express meaning."),
    "Earth and Space Science": ("Earth, weather, and space", "Exploring the Earth, its changing conditions, and the sky."),
    "Engineering, Technology, and Applications of Science": ("Designing and making things", "Exploring problems and designing possible solutions."),
    "Life Science": ("Living things", "Exploring plants, animals, and their surroundings."),
    "Physical Science": ("How things move and change", "Exploring materials, motion, and changes in the physical world."),
    "Dance": ("Dance and movement", "Creating, performing, and responding to movement."),
    "Media Arts": ("Creating with media", "Creating and discussing art made with media tools."),
    "Music": ("Making and exploring music", "Creating, performing, and listening to music."),
    "Theatre": ("Acting and storytelling", "Creating, performing, and discussing stories through acting."),
    "Visual Arts": ("Making and looking at art", "Creating artwork and exploring what it communicates."),
    "Physical Education": ("Movement, health, and working together", "Exploring movement skills, physical activity, and cooperation."),
    "Learning and Working Now and Long Ago, Grade K": ("Life now and long ago", "Exploring daily life, shared rules, and the past."),
    "A Child’s Place in Time and Space, Grade 1": ("Our place in the world", "Exploring communities, places, and life in the past."),
    "People Who Make a Difference, Grade 2": ("People and communities", "Exploring people, places, and their contributions to community life."),
    "Continuity and Change, Grade 3": ("How communities change", "Exploring local history and what changes or stays the same."),
    "California: A Changing State, Grade 4": ("California’s story", "Exploring California’s people, places, and history."),
    "United States History and Geography: Making a New Nation, Grade 5": ("The story of the United States", "Exploring the people, places, and events involved in forming a nation."),
}
for _code, _name in {
    "CC": "Counting and Cardinality", "OA": "Operations and Algebraic Thinking",
    "NBT": "Number and Operations in Base Ten", "NF": "Number and Operations—Fractions",
    "G": "Geometry", "MD": "Measurement and Data",
}.items():
    _DOMAIN_DETAILS[_name] = _DOMAIN_DETAILS[_code]


# Insertion order follows the domains in each source framework.
DOMAIN_LABELS: dict[str, dict[str, str]] = {
    "CA-CCSSM-2013": {
        "CC": "Counting", "OA": "Adding, subtracting, multiplying and dividing",
        "NBT": "Understanding place value", "NF": "Working with fractions",
        "MD": "Measuring things", "G": "Shapes and space",
    },
    "CA-CCSS-ELA-2013": {
        "RL": "Reading stories and poems", "RI": "Reading factual texts",
        "RF": "Sounding out and reading fluently", "W": "Writing",
        "SL": "Speaking and listening", "L": "Grammar, spelling and vocabulary",
    },
}
_DOMAIN_ALIASES = {
    "CA-CCSSM-2013": dict(zip(
        ["Counting and Cardinality", "Operations and Algebraic Thinking",
         "Number and Operations in Base Ten", "Number and Operations—Fractions",
         "Measurement and Data", "Geometry"], ["CC", "OA", "NBT", "NF", "MD", "G"])),
    "CA-CCSS-ELA-2013": dict(zip(
        ["Reading: Literature", "Reading: Informational Text", "Reading: Foundational Skills",
         "Writing", "Speaking and Listening", "Language"], ["RL", "RI", "RF", "W", "SL", "L"])),
}


def standard_domain(framework: str, domain: str | None, code: str = "") -> str:
    """Normalize stored names, deriving a missing domain from the standard code."""
    if domain and domain.strip():
        value = domain.strip()
        return _DOMAIN_ALIASES.get(framework, {}).get(value, value)
    parts = code.split(".")
    if framework == "CA-CCSSM-2013" and len(parts) > 1:
        return parts[1]
    if framework == "CA-CCSS-ELA-2013" and parts[0]:
        return parts[0]
    return code or "Unspecified"


def ordered_domains(framework: str, domains: list[str]) -> list[str]:
    """Known source order first; preserve input order for unmapped domains."""
    unique = list(dict.fromkeys(domains))
    return [d for d in DOMAIN_LABELS.get(framework, {}) if d in unique] + [
        d for d in unique if d not in DOMAIN_LABELS.get(framework, {})]


def domain_label(domain: str, grade: int, framework: str = "") -> tuple[str, str]:
    code = standard_domain(framework, domain)
    raw = next((name for name, alias in _DOMAIN_ALIASES.get(framework, {}).items()
                if alias == code), code)
    detail = _DOMAIN_DETAILS.get(raw, _DOMAIN_DETAILS.get(code, (code, "")))
    return DOMAIN_LABELS.get(framework, {}).get(code, detail[0]), detail[1]
