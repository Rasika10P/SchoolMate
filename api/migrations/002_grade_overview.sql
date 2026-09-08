CREATE TABLE IF NOT EXISTS grade_overview (
    framework_id TEXT NOT NULL REFERENCES framework(id),
    grade INT NOT NULL,
    domain_key TEXT NOT NULL,
    overview TEXT NOT NULL,
    PRIMARY KEY (framework_id, grade)
);
