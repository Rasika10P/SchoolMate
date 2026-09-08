-- Additive and repeatable; existing standards and translations are preserved.
ALTER TABLE standard ADD COLUMN IF NOT EXISTS plain_summary TEXT;
ALTER TABLE standard ADD COLUMN IF NOT EXISTS plain_example TEXT;
