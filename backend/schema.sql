-- Add new columns to the columns table
ALTER TABLE columns ADD COLUMN ai_confidence_score REAL DEFAULT 0.0;
ALTER TABLE columns ADD COLUMN uncertainty_reason TEXT;
ALTER TABLE columns ADD COLUMN needs_review INTEGER DEFAULT 0;
ALTER TABLE columns ADD COLUMN last_edited TEXT;

-- Create the description_feedback table
CREATE TABLE IF NOT EXISTS description_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column_id INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (column_id) REFERENCES columns(id)
);

-- Create the sample_data table
CREATE TABLE IF NOT EXISTS sample_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column_id INTEGER NOT NULL,
    value TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (column_id) REFERENCES columns(id)
); 