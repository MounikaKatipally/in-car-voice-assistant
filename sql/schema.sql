CREATE TABLE IF NOT EXISTS audio_files (
    audio_id VARCHAR(100) PRIMARY KEY,
    audio_path TEXT NOT NULL,
    source VARCHAR(100),
    split VARCHAR(50),
    sample_rate INTEGER,
    duration_seconds NUMERIC(10, 3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id SERIAL PRIMARY KEY,
    audio_id VARCHAR(100) NOT NULL,
    transcript_text TEXT NOT NULL,
    transcript_source VARCHAR(100) DEFAULT 'ground_truth',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_transcripts_audio
        FOREIGN KEY (audio_id)
        REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS intents (
    intent_id SERIAL PRIMARY KEY,
    audio_id VARCHAR(100) NOT NULL,
    intent_label VARCHAR(100),
    confidence_score NUMERIC(5, 4),
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_intents_audio
        FOREIGN KEY (audio_id)
        REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metadata (
    metadata_id SERIAL PRIMARY KEY,
    audio_id VARCHAR(100) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_metadata_audio
        FOREIGN KEY (audio_id)
        REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_audio_id ON transcripts(audio_id);
CREATE INDEX IF NOT EXISTS idx_intents_audio_id ON intents(audio_id);
CREATE INDEX IF NOT EXISTS idx_metadata_audio_id ON metadata(audio_id);
