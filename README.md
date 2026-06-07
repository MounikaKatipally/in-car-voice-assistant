# In-Car Voice Assistant

End-to-end conversational AI system that transcribes speech, identifies user intent, and answers context-aware queries � modeled after Cerence AI's in-car voice assistant platform.

## Problem Statement

Build a voice assistant that:
- Transcribes spoken commands using Automatic Speech Recognition (ASR)
- Classifies user intent (navigation, climate, music, etc.)
- Returns context-aware responses using Retrieval-Augmented Generation (RAG)

## Planned Stack

- **ASR:** Whisper
- **Intent:** DeBERTa-V3 with LoRA fine-tuning
- **RAG:** LangChain + FAISS
- **Serving:** FastAPI + Streamlit
- **Storage:** PostgreSQL
- **Packaging:** Docker

## System Architecture

Day 2 architecture diagram:

[View the Mermaid Architecture Diagram](docs/architecture-day2.md)

The system flow is:

User → Streamlit UI → FastAPI → Audio Input → Preprocessing → Whisper ASR → DeBERTa-V3 Intent Classifier → LangChain + FAISS RAG → Final Response
## Environment and Dependencies

Day 3 setup includes the initial Python dependencies and Docker container skeleton.

Core dependencies:
- transformers
- torch
- librosa
- spaCy
- pandas
- LangChain
- FAISS
- Streamlit
- FastAPI
- pytest

Docker is used to make the project reproducible across different machines.

## Dataset

For Day 4, this project uses a small LibriSpeech sample for early Automatic Speech Recognition (ASR) development and testing.

### Dataset Details

- **Source:** LibriSpeech dev-clean validation split
- **Initial subset:** 1,000 audio clips
- **Audio format:** WAV
- **Sample rate:** 16 kHz
- **Metadata file:** `data/metadata/librispeech_sample_1000.csv`

### Metadata Fields

The metadata CSV contains:

- `audio_id`
- `audio_path`
- `transcript`
- `sample_rate`
- `source`
- `split`

### Data Storage Decision

The raw audio files are stored locally in:

```text
data/raw/librispeech_sample/
```

They are excluded from GitHub using `.gitignore` to keep the repository lightweight.

Only the dataset preparation script and metadata file are committed to GitHub:

```text
scripts/prepare_librispeech_sample.py
data/metadata/librispeech_sample_1000.csv
```

### Regenerate the Dataset Sample

To regenerate the 1,000-clip LibriSpeech sample, run:

```powershell
python scripts\prepare_librispeech_sample.py
```
## Database Schema

For Day 5, PostgreSQL is added as the local database layer for the project.

The database is started locally using Docker Compose.

### Database Details

- **Database name:** `voice_assistant_db`
- **Database user:** `voice_user`
- **PostgreSQL container:** `in_car_voice_postgres`
- **SQL schema file:** `sql/schema.sql`
- **Docker Compose file:** `docker-compose.yml`

### Core Tables

- `audio_files`: stores audio file paths, source, split, sample rate, and duration
- `transcripts`: stores ground-truth or model-generated transcripts
- `intents`: stores predicted intent labels, confidence scores, and model name
- `metadata`: stores flexible key-value metadata for each audio file

### Run PostgreSQL Locally

To start PostgreSQL locally:

```powershell
docker compose up -d
```

To check that the container is running:

```powershell
docker ps
```

To connect to the database:

```powershell
docker exec -it in_car_voice_postgres psql -U voice_user -d voice_assistant_db
```

Inside the PostgreSQL prompt, list the tables:

```sql
\dt
```

To exit PostgreSQL:

```sql
\q
```
## Data Ingestion Pipeline

For Days 6-7, the project adds a Python ingestion pipeline that loads the prepared LibriSpeech metadata into PostgreSQL.

The ingestion script connects the local audio dataset with the database layer so the system can track audio files, transcripts, sample rates, durations, and ingestion status in a structured way.

### Ingestion Script

```text
scripts/ingest_librispeech_to_postgres.py
```

### What the Pipeline Does

- Reads `data/metadata/librispeech_sample_1000.csv`
- Verifies that each local audio file exists
- Extracts audio duration and sample rate using `soundfile`
- Inserts audio records into the `audio_files` table
- Inserts ground-truth transcripts into the `transcripts` table
- Adds logging and error handling for traceability
- Writes ingestion logs locally under `logs/`

### Ingestion Results

The pipeline successfully ingested:

- **1,000 records** into `audio_files`
- **1,000 records** into `transcripts`
- **0 failed records**

### Run the Ingestion Pipeline

Make sure PostgreSQL is running:

```powershell
docker compose up -d
```

Run the ingestion script:

```powershell
python scripts\ingest_librispeech_to_postgres.py
```

### Verify Records in PostgreSQL

Connect to the database:

```powershell
docker exec -it in_car_voice_postgres psql -U voice_user -d voice_assistant_db
```

Check record counts:

```sql
SELECT COUNT(*) FROM audio_files;
SELECT COUNT(*) FROM transcripts;
```

Expected result:

```text
1000
```
## Audio Preprocessing Pipeline

For Day 8, the project adds an audio preprocessing pipeline to prepare speech data before ASR model development.

The preprocessing script standardizes the LibriSpeech audio dataset into a cleaner and more consistent format.

### Preprocessing Steps

- Converts audio to mono
- Resamples audio to 16 kHz
- Normalizes audio loudness
- Applies noise reduction
- Saves processed WAV files locally
- Creates a processed metadata CSV

### Script

```text
scripts/preprocess_audio.py
```

### Processed Audio Location

```text
data/processed/librispeech_16k/
```

### Processed Metadata File

```text
data/metadata/librispeech_processed_1000.csv
```

### Run the Preprocessing Pipeline

```powershell
python scripts\preprocess_audio.py
```

## Transcript Cleaning Pipeline

For Day 9, the project adds a transcript cleaning pipeline to standardize text before ASR evaluation and intent classification.

The cleaning script prepares transcript text by:

- Lowercasing all transcripts
- Removing punctuation
- Normalizing whitespace
- Creating word counts
- Adding estimated word-level timestamp alignment where possible

### Script

```text
scripts/clean_transcripts.py
```

### Input

```text
data/metadata/librispeech_processed_1000.csv
```

### Output

```text
data/metadata/librispeech_cleaned_transcripts_1000.csv
```

### Timestamp Alignment Note

LibriSpeech does not provide exact word-level timestamps by default, so this pipeline uses an estimated even word spacing method based on audio duration and word count. This can later be replaced with Whisper-generated timestamps.

## Intent Annotation Dataset

For Day 10, the project adds a small intent annotation dataset for in-car voice commands.

The dataset contains 200 manually seeded command examples across 10 intent classes.

### Intent Classes

- `navigation`
- `play_music`
- `call_contact`
- `weather`
- `climate_control`
- `radio`
- `settings`
- `traffic`
- `cancel`
- `confirm`

### Dataset File

```text
data/metadata/intent_annotations_200.csv
```

### Annotation Method

The initial dataset uses manually written in-car command examples with rule-based intent labels. This creates a balanced seed dataset for future intent classification experiments.

Each intent has 20 examples, for a total of 200 labeled commands.

## ETL Orchestration Pipeline

For Day 11, the project adds a single ETL orchestration script that wraps dataset preparation, audio preprocessing, transcript cleaning, intent annotation, and PostgreSQL ingestion.

ETL stands for:

- **Extract:** Prepare the LibriSpeech dataset sample
- **Transform:** Preprocess audio, clean transcripts, and create intent annotations
- **Load:** Insert structured records into PostgreSQL

### Pipeline Script

```text
scripts/run_etl_pipeline.py
```

### Pipeline Steps

The orchestration script runs the following steps in order:

1. Prepare LibriSpeech sample
2. Preprocess audio
3. Clean transcripts
4. Create intent annotation dataset
5. Ingest LibriSpeech records into PostgreSQL

### Idempotency

The pipeline is designed to be idempotent, meaning it can be safely re-run without duplicating work.

If an expected output file already exists, that step is skipped. The PostgreSQL ingestion step is also safe to re-run because audio records are updated on conflict, and duplicate ground-truth transcripts are avoided.

### Run the ETL Pipeline

```powershell
python scripts\run_etl_pipeline.py
```

To force all steps to rerun:

```powershell
python scripts\run_etl_pipeline.py --force
```

## Data Governance

For Day 12, the project adds a basic data governance and compliance layer.

The governance layer includes:

- Pseudonymization of speaker identifiers using SHA-256 hashing
- Local-only storage for raw and processed audio files
- GitHub exclusion rules for large files, logs, virtual environments, secrets, and model artifacts
- Documentation of data handling practices in `DATA_GOVERNANCE.md`
- ISO/IEC 27001-inspired principles such as confidentiality, integrity, availability, traceability, risk management, and continuous improvement

### Governance Document

```text
DATA_GOVERNANCE.md
```
## Phase 2: Feature Engineering

Phase 2 focuses on converting processed audio, cleaned transcripts, and metadata into machine learning-ready features.

This phase prepares the project for future model development, evaluation, validation, and visualization.

## Audio Feature Extraction

For Day 13, the project adds an audio feature extraction pipeline using `librosa`.

The script extracts machine learning-ready audio representations from the processed LibriSpeech audio files.

### Extracted Audio Features

- MFCCs
- Mel spectrograms
- Pitch estimates
- Tempo estimates

### Script

```text
scripts/extract_audio_features.py
```

### Feature Output Location

```text
data/features/audio_npy/
```

### Feature Manifest

```text
data/features/audio_feature_manifest_1000.csv
```

The `.npy` feature files are stored locally and excluded from GitHub using `.gitignore`.

Only the feature extraction script and manifest CSV are committed.

### Day 13 Result

- 1,000 processed audio clips analyzed
- 4,000 `.npy` feature files generated locally
- 1 feature manifest CSV created
- 0 failed files

### Run Audio Feature Extraction

```powershell
python scripts\extract_audio_features.py
```

## Text Feature Extraction

For Day 14, the project adds transcript-level text feature extraction using `spaCy`.

The script processes cleaned transcripts and generates NLP features for downstream modeling and analysis.

### Extracted Text Features

- Tokens
- Lemmas
- POS tags
- Word count
- Character count
- Average word length
- Noun count
- Verb count
- Adjective count
- Adverb count

### Script

```text
scripts/extract_text_features.py
```

### Output

```text
data/features/text_features_1000.jsonl
```

### Run Text Feature Extraction

```powershell
python scripts\extract_text_features.py
```

## Metadata Feature Engineering

For Day 15, the project derives structured metadata features from the pseudonymized metadata file.

### Derived Metadata Features

- Clip duration
- Duration bucket
- Synthetic time of day
- Speaker gender availability field
- Sample rate
- Word count
- Word rate
- Transcript availability flag
- Speaker hash

### Script

```text
scripts/build_metadata_features.py
```

### Output

```text
data/features/metadata_features_1000.csv
```

### Run Metadata Feature Engineering

```powershell
python scripts\build_metadata_features.py
```

## Multimodal Feature Store

For Day 16, the project combines audio feature summaries, text features, and metadata features into one multimodal feature store.

The feature store is indexed by `record_id` and saved as a Parquet file.

### Feature Store Includes

- Audio feature paths
- Audio feature summary statistics
- Text feature counts
- Metadata features
- Duration and sample rate fields
- Speaker hash
- Transcript availability

### Script

```text
scripts/build_feature_store.py
```

### Output

```text
data/features/multimodal_feature_store_1000.parquet
```

### Run Feature Store Build

```powershell
python scripts\build_feature_store.py
```

## Feature Validation

For Day 17, the project adds automated feature validation using `pytest`.

The tests verify that the feature store is consistent, complete, and ready for model development.

### Validation Checks

- Feature store shape
- Missing value checks
- Sample rate validation
- Duration range validation
- Word count validation
- Duration bucket validation
- Duplicate audio ID checks

### Test File

```text
tests/test_feature_store.py
```

### Run Tests

```powershell
pytest tests\test_feature_store.py
```

## Feature Dashboard

For Day 18, the project adds a Streamlit dashboard for visualizing feature distributions and inspecting the multimodal feature store.

### Dashboard Includes

- Feature store overview
- Record and feature counts
- Dataset preview
- Duration distribution
- Word count trends
- MFCC mean distribution
- Pitch mean distribution
- Feature correlation preview
- Duration bucket filtering

### Dashboard File

```text
app/feature_dashboard.py
```

### Run Dashboard

```powershell
streamlit run app\feature_dashboard.py
```

### Pseudonymization Script

```text
scripts/pseudonymize_metadata.py
```

### Pseudonymized Metadata Output

```text
data/metadata/librispeech_pseudonymized_metadata_1000.csv
```

### Privacy Note

Pseudonymization is not the same as full anonymization. It reduces direct identification risk, but it does not guarantee that re-identification is impossible.

## Project Timeline

50-day independent build, documented daily on LinkedIn and GitHub.

## Status

**Day 18 of 50** — Feature engineering phase completed.

Completed so far:

- **Day 1:** Project charter, GitHub repository, virtual environment, starter folders, and smoke test completed
- **Day 2:** System architecture diagram created using Mermaid and linked in README
- **Day 3:** Environment dependencies, `requirements.txt`, Dockerfile skeleton, `.dockerignore`, and dependency check script completed
- **Day 4:** LibriSpeech dataset preparation completed with 1,000 audio clips generated locally and metadata committed to GitHub
- **Day 5:** PostgreSQL database schema added using Docker Compose
- **Days 6-7:** LibriSpeech ingestion pipeline completed with 1,000 audio records and 1,000 transcripts inserted into PostgreSQL
- **Day 8:** Audio preprocessing pipeline completed with normalization, resampling, and noise reduction
- **Day 9:** Transcript cleaning pipeline completed with lowercase normalization, punctuation removal, spacing cleanup, and estimated timestamp alignment
- **Day 10:** Intent annotation seed dataset completed with 200 labeled in-car command examples across 10 intent classes
- **Day 11:** ETL orchestration pipeline completed with idempotent, re-runnable pipeline execution
- **Day 12:** Data governance layer added with pseudonymized speaker identifiers and documented data handling practices
- **Day 13:** Audio feature extraction completed with MFCCs, mel spectrograms, pitch, and tempo
- **Day 14:** Text feature extraction completed with tokenization, lemmatization, and POS tagging
- **Day 15:** Metadata feature engineering completed with duration buckets, synthetic time of day, word rate, and speaker metadata fields
- **Day 16:** Multimodal feature store created as a Parquet file indexed by `record_id`
- **Day 17:** Feature validation added using pytest checks
- **Day 18:** Streamlit feature dashboard added for visualizing feature distributions

## Roadmap

- [x] Day 1: Repo + environment setup
- [x] Day 2: Architecture diagram + README architecture link
- [x] Day 3: Dependencies + Docker skeleton
- [x] Day 4: Public dataset preparation with LibriSpeech
- [x] Day 5: PostgreSQL database schema with Docker Compose
- [x] Days 6-7: Data ingestion pipeline into PostgreSQL
- [x] Day 8: Audio preprocessing pipeline
- [x] Day 9: Transcript cleaning and timestamp alignment
- [x] Day 10: Intent annotation seed dataset
- [x] Day 11: ETL orchestration pipeline
- [x] Day 12: Data governance and pseudonymization layer
- [x] Day 13: Audio feature extraction
- [x] Day 14: Text feature extraction
- [x] Day 15: Metadata feature engineering
- [x] Day 16: Multimodal feature store
- [x] Day 17: Feature validation with pytest
- [x] Day 18: Streamlit feature dashboard
- [ ] Days 19-30: Model development
- [ ] Days 31-42: RAG + deployment
- [ ] Days 43-50: Monitoring + polish

## Author

**Mounika Katipally**  
Building an end-to-end AI/ML project in public to document the process, technical decisions, implementation steps, and lessons learned.