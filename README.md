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

## Status

## Status

**Day 12 of 50** — Data governance and pseudonymization layer added.

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
- [ ] Days 13-18: Feature engineering
- [ ] Days 19-30: Model development
- [ ] Days 31-42: RAG + deployment
- [ ] Days 43-50: Monitoring + polish

## Author

**Mounika Katipally**  
Building an end-to-end AI/ML project in public to document the process, technical decisions, implementation steps, and lessons learned.