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
## Project Timeline

50-day independent build, documented daily on LinkedIn and GitHub.

## Status

## Status

**Day 5 of 50** — PostgreSQL database schema added.

## Roadmap

## Roadmap

- [x] Day 1: Repo + environment setup
- [x] Day 2: Architecture diagram + README architecture link
- [x] Day 3: Dependencies + Docker skeleton
- [x] Day 4: Public dataset preparation with LibriSpeech
- [x] Day 5: PostgreSQL database schema with Docker Compose
- [ ] Days 6–12: Data engineering pipeline
- [ ] Days 13–18: Feature engineering
- [ ] Days 19–30: Model development
- [ ] Days 31–42: RAG + deployment
- [ ] Days 43–50: Monitoring + polish

## Author

**Mounika Katipally**  
Building an end-to-end AI/ML project in public to document the process, technical decisions, implementation steps, and lessons learned.