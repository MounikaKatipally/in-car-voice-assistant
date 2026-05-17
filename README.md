# In-Car Voice Assistant

End-to-end conversational AI system that transcribes speech, identifies user intent, and answers context-aware queries — modeled after Cerence AI's in-car voice assistant platform.

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

## Project Timeline

50-day independent build, documented daily on LinkedIn and GitHub.

## Status

**Day 1 of 50** — Project charter complete. Environment set up.

## Roadmap

- [x] Day 1: Repo + environment
- [ ] Days 2–3: Architecture diagram + dependencies
- [ ] Days 4–12: Data engineering pipeline
- [ ] Days 13–18: Feature engineering
- [ ] Days 19–30: Model development
- [ ] Days 31–42: RAG + deployment
- [ ] Days 43–50: Monitoring + polish

## Author

Mounika Katipally — building in public.
