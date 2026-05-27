# Data Governance

This document describes the data handling, privacy, and governance practices used in the In-Car Voice Assistant project.

## Purpose

The goal of this project is to build an end-to-end conversational AI system while applying responsible data handling practices from the beginning.

Although this is a personal learning project, the system is designed with production-style principles such as traceability, access control, pseudonymization, reproducibility, and secure handling of data.

## Dataset

The current dataset is based on a 1,000-clip LibriSpeech sample.

The project stores:

- Audio file paths
- Ground-truth transcripts
- Cleaned transcripts
- Sample rate
- Audio duration
- Dataset source
- Dataset split
- Pseudonymized speaker identifiers

Large raw and processed audio files are stored locally and excluded from GitHub.

## Data Storage

Raw audio files are stored locally in:

`	ext
data/raw/librispeech_sample/
