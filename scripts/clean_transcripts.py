from pathlib import Path
import logging
import string

import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_processed_1000.csv'
)

OUTPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_cleaned_transcripts_1000.csv'
)

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'transcript_cleaning.log'

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s | %(message)s'))
logging.getLogger('').addHandler(console)

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def clean_transcript(text: str) -> str:
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    doc = nlp(text)
    tokens = [token.text.strip() for token in doc if token.text.strip()]

    cleaned_text = ' '.join(tokens)
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text


def estimate_word_timestamps(cleaned_text: str, duration_seconds: float):
    words = cleaned_text.split()

    if not words or duration_seconds <= 0:
        return []

    avg_word_duration = duration_seconds / len(words)

    timestamps = []

    for index, word in enumerate(words):
        start_time = round(index * avg_word_duration, 3)
        end_time = round((index + 1) * avg_word_duration, 3)

        timestamps.append({
            'word': word,
            'start_time': start_time,
            'end_time': end_time
        })

    return timestamps


def main():
    logging.info('Starting transcript cleaning pipeline.')

    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f'Input metadata file not found: {INPUT_METADATA_PATH}'
        )

    df = pd.read_csv(INPUT_METADATA_PATH)

    cleaned_records = []

    successful_records = 0
    failed_records = 0

    for _, row in df.iterrows():
        audio_id = row.get('audio_id', 'unknown')

        try:
            original_transcript = row.get('transcript', '')
            duration_seconds = float(row.get('duration_seconds', 0))

            cleaned_transcript = clean_transcript(original_transcript)
            word_count = len(cleaned_transcript.split())

            estimated_timestamps = estimate_word_timestamps(
                cleaned_transcript,
                duration_seconds
            )

            cleaned_records.append({
                'audio_id': audio_id,
                'processed_audio_path': row.get('processed_audio_path', ''),
                'original_transcript': original_transcript,
                'cleaned_transcript': cleaned_transcript,
                'word_count': word_count,
                'duration_seconds': duration_seconds,
                'sample_rate': row.get('sample_rate', 16000),
                'timestamp_alignment_method': 'estimated_even_word_spacing',
                'estimated_word_timestamps': estimated_timestamps
            })

            successful_records += 1

            if successful_records % 100 == 0:
                logging.info(
                    f'Cleaned {successful_records} transcripts so far.'
                )

        except Exception as error:
            failed_records += 1
            logging.error(
                f'Failed transcript cleaning for audio_id={audio_id}: {error}'
            )

    cleaned_df = pd.DataFrame(cleaned_records)

    cleaned_df.to_csv(
        OUTPUT_METADATA_PATH,
        index=False
    )

    logging.info('Transcript cleaning completed.')
    logging.info(f'Successful records: {successful_records}')
    logging.info(f'Failed records: {failed_records}')

    print()
    print('Transcript cleaning complete.')
    print(f'Successful records: {successful_records}')
    print(f'Failed records: {failed_records}')
    print(f'Output metadata: {OUTPUT_METADATA_PATH}')
    print()
    print(cleaned_df.head())


if __name__ == '__main__':
    main()
