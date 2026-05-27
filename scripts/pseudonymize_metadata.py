from pathlib import Path
import hashlib
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_cleaned_transcripts_1000.csv'
)

OUTPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_pseudonymized_metadata_1000.csv'
)

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'pseudonymization.log'

HASH_SALT = 'in_car_voice_assistant_day12_salt'

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


def create_speaker_id(audio_id: str) -> str:
    parts = str(audio_id).split('_')

    if len(parts) >= 2:
        return f'speaker_{parts[-1][:2]}'

    return 'speaker_unknown'


def hash_identifier(identifier: str) -> str:
    raw_value = f'{HASH_SALT}:{identifier}'
    return hashlib.sha256(raw_value.encode('utf-8')).hexdigest()


def main():
    logging.info('Starting pseudonymization step.')

    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f'Input metadata file not found: {INPUT_METADATA_PATH}'
        )

    df = pd.read_csv(INPUT_METADATA_PATH)

    if 'speaker_id' not in df.columns:
        df['speaker_id'] = df['audio_id'].apply(create_speaker_id)

    df['speaker_hash'] = df['speaker_id'].apply(hash_identifier)

    columns_to_remove = ['speaker_id']

    for column in columns_to_remove:
        if column in df.columns:
            df = df.drop(columns=[column])

    df.to_csv(OUTPUT_METADATA_PATH, index=False)

    logging.info(f'Pseudonymized metadata saved to: {OUTPUT_METADATA_PATH}')
    logging.info(f'Total records processed: {len(df)}')

    print('Pseudonymization complete.')
    print(f'Total records processed: {len(df)}')
    print(f'Output file: {OUTPUT_METADATA_PATH}')
    print()
    print(df[['audio_id', 'speaker_hash']].head())


if __name__ == '__main__':
    main()
