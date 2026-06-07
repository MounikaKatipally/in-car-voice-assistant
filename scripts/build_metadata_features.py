from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_pseudonymized_metadata_1000.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'features' / 'metadata_features_1000.csv'

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'metadata_feature_engineering.log'

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def duration_bucket(duration):
    if duration < 3:
        return 'short'
    if duration <= 7:
        return 'medium'
    return 'long'


def synthetic_time_of_day(index):
    options = ['morning', 'afternoon', 'evening', 'night']
    return options[index % len(options)]


def main():
    logging.info('Starting metadata feature engineering.')

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Input file not found: {INPUT_PATH}')

    df = pd.read_csv(INPUT_PATH)

    records = []

    for index, row in df.iterrows():
        duration_seconds = float(row.get('duration_seconds', 0))
        word_count = int(row.get('word_count', 0))

        word_rate = round(word_count / duration_seconds, 3) if duration_seconds > 0 else 0

        records.append({
            'record_id': row['audio_id'],
            'audio_id': row['audio_id'],
            'speaker_hash': row.get('speaker_hash', ''),
            'duration_seconds': duration_seconds,
            'duration_bucket': duration_bucket(duration_seconds),
            'synthetic_time_of_day': synthetic_time_of_day(index),
            'speaker_gender': 'unknown',
            'sample_rate': int(row.get('sample_rate', 16000)),
            'word_count': word_count,
            'word_rate': word_rate,
            'has_transcript': bool(str(row.get('cleaned_transcript', '')).strip())
        })

    output_df = pd.DataFrame(records)
    output_df.to_csv(OUTPUT_PATH, index=False)

    logging.info(f'Metadata features saved to: {OUTPUT_PATH}')
    logging.info(f'Total records: {len(output_df)}')

    print('Metadata feature engineering complete.')
    print(f'Total records: {len(output_df)}')
    print(f'Output file: {OUTPUT_PATH}')
    print(output_df.head())


if __name__ == '__main__':
    main()
