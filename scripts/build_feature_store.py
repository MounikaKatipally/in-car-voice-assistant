from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_MANIFEST_PATH = PROJECT_ROOT / 'data' / 'features' / 'audio_feature_manifest_1000.csv'
TEXT_FEATURES_PATH = PROJECT_ROOT / 'data' / 'features' / 'text_features_1000.jsonl'
METADATA_FEATURES_PATH = PROJECT_ROOT / 'data' / 'features' / 'metadata_features_1000.csv'

OUTPUT_PATH = PROJECT_ROOT / 'data' / 'features' / 'multimodal_feature_store_1000.parquet'

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'feature_store.log'

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


def summarize_array(relative_path):
    full_path = PROJECT_ROOT / relative_path

    if not full_path.exists():
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }

    array = np.load(full_path)

    if array.size == 0:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0
        }

    return {
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array))
    }


def load_text_features():
    records = []

    with open(TEXT_FEATURES_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)

            records.append({
                'audio_id': item['audio_id'],
                'text_word_count': item['word_count'],
                'text_character_count': item['character_count'],
                'text_avg_word_length': item['avg_word_length'],
                'text_noun_count': item['noun_count'],
                'text_verb_count': item['verb_count'],
                'text_adjective_count': item['adjective_count'],
                'text_adverb_count': item['adverb_count']
            })

    return pd.DataFrame(records)


def main():
    logging.info('Building multimodal feature store.')

    audio_df = pd.read_csv(AUDIO_MANIFEST_PATH)
    text_df = load_text_features()
    metadata_df = pd.read_csv(METADATA_FEATURES_PATH)

    audio_summary_records = []

    for _, row in audio_df.iterrows():
        mfcc_stats = summarize_array(row['mfcc_path'])
        mel_stats = summarize_array(row['mel_spectrogram_path'])
        pitch_stats = summarize_array(row['pitch_path'])
        tempo_stats = summarize_array(row['tempo_path'])

        audio_summary_records.append({
            'audio_id': row['audio_id'],
            'mfcc_path': row['mfcc_path'],
            'mel_spectrogram_path': row['mel_spectrogram_path'],
            'pitch_path': row['pitch_path'],
            'tempo_path': row['tempo_path'],
            'audio_duration_seconds': row['duration_seconds'],
            'audio_sample_rate': row['sample_rate'],
            'mfcc_mean': mfcc_stats['mean'],
            'mfcc_std': mfcc_stats['std'],
            'mel_mean': mel_stats['mean'],
            'mel_std': mel_stats['std'],
            'pitch_mean': pitch_stats['mean'],
            'pitch_std': pitch_stats['std'],
            'tempo_mean': tempo_stats['mean']
        })

    audio_summary_df = pd.DataFrame(audio_summary_records)

    feature_store = metadata_df.merge(
        audio_summary_df,
        on='audio_id',
        how='inner'
    ).merge(
        text_df,
        on='audio_id',
        how='inner'
    )

    feature_store = feature_store.set_index('record_id')

    feature_store.to_parquet(OUTPUT_PATH)

    logging.info(f'Feature store saved to: {OUTPUT_PATH}')
    logging.info(f'Feature store shape: {feature_store.shape}')

    print('Multimodal feature store created.')
    print(f'Output file: {OUTPUT_PATH}')
    print(f'Shape: {feature_store.shape}')
    print(feature_store.head())


if __name__ == '__main__':
    main()
