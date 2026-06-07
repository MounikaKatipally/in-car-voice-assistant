from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = PROJECT_ROOT / 'data' / 'features' / 'multimodal_feature_store_1000.parquet'


def load_feature_store():
    assert FEATURE_STORE_PATH.exists(), f'Feature store not found: {FEATURE_STORE_PATH}'
    return pd.read_parquet(FEATURE_STORE_PATH)


def test_feature_store_shape():
    df = load_feature_store()
    assert df.shape[0] == 1000
    assert df.shape[1] > 10


def test_no_missing_core_values():
    df = load_feature_store()

    required_columns = [
        'audio_id',
        'duration_seconds',
        'sample_rate',
        'word_count',
        'mfcc_mean',
        'mfcc_std',
        'mel_mean',
        'mel_std',
        'text_word_count'
    ]

    for column in required_columns:
        assert column in df.columns
        assert df[column].notnull().all(), f'Missing values found in {column}'


def test_sample_rate_is_16khz():
    df = load_feature_store()
    assert (df['sample_rate'] == 16000).all()


def test_duration_range_is_valid():
    df = load_feature_store()
    assert (df['duration_seconds'] > 0).all()
    assert (df['duration_seconds'] < 60).all()


def test_word_counts_are_valid():
    df = load_feature_store()
    assert (df['word_count'] > 0).all()
    assert (df['text_word_count'] > 0).all()


def test_duration_bucket_values():
    df = load_feature_store()
    valid_buckets = {'short', 'medium', 'long'}
    assert set(df['duration_bucket'].unique()).issubset(valid_buckets)


def test_no_duplicate_audio_ids():
    df = load_feature_store()
    assert df['audio_id'].is_unique
