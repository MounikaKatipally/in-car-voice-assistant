from pathlib import Path
import logging

import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_processed_1000.csv'
)

OUTPUT_FEATURE_DIR = (
    PROJECT_ROOT / 'data' / 'features' / 'audio_npy'
)

OUTPUT_MANIFEST_PATH = (
    PROJECT_ROOT / 'data' / 'features' / 'audio_feature_manifest_1000.csv'
)

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'audio_feature_extraction.log'

TARGET_SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 128


OUTPUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
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


def extract_features(audio_path: Path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    duration_seconds = round(len(audio) / sample_rate, 3)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MELS
    )

    mel_spectrogram_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )

    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sample_rate
    )

    pitch_values = []

    for frame_index in range(pitches.shape[1]):
        magnitude_column = magnitudes[:, frame_index]

        if magnitude_column.max() > 0:
            pitch_index = magnitude_column.argmax()
            pitch_value = pitches[pitch_index, frame_index]

            if pitch_value > 0:
                pitch_values.append(pitch_value)

    pitch_values = np.array(pitch_values, dtype=np.float32)

    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate
    )

    tempo = librosa.beat.tempo(
        onset_envelope=onset_envelope,
        sr=sample_rate
    )

    tempo = np.array(tempo, dtype=np.float32)

    return {
        'mfcc': mfcc.astype(np.float32),
        'mel_spectrogram': mel_spectrogram_db.astype(np.float32),
        'pitch': pitch_values,
        'tempo': tempo,
        'duration_seconds': duration_seconds,
        'sample_rate': sample_rate
    }


def save_feature_array(audio_id: str, feature_name: str, array: np.ndarray):
    output_path = OUTPUT_FEATURE_DIR / f'{audio_id}_{feature_name}.npy'
    np.save(output_path, array)
    return output_path


def main():
    logging.info('Starting audio feature extraction pipeline.')

    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f'Input metadata file not found: {INPUT_METADATA_PATH}'
        )

    metadata_df = pd.read_csv(INPUT_METADATA_PATH)

    manifest_records = []
    successful_files = 0
    failed_files = 0

    for _, row in metadata_df.iterrows():
        audio_id = row['audio_id']

        try:
            audio_path = Path(row['processed_audio_path'])

            if not audio_path.is_absolute():
                audio_path = PROJECT_ROOT / audio_path

            if not audio_path.exists():
                raise FileNotFoundError(f'Audio file not found: {audio_path}')

            features = extract_features(audio_path)

            mfcc_path = save_feature_array(
                audio_id,
                'mfcc',
                features['mfcc']
            )

            mel_path = save_feature_array(
                audio_id,
                'mel_spectrogram',
                features['mel_spectrogram']
            )

            pitch_path = save_feature_array(
                audio_id,
                'pitch',
                features['pitch']
            )

            tempo_path = save_feature_array(
                audio_id,
                'tempo',
                features['tempo']
            )

            manifest_records.append({
                'audio_id': audio_id,
                'mfcc_path': str(mfcc_path.relative_to(PROJECT_ROOT)),
                'mel_spectrogram_path': str(mel_path.relative_to(PROJECT_ROOT)),
                'pitch_path': str(pitch_path.relative_to(PROJECT_ROOT)),
                'tempo_path': str(tempo_path.relative_to(PROJECT_ROOT)),
                'sample_rate': features['sample_rate'],
                'duration_seconds': features['duration_seconds'],
                'mfcc_shape': str(features['mfcc'].shape),
                'mel_spectrogram_shape': str(features['mel_spectrogram'].shape),
                'pitch_shape': str(features['pitch'].shape),
                'tempo_shape': str(features['tempo'].shape),
                'feature_format': 'npy'
            })

            successful_files += 1

            if successful_files % 100 == 0:
                logging.info(
                    f'Extracted audio features for {successful_files} files.'
                )

        except Exception as error:
            failed_files += 1
            logging.error(
                f'Failed feature extraction for audio_id={audio_id}: {error}'
            )

    manifest_df = pd.DataFrame(manifest_records)

    manifest_df.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False
    )

    logging.info('Audio feature extraction completed.')
    logging.info(f'Successful files: {successful_files}')
    logging.info(f'Failed files: {failed_files}')
    logging.info(f'Manifest path: {OUTPUT_MANIFEST_PATH}')

    print()
    print('Audio feature extraction complete.')
    print(f'Successful files: {successful_files}')
    print(f'Failed files: {failed_files}')
    print(f'Feature manifest: {OUTPUT_MANIFEST_PATH}')
    print()
    print(manifest_df.head())


if __name__ == '__main__':
    main()
