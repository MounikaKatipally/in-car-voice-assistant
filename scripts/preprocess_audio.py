from pathlib import Path
import logging

import librosa
import noisereduce as nr
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_sample_1000.csv'
)

OUTPUT_AUDIO_DIR = (
    PROJECT_ROOT / 'data' / 'processed' / 'librispeech_16k'
)

OUTPUT_METADATA_PATH = (
    PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_processed_1000.csv'
)

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'preprocessing.log'

TARGET_SAMPLE_RATE = 16000


OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
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


def preprocess_audio(input_path: Path, output_path: Path):
    audio, sample_rate = librosa.load(
        input_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    audio = librosa.util.normalize(audio)

    reduced_noise_audio = nr.reduce_noise(
        y=audio,
        sr=TARGET_SAMPLE_RATE
    )

    sf.write(
        output_path,
        reduced_noise_audio,
        TARGET_SAMPLE_RATE
    )

    duration_seconds = round(
        len(reduced_noise_audio) / TARGET_SAMPLE_RATE,
        3
    )

    return {
        'sample_rate': TARGET_SAMPLE_RATE,
        'duration_seconds': duration_seconds
    }


def main():
    logging.info('Starting audio preprocessing pipeline.')

    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f'Metadata file not found: {INPUT_METADATA_PATH}'
        )

    metadata_df = pd.read_csv(INPUT_METADATA_PATH)

    processed_records = []

    successful_files = 0
    failed_files = 0

    for _, row in metadata_df.iterrows():
        audio_id = row['audio_id']

        try:
            input_audio_path = PROJECT_ROOT / row['audio_path']

            output_file_name = f'{audio_id}.wav'
            output_audio_path = OUTPUT_AUDIO_DIR / output_file_name

            audio_metadata = preprocess_audio(
                input_audio_path,
                output_audio_path
            )

            processed_records.append({
                'audio_id': audio_id,
                'processed_audio_path': str(output_audio_path),
                'transcript': row['transcript'],
                'sample_rate': audio_metadata['sample_rate'],
                'duration_seconds': audio_metadata['duration_seconds']
            })

            successful_files += 1

            if successful_files % 100 == 0:
                logging.info(
                    f'Processed {successful_files} audio files so far.'
                )

        except Exception as error:
            failed_files += 1

            logging.error(
                f'Failed preprocessing for audio_id={audio_id}: {error}'
            )

    processed_df = pd.DataFrame(processed_records)

    processed_df.to_csv(
        OUTPUT_METADATA_PATH,
        index=False
    )

    logging.info('Audio preprocessing completed.')
    logging.info(f'Successful files: {successful_files}')
    logging.info(f'Failed files: {failed_files}')

    print()
    print('Preprocessing complete.')
    print(f'Successful files: {successful_files}')
    print(f'Failed files: {failed_files}')
    print(f'Processed metadata: {OUTPUT_METADATA_PATH}')


if __name__ == '__main__':
    main()
