from pathlib import Path
import logging

import pandas as pd
import psycopg2
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_sample_1000.csv'
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'ingestion.log'

DB_CONFIG = {
    'dbname': 'voice_assistant_db',
    'user': 'voice_user',
    'password': 'voice_password',
    'host': 'localhost',
    'port': 5432,
}


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


def get_audio_metadata(audio_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    info = sf.info(str(audio_path))
    duration_seconds = round(info.frames / info.samplerate, 3)

    return {
        'sample_rate': info.samplerate,
        'duration_seconds': duration_seconds,
    }


def connect_to_postgres():
    return psycopg2.connect(**DB_CONFIG)


def insert_audio_file(cursor, record, audio_metadata):
    cursor.execute(
        '''
        INSERT INTO audio_files (
            audio_id,
            audio_path,
            source,
            split,
            sample_rate,
            duration_seconds
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (audio_id)
        DO UPDATE SET
            audio_path = EXCLUDED.audio_path,
            source = EXCLUDED.source,
            split = EXCLUDED.split,
            sample_rate = EXCLUDED.sample_rate,
            duration_seconds = EXCLUDED.duration_seconds;
        ''',
        (
            record['audio_id'],
            record['audio_path'],
            record['source'],
            record['split'],
            audio_metadata['sample_rate'],
            audio_metadata['duration_seconds'],
        )
    )


def transcript_exists(cursor, audio_id):
    cursor.execute(
        '''
        SELECT COUNT(*)
        FROM transcripts
        WHERE audio_id = %s
          AND transcript_source = 'ground_truth';
        ''',
        (audio_id,)
    )
    return cursor.fetchone()[0] > 0


def insert_transcript(cursor, record):
    if transcript_exists(cursor, record['audio_id']):
        return

    cursor.execute(
        '''
        INSERT INTO transcripts (
            audio_id,
            transcript_text,
            transcript_source
        )
        VALUES (%s, %s, %s);
        ''',
        (
            record['audio_id'],
            record['transcript'],
            'ground_truth',
        )
    )


def main():
    logging.info('Starting LibriSpeech ingestion pipeline.')

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f'Metadata CSV not found: {METADATA_PATH}')

    df = pd.read_csv(METADATA_PATH)
    logging.info(f'Loaded metadata CSV with {len(df)} records.')

    successful_records = 0
    failed_records = 0

    conn = connect_to_postgres()

    try:
        with conn:
            with conn.cursor() as cursor:
                for _, record in df.iterrows():
                    audio_id = record.get('audio_id', 'unknown')

                    try:
                        audio_path = PROJECT_ROOT / record['audio_path']
                        audio_metadata = get_audio_metadata(audio_path)

                        insert_audio_file(cursor, record, audio_metadata)
                        insert_transcript(cursor, record)

                        successful_records += 1

                        if successful_records % 100 == 0:
                            logging.info(f'Ingested {successful_records} records so far.')

                    except Exception as error:
                        failed_records += 1
                        logging.error(f'Failed to ingest audio_id={audio_id}: {error}')

        logging.info('Ingestion pipeline completed.')
        logging.info(f'Successful records: {successful_records}')
        logging.info(f'Failed records: {failed_records}')

    finally:
        conn.close()


if __name__ == '__main__':
    main()
