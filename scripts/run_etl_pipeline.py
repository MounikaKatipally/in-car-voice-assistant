from pathlib import Path
import logging
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'etl_pipeline.log'

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


PIPELINE_STEPS = [
    {
        'name': 'Prepare LibriSpeech sample',
        'script': 'scripts/prepare_librispeech_sample.py',
        'expected_output': 'data/metadata/librispeech_sample_1000.csv',
    },
    {
        'name': 'Preprocess audio',
        'script': 'scripts/preprocess_audio.py',
        'expected_output': 'data/metadata/librispeech_processed_1000.csv',
    },
    {
        'name': 'Clean transcripts',
        'script': 'scripts/clean_transcripts.py',
        'expected_output': 'data/metadata/librispeech_cleaned_transcripts_1000.csv',
    },
    {
        'name': 'Create intent annotations',
        'script': 'scripts/create_intent_annotations.py',
        'expected_output': 'data/metadata/intent_annotations_200.csv',
    },
    {
        'name': 'Ingest LibriSpeech data into PostgreSQL',
        'script': 'scripts/ingest_librispeech_to_postgres.py',
        'expected_output': None,
    },
]


def path_exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).exists()


def run_script(script_path: str) -> None:
    full_script_path = PROJECT_ROOT / script_path

    if not full_script_path.exists():
        raise FileNotFoundError(f'Script not found: {full_script_path}')

    command = [sys.executable, str(full_script_path)]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.stdout:
        logging.info(result.stdout)

    if result.stderr:
        logging.warning(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f'Script failed: {script_path} with return code {result.returncode}'
        )


def run_pipeline(force: bool = False) -> None:
    logging.info('Starting Day 11 ETL orchestration pipeline.')
    start_time = time.time()

    completed_steps = 0
    skipped_steps = 0
    failed_steps = 0

    for step in PIPELINE_STEPS:
        step_name = step['name']
        script_path = step['script']
        expected_output = step['expected_output']

        logging.info(f'Starting step: {step_name}')

        try:
            if expected_output and path_exists(expected_output) and not force:
                logging.info(
                    f'Skipping step because output already exists: {expected_output}'
                )
                skipped_steps += 1
                continue

            run_script(script_path)
            completed_steps += 1
            logging.info(f'Completed step: {step_name}')

        except Exception as error:
            failed_steps += 1
            logging.error(f'Failed step: {step_name}. Error: {error}')
            raise

    elapsed_seconds = round(time.time() - start_time, 2)

    logging.info('ETL orchestration pipeline finished.')
    logging.info(f'Completed steps: {completed_steps}')
    logging.info(f'Skipped steps: {skipped_steps}')
    logging.info(f'Failed steps: {failed_steps}')
    logging.info(f'Elapsed seconds: {elapsed_seconds}')

    print()
    print('ETL pipeline finished.')
    print(f'Completed steps: {completed_steps}')
    print(f'Skipped steps: {skipped_steps}')
    print(f'Failed steps: {failed_steps}')
    print(f'Elapsed seconds: {elapsed_seconds}')


if __name__ == '__main__':
    force_run = '--force' in sys.argv
    run_pipeline(force=force_run)
