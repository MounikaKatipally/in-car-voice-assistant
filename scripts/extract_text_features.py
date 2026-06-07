from pathlib import Path
import json
import logging

import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'librispeech_cleaned_transcripts_1000.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'features' / 'text_features_1000.jsonl'

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_PATH = LOG_DIR / 'text_feature_extraction.log'

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

nlp = spacy.load('en_core_web_sm')


def extract_text_features(audio_id, text):
    if not isinstance(text, str):
        text = ''

    doc = nlp(text)

    tokens = []
    lemmas = []
    pos_tags = []

    for token in doc:
        if token.text.strip():
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)

    noun_count = sum(1 for tag in pos_tags if tag in ['NOUN', 'PROPN'])
    verb_count = sum(1 for tag in pos_tags if tag == 'VERB')
    adjective_count = sum(1 for tag in pos_tags if tag == 'ADJ')
    adverb_count = sum(1 for tag in pos_tags if tag == 'ADV')

    word_count = len(tokens)
    character_count = len(text)

    if word_count > 0:
        avg_word_length = round(sum(len(token) for token in tokens) / word_count, 3)
    else:
        avg_word_length = 0

    return {
        'audio_id': audio_id,
        'cleaned_transcript': text,
        'tokens': tokens,
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'word_count': word_count,
        'character_count': character_count,
        'avg_word_length': avg_word_length,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adjective_count': adjective_count,
        'adverb_count': adverb_count
    }


def main():
    logging.info('Starting text feature extraction.')

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Input file not found: {INPUT_PATH}')

    df = pd.read_csv(INPUT_PATH)

    successful_records = 0
    failed_records = 0

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
        for _, row in df.iterrows():
            audio_id = row.get('audio_id', 'unknown')

            try:
                features = extract_text_features(
                    audio_id,
                    row.get('cleaned_transcript', '')
                )

                output_file.write(json.dumps(features) + '\n')
                successful_records += 1

                if successful_records % 100 == 0:
                    logging.info(f'Extracted text features for {successful_records} records.')

            except Exception as error:
                failed_records += 1
                logging.error(f'Failed text feature extraction for audio_id={audio_id}: {error}')

    logging.info('Text feature extraction completed.')
    logging.info(f'Successful records: {successful_records}')
    logging.info(f'Failed records: {failed_records}')

    print('Text feature extraction complete.')
    print(f'Successful records: {successful_records}')
    print(f'Failed records: {failed_records}')
    print(f'Output file: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
