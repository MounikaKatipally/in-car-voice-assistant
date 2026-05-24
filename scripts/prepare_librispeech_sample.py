from datasets import load_dataset, Audio
from pathlib import Path
from itertools import islice
import pandas as pd
import soundfile as sf
import io

OUTPUT_AUDIO_DIR = Path('data/raw/librispeech_sample')
OUTPUT_METADATA_PATH = Path('data/metadata/librispeech_sample_1000.csv')
SAMPLE_SIZE = 1000

OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

print('Loading LibriSpeech validation split in streaming mode...')

dataset = load_dataset(
    'openslr/librispeech_asr',
    'clean',
    split='validation',
    streaming=True
)

dataset = dataset.cast_column('audio', Audio(decode=False))

records = []

print(f'Preparing {SAMPLE_SIZE} audio clips...')

for idx, item in enumerate(islice(dataset, SAMPLE_SIZE)):
    audio = item['audio']
    transcript = item['text']

    audio_bytes = audio.get('bytes')

    if audio_bytes is None:
        audio_path = audio.get('path')
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

    file_name = f'librispeech_{idx:04d}.wav'
    file_path = OUTPUT_AUDIO_DIR / file_name

    sf.write(file_path, audio_array, sample_rate)

    records.append({
        'audio_id': f'librispeech_{idx:04d}',
        'audio_path': str(file_path).replace('\\\\', '/'),
        'transcript': transcript,
        'sample_rate': sample_rate,
        'source': 'LibriSpeech dev-clean',
        'split': 'validation'
    })

metadata = pd.DataFrame(records)
metadata.to_csv(OUTPUT_METADATA_PATH, index=False)

print('Done.')
print(f'Audio folder: {OUTPUT_AUDIO_DIR}')
print(f'Metadata file: {OUTPUT_METADATA_PATH}')
print(metadata.head())
