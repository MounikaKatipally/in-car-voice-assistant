from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = PROJECT_ROOT / 'data' / 'features' / 'multimodal_feature_store_1000.parquet'


st.set_page_config(
    page_title='In-Car Voice Assistant Feature Dashboard',
    layout='wide'
)

st.title('In-Car Voice Assistant Feature Dashboard')
st.write('Day 18: Feature distribution dashboard for the multimodal feature store.')


@st.cache_data
def load_data():
    return pd.read_parquet(FEATURE_STORE_PATH)


df = load_data()

st.subheader('Feature Store Overview')

col1, col2, col3, col4 = st.columns(4)

col1.metric('Records', df.shape[0])
col2.metric('Features', df.shape[1])
col3.metric('Avg Duration', round(df['duration_seconds'].mean(), 2))
col4.metric('Avg Word Count', round(df['word_count'].mean(), 2))

st.subheader('Dataset Preview')
st.dataframe(df.head(20))

st.subheader('Duration Distribution')
st.bar_chart(df['duration_bucket'].value_counts())

st.subheader('Duration Seconds')
st.line_chart(df['duration_seconds'])

st.subheader('Word Count Distribution')
st.line_chart(df['word_count'])

st.subheader('MFCC Mean Distribution')
st.line_chart(df['mfcc_mean'])

st.subheader('Pitch Mean Distribution')
st.line_chart(df['pitch_mean'])

st.subheader('Feature Correlation Preview')

numeric_columns = [
    'duration_seconds',
    'word_count',
    'word_rate',
    'mfcc_mean',
    'mfcc_std',
    'mel_mean',
    'mel_std',
    'pitch_mean',
    'pitch_std',
    'tempo_mean',
    'text_word_count',
    'text_character_count',
    'text_avg_word_length'
]

available_numeric_columns = [
    column for column in numeric_columns if column in df.columns
]

correlation_df = df[available_numeric_columns].corr()

st.dataframe(correlation_df)

st.subheader('Filter by Duration Bucket')

selected_bucket = st.selectbox(
    'Choose duration bucket',
    options=['all'] + sorted(df['duration_bucket'].unique().tolist())
)

if selected_bucket != 'all':
    filtered_df = df[df['duration_bucket'] == selected_bucket]
else:
    filtered_df = df

st.write(f'Showing {len(filtered_df)} records')
st.dataframe(filtered_df.head(50))
