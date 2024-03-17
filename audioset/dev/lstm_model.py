import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset from the CSV file
df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/vggish/embeddings_28fev.csv')

# Calculating sequence lengths and including targets
sequence_summary = df.groupby('cat_id').apply(lambda x: {"Length": len(x), "Targets": x['target'].unique()})

# Printing the summary
for cat_id, info in sequence_summary.items():
    print(f'Cat ID: {cat_id}, Sequence Length: {info["Length"]}, Targets: {info["Targets"]}')

