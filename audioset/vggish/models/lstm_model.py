#!/usr/bin/env python
# coding: utf-8

# In[50]:




# In[51]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix


# In[52]:


# Load the dataset from the CSV file
df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/vggish/embeddings_28fev.csv')


# In[53]:


df


# In[54]:


def assign_age_group(age, age_groups):
    for group_name, age_range in age_groups.items():
        if age_range[0] <= age < age_range[1]:
            return group_name
    return 'Unknown'  # For any age that doesn't fit the defined groups

# Define your age groups
age_groups = {
    'kitten': (0, 1),
    'adult': (1, 12),
    'senior': (12, 19)
}

# Apply the function to create a new column for the age group
df['age_group'] = df['target'].apply(assign_age_group, age_groups=age_groups)


# In[55]:


# Calculating sequence lengths and including targets
sequence_summary = df.groupby('cat_id').apply(lambda x: {"Length": len(x), "Targets": x['age_group'].unique()})

# Printing the summary
for cat_id, info in sequence_summary.items():
    print(f'Cat ID: {cat_id}, Sequence Length: {info["Length"]}, Targets: {info["Targets"]}')


# In[56]:


# Separate features and labels
X = df.iloc[:, :-2]  # Excludes 'target' and 'cat_id'
y = df['age_group']  # Target labels

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

max_sequence_length = 5

new_sequences = []
new_labels = []

for _, group in df.groupby('cat_id'):
    sequence = scaler.transform(group.iloc[:, :-2])
    label = group['age_group'].iloc[0]
    # Split sequences longer than max_sequence_length into smaller chunks
    for start in range(0, len(sequence), max_sequence_length):
        end = min(start + max_sequence_length, len(sequence))
        chunk = sequence[start:end]
        new_sequences.append(chunk)
        new_labels.append(label)

# Pad the sequences
X_padded = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

# Pad sequences to have the same length
max_sequence_length = max(len(sequence) for sequence in new_sequences)
X_padded = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')


# In[57]:


# Assuming `new_labels` holds the label for each sequence after chunking and padding
# Assuming the initial steps are correct and `encoder` has been defined
if isinstance(new_labels[0], str):
    y_encoded = encoder.transform(new_labels)  # This step assumes 'new_labels' matches 'new_sequences' one-to-one
else:
    y_encoded = np.array(new_labels)  # If already encoded, ensure this matches 'new_sequences'

y_one_hot = to_categorical(y_encoded)

# Convert encoded labels back to original labels for counting
final_labels = encoder.inverse_transform(y_encoded)

# Count the number of sequences per label
sequence_counts_per_label = {}
for label in final_labels:
    if label in sequence_counts_per_label:
        sequence_counts_per_label[label] += 1
    else:
        sequence_counts_per_label[label] = 1

# Print the counts for verification
print("Number of Sequences per Label:")
for label, count in sequence_counts_per_label.items():
    print(f"{label}: {count}")


# In[60]:


# Splitting the dataset into training and test sets
# y_encoded is used for stratification because train_test_split requires the labels in a
# format that indicates the class of each sample (i.e., not one-hot encoded).
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_one_hot, test_size=0.2, stratify=y_encoded, random_state=42)


# In[61]:


# Building the LSTM model with dropout and regularization
model = Sequential([
    Masking(mask_value=0., input_shape=(max_sequence_length, X_padded.shape[2])),
    LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # Add dropout
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Add dropout
    Dense(y_one_hot.shape[1], activation='softmax', kernel_regularizer='l2')  # Add L2 regularization
])

# Use a learning rate scheduler
lr_schedule = ReduceLROnPlateau(factor=0.1, patience=5)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# Training the model with callbacks
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[lr_schedule, early_stopping])

# Evaluating the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, Test precision: {precision}, Test recall: {recall}')


# In[45]:


# Evaluating the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, Test precision: {precision}, Test recall: {recall}')


# In[47]:


# Predict classes with the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate and print classification report
print(classification_report(y_true_classes, y_pred_classes))

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print(conf_matrix)


# In[ ]:




