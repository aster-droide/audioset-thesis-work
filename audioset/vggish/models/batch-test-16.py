import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import keras

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Verify that TensorFlow does not see the GPU (should print an empty list)
print("Available GPU devices: ", tf.config.list_physical_devices('GPU'))


# Set a fixed random seed for reproducibility
random.seed(42) # <- added
np.random.seed(42)
tf.random.set_seed(42)


# Load the dataset from the CSV file
dataframe = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/vggish/all_embeddings_11Mar.csv')

augmented_df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/vggish/Aug_embed_march_11.csv')

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
dataframe['age_group'] = dataframe['target'].apply(assign_age_group, age_groups=age_groups)

# Apply the function to create a new column for the age group
augmented_df['age_group'] = augmented_df['target'].apply(assign_age_group, age_groups=age_groups)


# Separate features and labels
X = dataframe.iloc[:, :-3].values  # all columns except the last three

# First, encode the 'age_group' column as integers using LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(dataframe['age_group'].values)

# Now use the encoded labels for splitting and one-hot encoding
y = encoded_y  # This will be used in the GroupKFold

# Initialize lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
conf_matrices = []
training_accuracies = []

# Number of folds
n_splits = 5

# GroupKFold gives you indices of the dataset to be used for training and validation in each fold
gkf = StratifiedGroupKFold(n_splits=n_splits)

# Convert your 'cat_id' column to numpy array to be used as groups array for GroupKFold
groups = dataframe['cat_id'].values

# Initialize model performance list
model_performances = []

fold = 0

# To store training times
training_times = []

for train_index, val_index in gkf.split(X, y, groups):
    fold += 1
    print(f"Fold {fold}")

    # Split the original dataframe into training and validation dataframes
    df_train, df_val = dataframe.iloc[train_index], dataframe.iloc[val_index]

    # Include augmented data in the training dataframe
    # Make sure augmented_df includes a 'cat_id' column to filter by train_cat_ids
    train_cat_ids = np.unique(groups[train_index])
    augmented_train_df = augmented_df[augmented_df['cat_id'].isin(train_cat_ids)]

    # Combine the original training data with the augmented data
    combined_train_df = pd.concat([df_train, augmented_train_df])

    # After combining the data, check the distribution
    training_age_group_counts = combined_train_df['age_group'].value_counts()
    validation_age_group_counts = df_val['age_group'].value_counts()

    print("Training set age group distribution:")
    print(training_age_group_counts)
    print("Validation set age group distribution:")
    print(validation_age_group_counts)

    # Separate features and labels again, now from the combined dataframe
    X_train = combined_train_df.iloc[:, :-3].values  # adjust as per your dataframe structure
    y_train = label_encoder.transform(combined_train_df['age_group'].values)  # Re-encode the combined age_group
    y_train_encoded = to_categorical(y_train)  # One-hot encode the labels

    # Preprocess the validation set (no augmented data included)
    X_val = df_val.iloc[:, :-3].values
    y_val = label_encoder.transform(df_val['age_group'].values)
    y_val_encoded = to_categorical(y_val)

    # Normalize features
    scaler = StandardScaler().fit(X_train)  # Fit on combined training data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = Sequential([
        Dense(480, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(224, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(y_train_encoded.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.legacy.Adamax(learning_rate=0.0056663), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define an EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,  # Minimum change considered as improvement
        patience=100,
        # Stop if no improvement in validation loss for 10 consecutive epochs # <- changed, 10 was very aggressive
        verbose=1,  # Print messages when stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
    )

    # Start timing
    start_time = time.time()

    with tf.device('/CPU:0'):
        # Train the model using the validation set
        history = model.fit(X_train_scaled, y_train_encoded, epochs=200, batch_size=16,
                            validation_data=(X_val_scaled, y_val_encoded), callbacks=[early_stopping])

    # End timing
    end_time = time.time()

    # Calculate elapsed time and store
    elapsed_time = end_time - start_time
    training_times.append(elapsed_time)

    print(f"Training for this fold completed in {elapsed_time:.2f} seconds.")

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val_encoded)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

    # Store the model's performance
    model_performances.append(val_accuracy)

    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)

    # Convert predictions to class labels
    y_val_pred_class = np.argmax(y_val_pred, axis=1)
    y_val_true_class = np.argmax(y_val_encoded, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_val_true_class, y_val_pred_class)
    precision = precision_score(y_val_true_class, y_val_pred_class, average='macro')
    recall = recall_score(y_val_true_class, y_val_pred_class, average='macro')
    f1 = f1_score(y_val_true_class, y_val_pred_class, average='macro')

    # Get the training accuracy from the last epoch
    train_accuracy = history.history['accuracy'][-1]
    training_accuracies.append(train_accuracy)

    # Calculate confusion matrix
    cm = confusion_matrix(y_val_true_class, y_val_pred_class)
    conf_matrices.append(cm)

    # Append metrics to lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Print fold results
    print(f"Fold {fold + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(f"Fold {fold + 1} - Confusion Matrix:\n{cm}")

# Calculate the average performance across all folds
average_performance = sum(model_performances) / n_splits
print(f"Average Validation Accuracy across all folds: {average_performance}")

# Calculate and print the average and standard deviation of the metrics
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
average_precision = np.mean(precisions)
std_precision = np.std(precisions)
average_recall = np.mean(recalls)
std_recall = np.std(recalls)
average_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Average Accuracy: {average_accuracy} (±{std_accuracy})")
print(f"Average Precision: {average_precision} (±{std_precision})")
print(f"Average Recall: {average_recall} (±{std_recall})")
print(f"Average F1-Score: {average_f1} (±{std_f1})")

# Calculate and print the average training accuracy after all folds
average_train_accuracy = np.mean(training_accuracies)
print(f"Average Training Accuracy across all folds: {average_train_accuracy}")

# Optional: Aggregate and print the total confusion matrix
total_cm = np.sum(conf_matrices, axis=0)
print(f"Total Confusion Matrix:\n{total_cm}")
