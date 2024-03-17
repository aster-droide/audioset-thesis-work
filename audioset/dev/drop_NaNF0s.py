import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/dev/peak_normalised_cat_meows_f0_data.csv')

# Drop rows with missing pitch values
df_clean = df.dropna(subset=['MeanF0'])

# Save the cleaned data to a new CSV file
clean_csv_path = 'mean_f0_target_clean_normalised.csv'
df_clean.to_csv(clean_csv_path, index=False)
