import pandas as pd

# Read the Excel file into a pandas DataFrame
df = pd.read_excel('input.xlsx')

# Sort the DataFrame by 'Text ID' column
df_sorted = df.sort_values('Text ID')

# Merge 'Text' values together based on 'Text ID'
df_merged = df_sorted.groupby('Text ID')['Text'].apply(' '.join).reset_index()

# Add the merged text as a new column
df_merged.columns = ['Text ID', 'Merged Text']

# Write the merged DataFrame to a new Excel file
df_merged.to_excel('output.xlsx', index=False)
