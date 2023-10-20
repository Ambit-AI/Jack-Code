import pandas as pd

# Read the Excel file into a pandas DataFrame
df = pd.read_excel('input.xlsx')

# Sort the DataFrame by 'Text ID' column
df_sorted = df.sort_values('Text ID')

# Write the sorted DataFrame to a new Excel file
df_sorted.to_excel('output.xlsx', index=False)
