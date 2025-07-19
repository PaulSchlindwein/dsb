# Importing necessary libraries
import pandas as pd


df = pd.read_csv('ks-projects-201612.csv', encoding='latin-1')

# Display the first few rows of the dataframe
print(df.head())

# Display the last few rows of the dataframe
print(df.tail())

# Display the shape of the dataframe
print(df.shape)

# Display the column names to see what we're working with
print("\nOriginal column names:")
print(df.columns.tolist())

# Data cleaning function
def clean_dataframe(df):
    # Remove trailing spaces from column titles first
    df.columns = df.columns.str.strip()
    
    # Remove spaces from column titles
    df.columns = df.columns.str.replace(' ', '_')
    
    # Find the index of usd_pledged column (after spaces are replaced)
    usd_pledged_index = df.columns.get_loc('usd_pledged')
    
    # Keep only columns up to and including usd_pledged
    df = df.iloc[:, :usd_pledged_index + 1]
    
    return df

# Apply the cleaning function
df_cleaned = clean_dataframe(df)

# Display the cleaned dataframe info
print("\nCleaned dataframe shape:", df_cleaned.shape)
print("\nCleaned column names:")
print(df_cleaned.columns.tolist())

# Function to create campaign duration column
def add_campaign_duration(df):
    # Convert launched and deadline columns to datetime with error handling
    df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
    df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
    
    # Calculate campaign duration in days
    df['campaign_duration'] = (df['deadline'] - df['launched']).dt.days
    
    return df

# Apply the campaign duration function
df_cleaned = add_campaign_duration(df_cleaned)

# Display the updated dataframe info
print("\nUpdated dataframe shape:", df_cleaned.shape)
print("\nUpdated column names:")
print(df_cleaned.columns.tolist())