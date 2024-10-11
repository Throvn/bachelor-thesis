import torch
import pandas as pd

def find_closest_entry(df, target_datetime):
    """Find the closest entry in a DataFrame based on the datetime column."""
    if df.empty or 'datetime' not in df.columns:
        return -1
    closest_row = df.iloc[(df['datetime'] - target_datetime).abs().argmin()]
    return closest_row['value'] if not closest_row.empty else -1

def find_exact_entry(entries, target_datetime):
    """Find the entry in a list of entries that matches the target_datetime exactly."""
    if not entries:
        return None
    for entry in entries:
        if entry['datetime'] == target_datetime:
            return entry['value']
    return None

def flatten_data(data):
    entr = {
        'slug': data['slug'],
        'observations': []
    }

    # Convert priceUSD data to a pandas DataFrame
    price_df = pd.DataFrame(data.get('priceUSD', []))
    if 'datetime' in price_df.columns:
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])
    # else:
        # return # Skip if no datetime exists in price data

    # Convert twitterFollowers and devActivity data to DataFrames, or handle missing fields
    twitter_df = pd.DataFrame(data.get('twitterFollowers', []))
    if 'datetime' in twitter_df.columns:
        twitter_df['datetime'] = pd.to_datetime(twitter_df['datetime'])
    
    dev_df = pd.DataFrame(data.get('devActivity', []))
    if 'datetime' in dev_df.columns:
        dev_df['datetime'] = pd.to_datetime(dev_df['datetime'])

    # Iterate through each price entry
    for i, row in price_df.iterrows():
        datetime_num = row['datetime']
        
        # Check if `twitterFollowers` exists and find the closest entry
        twitter_value = find_closest_entry(twitter_df, datetime_num) if not twitter_df.empty else -1
        
        # Check if `devActivity` exists and find the closest entry
        dev_activity_value = find_closest_entry(dev_df, datetime_num) if not dev_df.empty else -1

        # Check if `isActive` exists, and get its value or None if missing
        is_active_value = data['isActive'][i] if 'isActive' in data and i < len(data['isActive']) else -1

        # Create an observation entry
        entry = {
            'datetime': datetime_num.timestamp(),
            'priceUSD': row['value'],
            'twitterFollowers': twitter_value,
            'devActivity': dev_activity_value,
            'isActive': is_active_value
        }

        # Append the observation to the slug's observations list
        entr['observations'].append(entry)

    return entr

# Example usage
data = [{
    "slug": "3space-art",
    "priceUSD": [
        {"datetime": "2023-12-12T00:00:00Z", "value": 0.048865316079},
        {"datetime": "2024-08-09T00:00:00Z", "value": 0.235388073394}
    ],
    "twitterFollowers": [
        {"datetime": "2023-12-12T00:00:00Z", "value": 500},  # Exact match
        {"datetime": "2024-08-08T00:00:00Z", "value": 520}   # No match
    ],
    "devActivity": [
        {"datetime": "2023-12-12T00:00:00Z", "value": 3},    # Exact match
        {"datetime": "2024-08-10T00:00:00Z", "value": 5}     # No match
    ],
    "isActive": [1, 1]
}]

# flattened = flatten_data(data)
# print(flattened)