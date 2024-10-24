import pandas as pd
import numpy as np

def create_sequences(observation: pd.Series, window_size: int):
    X_seq, y_seq = [], []

    # Convert each subcategory of observation into a DataFrame, indexed by datetime
    price_df = pd.DataFrame(observation['priceUSD']).rename(columns={'value': 'priceUSD'})
    dev_df = pd.DataFrame(observation['devActivity']).rename(columns={'value': 'devActivity'})
    twitter_df = pd.DataFrame(observation['twitterFollowers']).rename(columns={'value': 'twitterFollowers'})
    
    # Handle empty DataFrames by creating a 'datetime' column if necessary
    if price_df.empty or 'datetime' not in price_df.columns:
        print("Price data is missing or empty.")
        return np.array([]), np.array([])

    # Ensure 'datetime' column exists in dev_df and twitter_df, or else add it as NaN for alignment
    if dev_df.empty:
        dev_df = pd.DataFrame({'datetime': price_df['datetime'], 'devActivity': [np.nan] * len(price_df)})
    if twitter_df.empty:
        twitter_df = pd.DataFrame({'datetime': price_df['datetime'], 'twitterFollowers': [np.nan] * len(price_df)})



    # print(dev_df)
    # Merge the dataframes on datetime
    df = price_df.merge(dev_df, on='datetime', how='inner') \
                 .merge(twitter_df, on='datetime', how='inner')

    # Sort by datetime to ensure proper sequence
    df = df.sort_values(by='datetime').reset_index(drop=True)

    isActive = observation['isActive']
    # Only activate if outer join.
    # # The isActive is already aligned with priceUSD, so no need to merge on datetime
    # isActive = pd.Series(observation['isActive'], index=price_df['datetime'])
    # # Reindex isActive to match the datetime index in df
    # isActive = isActive.reindex(df['datetime']).reset_index(drop=True)

    # Fill any missing values with interpolation for the features
    df[['priceUSD', 'devActivity', 'twitterFollowers']] = df[['priceUSD', 'devActivity', 'twitterFollowers']].interpolate(method='linear')

    # Ensure we have enough data points for the specified window size
    if len(df) < window_size:
        print(f"Insufficient data for window size {window_size}. Skipping.")
        return np.array([]), np.array([])

    # Iterate and create sequences
    # Limit to 3000 to not run out of mps space.
    for i in range(len(df) - window_size): # , 0, -1):
        # Extract the window of features
        X_window = df[['priceUSD', 'devActivity', 'twitterFollowers']].iloc[i:(i + window_size)].values
        
        # Make sure that 
        X_seq.append(X_window)

        # print("Lengths:", len(isActive), i + window_size - 1, len(df['priceUSD']))
        # Extract the target value for the end of the window from isActive array
        y_seq.append(isActive[i + window_size - 1])

    # Convert lists to numpy arrays
    X_seq = np.asarray(X_seq)
    y_seq = np.asarray(y_seq)

    return X_seq, y_seq