import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('avocado.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())

# No NAN or null values in this dataset
df.drop_duplicates(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)

df['price_change'] = df['AveragePrice'].pct_change()
df['volume_change'] = df['Total Volume'].pct_change()

#Implement Rolling mean and standard deviation
df['price_moving_avg'] = df['AveragePrice'].rolling(window=5).mean()
df['volume_moving_avg'] = df['Total Volume'].rolling(window=5).mean()

#Use Quantiles to define states

df['price_state'] = pd.qcut(df['price_change'], q=3, labels=['Low', 'Medium', 'High'])
df['volume_state'] = pd.qcut(df['volume_change'], q=3, labels=['Low', 'Medium', 'High'])

#encode price and volume states into numerical vals

# Low = 0, Med = 1, High = 2
le_price = LabelEncoder()
df['price_state_encoded'] = le_price.fit_transform(df['price_state'])

le_volume = LabelEncoder()
df['volume_state_encoded'] = le_volume.fit_transform(df['volume_state'])

#obs for HMM training 
df.drop(columns=['price_state', 'volume_state'], inplace=True)
observations = df[['price_state_encoded', 'volume_state_encoded']].values

df.to_csv("processed_avocado.csv", index=False)
