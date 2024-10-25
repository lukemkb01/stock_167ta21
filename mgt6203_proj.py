import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df_14 = pd.read_csv('2014_Financial_Data.csv')
df_15 = pd.read_csv('2015_Financial_Data.csv')
df_16 = pd.read_csv('2016_Financial_Data.csv')
df_17 = pd.read_csv('2017_Financial_Data.csv')
df_18 = pd.read_csv('2018_Financial_Data.csv')

df_set = [df_14, df_15, df_16, df_17, df_18]

for i, df in enumerate(df_set):
    df_set[i].dropna(subset=['Market Cap'], inplace=True)
    df_set[i].rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    df_set[i] = df_set[i][df_set[i]['Market Cap'] != 0]
    print('RSLS', df_set[i][df_set[i]['Ticker'] == 'RSLS'][['Market Cap', 'Enterprise Value']])


key_set = set(df_set[0]['Ticker']).intersection(*[df['Ticker'] for df in df_set[1:]])
print(len(key_set))

for i in range(len(df_set)):
    df_set[i] = df_set[i][df_set[i]['Ticker'].isin(key_set)]
    print('length', len(df))

def get_pct_change(df_prev, df_curr):
    df_prev = df_prev[['Ticker', 'Market Cap']]
    df_curr = df_curr[['Ticker', 'Market Cap']]
    df_result = df_prev.merge(df_curr, how='left', on='Ticker', suffixes=('_prev', '_curr'))
    df_result['pct_change'] = (df_result['Market Cap_curr'] - df_result['Market Cap_prev']) / df_result['Market Cap_prev']

    return df_result

inital_investment = 100000
portfolio_values = [inital_investment]
portfolio_value = inital_investment
for i in range(0, 3):
    df_pct = get_pct_change(df_set[i], df_set[i+1])
    pct_change = df_pct['pct_change'].to_numpy()
    investment_weights = np.ones((len(key_set))) / len(key_set)
    # print('sanity check', np.isnan(pct_change).any(), np.isnan(investment_weights).any())
    portfolio_value = np.dot(pct_change, investment_weights) * portfolio_value
    portfolio_values.append(portfolio_value)
    print(portfolio_value)

portfolio_performance = (portfolio_value - inital_investment) / inital_investment
plt.plot(portfolio_values)
plt.title(f'Portfolio Performance: {portfolio_performance}')
plt.show()





