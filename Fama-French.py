import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

df_14 = pd.read_csv('2014_Financial_Data.csv')
df_15 = pd.read_csv('2015_Financial_Data.csv')
df_16 = pd.read_csv('2016_Financial_Data.csv')
df_17 = pd.read_csv('2017_Financial_Data.csv')
df_18 = pd.read_csv('2018_Financial_Data.csv')
df_set = [df_14, df_15, df_16, df_17, df_18]

for i, df in enumerate(df_set):
    df.dropna(subset=['Market Cap'], inplace=True)
    df.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    df = df[df['Market Cap'] != 0]
    df_set[i] = df
key_set = set(df_set[0]['Ticker']).intersection(*[df['Ticker'] for df in df_set[1:]])

for i in range(len(df_set)):
    df_set[i] = df_set[i][df_set[i]['Ticker'].isin(key_set)]

fama_french_factors['Date'] = pd.to_datetime(fama_french_factors['Date'])
fama_french_factors.set_index('Date', inplace=True)

def get_portfolio_return(df_prev, df_curr, investment_weights):
    df_prev = df_prev[['Ticker', 'Market Cap']]
    df_curr = df_curr[['Ticker', 'Market Cap']]
    df_result = df_prev.merge(df_curr, how='left', on='Ticker', suffixes=('_prev', '_curr'))
    df_result['pct_change'] = (df_result['Market Cap_curr'] - df_result['Market Cap_prev']) / df_result['Market Cap_prev']
    pct_change = df_result['pct_change'].to_numpy()
    return np.dot(pct_change, investment_weights)

initial_investment = 100000
portfolio_values = [initial_investment]
investment_weights = np.ones((len(key_set))) / len(key_set)
portfolio_returns = []

for i in range(0, len(df_set) - 1):
    portfolio_return = get_portfolio_return(df_set[i], df_set[i + 1], investment_weights)
    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    portfolio_returns.append(portfolio_return)

portfolio_returns_df = pd.DataFrame({'Date': pd.date_range(start='2014-12-31', periods=len(portfolio_returns), freq='Y'),
                                     'Portfolio_Return': portfolio_returns})
portfolio_returns_df.set_index('Date', inplace=True)

regression_data = fama_french_factors.merge(portfolio_returns_df, how='inner', left_index=True, right_index=True)

y = regression_data['Portfolio_Return']
X = regression_data[['Mkt-RF', 'SMB', 'HML']]  # Market, Size, Value factors

additional_factors = [col for col in regression_data.columns if col not in ['Portfolio_Return', 'Mkt-RF', 'SMB', 'HML']]
if additional_factors:
    X = regression_data[['Mkt-RF', 'SMB', 'HML'] + additional_factors]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

total_performance = (portfolio_values[-1] - initial_investment) / initial_investment
plt.plot(portfolio_values)
plt.title(f'Portfolio Performance: {total_performance:.2%}')
plt.xlabel('Year')
plt.ylabel('Portfolio Value')
plt.show()