import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

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

def get_pct_change(df_prev, df_curr):
    df_prev = df_prev[['Ticker', 'Market Cap']]
    df_curr = df_curr[['Ticker', 'Market Cap']]
    df_result = df_prev.merge(df_curr, how='left', on='Ticker', suffixes=('_prev', '_curr'))
    df_result['pct_change'] = (df_result['Market Cap_curr'] - df_result['Market Cap_prev']) / df_result['Market Cap_prev']
    return df_result


initial_investment = 100000
portfolio_values = [initial_investment]
portfolio_value = initial_investment

for i in range(len(df_set) - 1):
    df_pct = get_pct_change(df_set[i], df_set[i + 1])
    pct_change = df_pct['pct_change'].fillna(0).to_numpy()
    investment_weights = np.ones(len(pct_change)) / len(pct_change)
    portfolio_value = np.dot(pct_change, investment_weights) * portfolio_value
    portfolio_values.append(portfolio_value)


portfolio_performance = (portfolio_value - initial_investment) / initial_investment
plt.plot(portfolio_values)
plt.title(f'Portfolio Performance: {portfolio_performance:.2f}')
plt.show()


df_final = pd.concat(df_set)
df_final = df_final[['Ticker', 'Market Cap', 'HQ']].dropna()
df_final['HQ'] = df_final['HQ'].astype(int)

X = df_final[['Market Cap']]
y = df_final['HQ']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Note
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.2f}")


lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.2f}")
