import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# wikipediaの閲覧数を使って、予測する

from datetime import datetime,timedelta

df=pd.read_csv('data/pageviews-20240818-20240907-soccer-player.csv')
print(df)

df['cap'] = 8.5
plt.title('Number of views: wikipedia for a soccer player')
left = df['Date']
height = df['伊東純也']
plt.bar(left, height)
plt.xticks(rotation=45)
plt.show()

df['cap'] = 8.5
m = Prophet(growth='logistic')

df['ds'] = df['Date']
df['y'] = df['伊東純也'].apply(np.log)
print(df)
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
fig = m.plot(fcst)
plt.show()
plt.savefig("fcst")

df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)
plt.show()
plt.savefig("fcst")