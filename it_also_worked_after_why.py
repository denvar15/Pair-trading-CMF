import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from datetime import datetime
import numpy as np

data = pd.read_csv('stock_data.csv', parse_dates=['date'])
data0 = data.copy()
data0.drop(data0.loc[data0['date'] < '2020-01-01'].index, inplace=True)
data0.drop(data0.loc[data0['date'] > '2020-12-31'].index, inplace=True)
data0 = data0.pivot_table(index=['date'], columns=['ticker'], values='volume')
new_df = data[['ticker', 'date', 'close']].copy()
data1 = new_df.copy()

n = data0.shape[1]
keys = data0.keys()
res = {}
for i in keys:
    res[i] = 0
for i in range(n):
    try:
        S1 = data1[keys[i]]
        res[keys[i]] += sum(S1)
    except:
        pass
res = dict(sorted(res.items(), key=lambda item: item[1]))
a = int(len(res) * 0.33)
neededTickers = []
for i in range(a):
    neededTickers.append(list(res.items())[-i - 1][0])

new_df = data[['ticker', 'date', 'close']].copy()
new_df.drop(new_df.loc[new_df['date'] < '2020-01-01'].index, inplace=True)
new_df.drop(new_df.loc[new_df['date'] > '2020-12-31'].index, inplace=True)
new_df = new_df.pivot_table(index=['date'], columns=['ticker'], values='close')
for i in new_df.keys():
    if i not in neededTickers:
        del new_df[i]
#new_df.drop(new_df.iloc[:, 500:], inplace=True, axis=1)


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def kagi_pairs(data, T):
    n = data.shape[1]
    pairs = []
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            signals = 0
            V = 0
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            y = np.log(S1) - np.log(S2)
            C = y.std()
            H = C
            try:
                y1 = y.tolist()
                k = 1
                while ((max(y1[:k]) - min(y1[:k])) < H) and k < T:
                    k += 1
                l = 0
                while (abs(y1[l] - y1[k]) < H) and l < k:
                    l += 1
                k = l + 1
                k1 = k
                l1 = l
                while k < T and l < T:
                    while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                        k += 1
                    while (abs(y1[l] - y1[k]) > H) and l < k:
                        l += 1
                    if y1[l] > 0 and y1[l1] > 0:
                        V += abs(y1[l] - y1[l1])
                    l1 = l
                    if k1 < k:
                        if S1[l] > 0 and S2[l] > 0:
                            signals += 1
                        k1 = k
                        l = k1
                    l += 1
            except:
                pass
            pairs.append((keys[i], keys[j], H, signals, V))
    return pairs


def kagi_trade(data, pairs, T):
    money = 0
    moneySpread = []
    for pair in pairs:
        try:
            countS1 = 1
            countS2 = -1
            S1 = data[pair[0]]
            S2 = data[pair[1]]
            H = pair[2]
            y = np.log(S1) - np.log(S2)
            signals = []
            y1 = y.tolist()
            k = 1
            while ((max(y1[:k]) - min(y1[:k])) < H) and k < T:
                k += 1
            l = 0
            while (abs(y1[l] - y1[k]) < H) and l < k:
                l += 1
            maximin = (y1[l] - y1[k]) > 0
            k = l + 1
            k1 = k
            while k < T and l < T:
                while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                    k += 1
                while (abs(y1[l] - y1[k]) > H) and l < k:
                    l += 1
                if k1 != k:
                    if S1[l] > 0 and S2[l] > 0:
                        money += S1[l] * countS1 + S2[l] * countS2
                    countS1 = 0
                    countS2 = 0
                    signals.append((k1, l))
                    moneySpread.append(money)
                    k1 = k
                    l = k1
                l += 1
                if len(signals) % 2 != maximin:
                    countS1 -= 1
                    countS2 += 1
                else:
                    countS1 += 1
                    countS2 -= 1
        except:
            pass
    return money, moneySpread


pairs = kagi_pairs(new_df, days_between('2020-01-01', '2020-12-31'))
pairs = [x for x in pairs if x[3] != 0]
pairs = sorted(pairs, key=lambda tup: (tup[3], tup[4] / tup[3]))
pairs.reverse()
found = []
i = 0
while i < len(pairs):
    if pairs[i][0] in found or pairs[i][1] in found:
        del pairs[i]
    else:
        found.append(pairs[i][0])
        found.append(pairs[i][1])
        i += 1
pairs1 = pairs[:20]
print(pairs1)

data2 = data[['ticker', 'date', 'close']].copy()
data2.drop(data2.loc[data2['date'] < '2021-01-01'].index, inplace=True)
data2.drop(data2.loc[data2['date'] > '2021-12-31'].index, inplace=True)
data2 = data2.pivot_table(index=['date'], columns=['ticker'], values='close')

money, moneySpread = kagi_trade(data2, pairs1, days_between('2021-01-01', '2021-12-31'))
print("test money", money)
daily = pd.DataFrame(moneySpread, columns=['return'])
print("test Sharp ratio", daily['return'].mean() / daily['return'].std())


data3 = data[['ticker','date', 'close']].copy()
data3.drop(data3.loc[data3['date'] < '2022-01-01'].index, inplace=True)
data3.drop(data3.loc[data3['date'] > '2022-05-11'].index, inplace=True)
data3 = data3.pivot_table(index=['date'], columns=['ticker'], values='close')

money, moneySpread = kagi_trade(data3, pairs1, days_between('2022-01-01', '2022-05-11'))
print("final money", money)
daily = pd.DataFrame(moneySpread, columns=['return'])
print("final Sharp ratio", daily['return'].mean() / daily['return'].std())
