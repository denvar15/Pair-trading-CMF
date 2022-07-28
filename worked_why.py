import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import seaborn
import matplotlib.pyplot as plt
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session.

data = pd.read_csv('stock_data.csv', parse_dates=['date'])
data0 = data.copy()
data0.drop(data0.loc[data0['date']<'2011-01-01'].index, inplace=True)
data0.drop(data0.loc[data0['date']>'2020-12-31'].index, inplace=True)
data0 = data0.pivot_table(index=['date'], columns=['ticker'], values='volume')
new_df = data[['ticker','date', 'close']].copy()
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
#print(res.items())
a = int(len(res) * 0.33)
#print(a)
neededTickers = []
for i in range(a):
    neededTickers.append(list(res.items())[-i - 1][0])
print(neededTickers)

new_df = data[['ticker','date', 'close']].copy()
new_df.drop(new_df.loc[new_df['date']<'2011-10-01'].index, inplace=True)
new_df.drop(new_df.loc[new_df['date']>'2012-12-31'].index, inplace=True)
#new_df.drop(new_df.loc[new_df['ticker'] not in neededTickers].index, inplace=True)
new_df = new_df.pivot_table(index=['date'], columns=['ticker'], values='close')
for i in new_df.keys():
    if i not in neededTickers:
        del new_df[i]
#data1 = new_df.copy()
#new_df.drop(new_df.iloc[:, 100:], inplace=True, axis=1)
#print(data1)

#print(new_df)

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def kagi_pairs(data):
    n = data.shape[1]
    pairs = []
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            try:
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                y = np.log(S1) - np.log(S2)
                C = y.std()
                H = C
                pairs.append((keys[i], keys[j], H))
            except:
                pass
    return pairs

def kagi_trade(data, pairs, T):
    money = 0
    moneySpread = []
    for pair in pairs:
        try:
            #print("pair ", pair[0], pair[1])
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
            #print("maximin", maximin)
            k = l + 1
            k1 = k
            while k < T and l < T:
                while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                    k += 1
                while (abs(y1[l] - y1[k]) < H) and l < k:
                    l += 1
                if k1 != k:
                    #print("m ", S1[l] * countS1 + S2[l] * countS2)
                    money += S1[l] * countS1 + S2[l] * countS2
                    countS1 = 0
                    countS2 = 0
                    signals.append((k, l))
                    moneySpread.append(money)
                    k1 = k
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

def kagi_find_pairs(data, T):
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    money = 0
    moneySpread = []
    for i in range(n):
        for j in range(i+1, n):
            try:
                countS1 = 1
                countS2 = -1
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                y = np.log(S1) - np.log(S2)
                C = y.std()
                H = C
                V = 0
                signals = []
                #print("H ", H)
                y1 = y.tolist()
                k = 1
                #print((max(y1[:k]), min(y1[:k])), H)
                while ((max(y1[:k]) - min(y1[:k])) < H) and k < T:
                    #print("HWRHWEHREWH ")
                    k += 1
                #print("K ", k)
                l = 0
                while (abs(y1[l] - y1[k]) < H) and l < k:
                    #print("QWEQWEQWEWQ  ")
                    l += 1
                #print("L ", l)
                #signals.append((k, l))
                #print("Фух")
                k = l + 1
                k1 = k
                l1 = l
                while k < T and l < T:
                    #print("HWRHWEHREWH ", k, l, T)
                    while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                        k += 1
                        #print("Mmm ", k < T)
                    while (abs(y1[l] - y1[k]) < H) and l < k:
                        # print("QWEQWEQWEWQ  ")
                        l += 1
                        #print("Sss ", l < k)
                    V += abs(y1[l] - y1[l1])
                    l1 = l
                    if k1 != k:
                        #print(int(S1[l] * countS1 + S2[l] * countS2))
                        money += int(S1[l] * countS1 + S2[l] * countS2)
                        moneySpread.append(money)
                        signals.append((k, l))
                        k1 = k
                    l += 1
                    if len(signals) % 2 == 0:
                        countS1 -= 1
                        countS2 += 1
                    else:
                        countS1 += 1
                        countS2 -= 1
                pairs.append((keys[i], keys[j], H, signals, V))
            except:
                pass
    return pairs, money, moneySpread

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            try:
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                if (S1 == 0).all():
                    #print("AAAA1")
                    pass
                else:
                    if (S2 == 0).all():
                        #print("AAAA2")
                        pass
                    else:
                        result = coint(S1, S2)
                        score = result[0]
                        pvalue = result[1]
                        score_matrix[i, j] = score
                        pvalue_matrix[i, j] = pvalue
                        if pvalue < 0.05:
                            #print(pvalue)
                            pairs.append((keys[i], keys[j]))
            except:
                pass
    return score_matrix, pvalue_matrix, pairs

def trade(S1, S2, window1, window2):
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1,
                         center=False).mean()
    ma2 = ratios.rolling(window=window2,
                         center=False).mean()
    std = ratios.rolling(window=window2,
                         center=False).std()
    zscore = (ma1 - ma2) / std

    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    temp = []
    for i in range(len(ratios)):
        if money != 0:
            temp.append(money)
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            # print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            # print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            # print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))

    return money, temp

#scores, pvalues, pairs = find_cointegrated_pairs(df)
#print(pairs)

#scores1, pvalues1, pairs1 = find_cointegrated_pairs(new_df.head())
pairs = kagi_pairs(new_df)
pairs = sorted(pairs, key=lambda tup: tup[2])
pairs.reverse()
found = []
i = 0
while i < len(pairs):
    #print("FFF ", pairs[i][0], pairs[i][1], pairs[i][0] in found, pairs[i][1] in found)
    if pairs[i][0] in found or pairs[i][1] in found:
        del pairs[i]
    else:
        found.append(pairs[i][0])
        found.append(pairs[i][1])
        i += 1
pairs1 = pairs[:20]
print(pairs1)

data2 = data[['ticker','date', 'close']].copy()
data2.drop(data2.loc[data2['date']<'2021-01-01'].index, inplace=True)
data2.drop(data2.loc[data2['date']>'2021-12-31'].index, inplace=True)
data2 = data2.pivot_table(index=['date'], columns=['ticker'], values='close')

#print(days_between('2021-01-01', '2021-12-31'))
money, moneySpread = kagi_trade(data2, pairs1, days_between('2021-01-01', '2021-12-31'))
print("test money", money)
#print("moneySpread", moneySpread)
daily = pd.DataFrame(moneySpread, columns=['return'])
print("test Sharp ratio", daily['return'].mean() / daily['return'].std())


data3 = data[['ticker','date', 'close']].copy()
data3.drop(data3.loc[data3['date']<'2022-01-01'].index, inplace=True)
data3.drop(data3.loc[data3['date']>'2022-05-11'].index, inplace=True)
data3 = data3.pivot_table(index=['date'], columns=['ticker'], values='close')

#print(days_between('2022-01-01', '2022-05-11'))
money, moneySpread = kagi_trade(data3, pairs1, days_between('2022-01-01', '2022-05-11'))
print("final money", money)
#print("moneySpread", moneySpread)
daily = pd.DataFrame(moneySpread, columns=['return'])
print("final Sharp ratio", daily['return'].mean() / daily['return'].std())

#pairs, money, moneySpread = kagi_find_pairs(new_df, 60)
#print(pairs[2])
#pairs = sorted(pairs, key=lambda tup: tup[2])
#pairs2 = sorted(pairs, key=lambda tup: len(tup[3]))
#print(len(pairs))
#pairs1 = pairs[-20:]
#pairs2 = pairs2[-20:]
#print(pairs1)
#print(pairs2)


'''
data2 = data[['ticker','date', 'close']].copy()
data2.drop(data2.loc[data2['date']<'2021-01-01'].index, inplace=True)
data2.drop(data2.loc[data2['date']>'2021-12-31'].index, inplace=True)
data2 = data2.pivot_table(index=['date'], columns=['ticker'], values='close').fillna(0)

cum = 0
tempCum = []
for i in pairs1:
    a, temp = trade(data2[i[0]], data2[i[1]], 60, 5)
    if tempCum != []:
        temp.extend([0, ] * (len(tempCum) - len(temp)))
        tempCum = list(map(sum, zip(tempCum, temp)))
        #tempCum += temp
    else:
        tempCum = temp
        #temp.extend([0, ] * (len(data2[i[0]]) - len(temp)))
    cum += a
#print(tempCum)
daily = pd.DataFrame(tempCum, columns=['return'])
print("Sharp ratio", daily['return'].mean()/daily['return'].std())
print("test return", cum)


data3 = data[['ticker','date', 'close']].copy()
data3.drop(data3.loc[data3['date']<'2022-01-01'].index, inplace=True)
data3.drop(data3.loc[data3['date']>'2022-05-11'].index, inplace=True)
data3 = data3.pivot_table(index=['date'], columns=['ticker'], values='close').fillna(0)

cum = 0
tempCum = []
for i in pairs1:
    a, temp = trade(data3[i[0]], data3[i[1]], 60, 5)
    if tempCum != []:
        temp.extend([0, ] * (len(tempCum) - len(temp)))
        tempCum = list(map(sum, zip(tempCum, temp)))
        #tempCum += temp
    else:
        tempCum = temp
        #temp.extend([0, ] * (len(data3[i[0]]) - len(temp)))
    cum += a
#print(tempCum)
daily = pd.DataFrame(tempCum, columns=['return'])
print("Sharp ratio", daily['return'].mean()/daily['return'].std())
print("final return", cum)



#a = trade(df['ADBE'].iloc[881:], df['EBAY'].iloc[881:], 60, 5)
#print(a)
'''