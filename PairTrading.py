import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Данные китайских компаний
data = pd.read_csv('stock_data.csv', parse_dates=['date'])

# Создание списка компаний с необходимым объёмом торгов (используем вторую треть)
# Требуется обрезать изначальную таблицу, оставить только нужные поля и "повернуть"
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
a = int(len(res) * 0.33) * 2

# Сам список нужных компаний
neededTickers = []
for i in range(a):
    neededTickers.append(list(res.items())[-i - 1][0])

# Создание списка компаний с ценами закрытия и удаление из него всех компаний, не попавших в список выше
# Требуется обрезать изначальную таблицу, оставить только нужные поля и "повернуть"
new_df = data[['ticker', 'date', 'close']].copy()
new_df.drop(new_df.loc[new_df['date'] < '2020-01-01'].index, inplace=True)
new_df.drop(new_df.loc[new_df['date'] > '2020-12-31'].index, inplace=True)
new_df = new_df.pivot_table(index=['date'], columns=['ticker'], values='close')
for i in new_df.keys():
    if i not in neededTickers:
        del new_df[i]

# Для быстрого тестирования
# new_df.drop(new_df.iloc[:, 500:], inplace=True, axis=1)


# Функция для нахождения разницы дат
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


# Функция для формирования пар тикеров
def kagi_pairs(data, T):
    n = data.shape[1]
    pairs = []
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            # signals - счётчик инверсий, считает сколько раз находили тау-бета
            signals = 0
            # V - поиск вертикальной суммы
            V = 0
            # Последовательности цен закрытия для двух выбранных компаний
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            # Спред, с которым мы работаем далее
            y = np.log(S1) - np.log(S2)
            # H - для построения H-процесса
            C = y.std()
            H = C
            try:
                y1 = y.tolist()
                k = 1
                # Сначала мы ищем тау-альфа-нулевое и тау-бета-нулевое
                while ((max(y1[:k]) - min(y1[:k])) < H) and k < T:
                    k += 1
                l = 0
                while (abs(y1[l] - y1[k]) > H) and l < k:
                    l += 1
                k = l + 1
                k1 = k
                l1 = 0
                # Заходим в цикл и единообразно ищем все последующие точки сигналов, пока не превысим время
                while k < T and l < T:
                    while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                        k += 1
                    while (abs(y1[l] - y1[k]) > H) and l < k:
                        l += 1
                    if y1[l] > 0 and y1[l1] > 0:
                        V += abs(y1[l] - y1[l1])
                    l1 = l
                    if k1 < k:
                        # Проверка необходима из-за наличая NaN полей в таблице. Заполнить её нулями/другими числами
                        # Не представляется возможным, это ведёт к неверным результатам и ошибкам
                        if S1[l] > 0 and S2[l] > 0:
                            signals += 1
                        # Сохраняем прошлый экстремум
                        k1 = k
                        l = k1
                    l += 1
            except:
                pass
            pairs.append((keys[i], keys[j], H, signals, V))
    return pairs


# Функция для торговли парами
def kagi_trade(data, pairs, T):
    # Изначально имеем нулевой счёт
    money = 0
    # Последовательность доходов в точках времени
    moneySpread = []
    for pair in pairs:
        try:
            # Изначально покупаем позицию на спот i и продаём позицию на спот j
            countS1 = 1
            countS2 = -1
            # Последовательности цен закрытия для двух выбранных компаний
            S1 = data[pair[0]]
            S2 = data[pair[1]]
            # H - для построения H-процесса
            H = pair[2]
            # Спред, с которым мы работаем далее
            y = np.log(S1) - np.log(S2)
            # signals - массив сигналов
            signals = []
            y1 = y.tolist()
            k = 1
            # Сначала мы ищем тау-альфа-нулевое и тау-бета-нулевое
            while ((max(y1[:k]) - min(y1[:k])) < H) and k < T:
                k += 1
            l = 0
            while (abs(y1[l] - y1[k]) > H) and l < k:
                l += 1
            # Определение типа первого экстремума
            maximin = (y1[l] - y1[k]) > 0
            k = l + 1
            k1 = k
            # Заходим в цикл и единообразно ищем все последующие точки сигналов, пока не превысим время
            while k < T and l < T:
                while ((max(y1[l:k]) - min(y1[l:k])) < H) and k < T:
                    k += 1
                while (abs(y1[l] - y1[k]) > H) and l < k:
                    l += 1
                if k1 != k:
                    # Проверка необходима из-за наличая NaN полей в таблице. Заполнить её нулями/другими числами
                    # Не представляется возможным, это ведёт к неверным результатам и ошибкам
                    if S1[l] > 0 and S2[l] > 0:
                        # Закрываем все позиции
                        money += S1[l] * countS1 + S2[l] * countS2
                    countS1 = 0
                    countS2 = 0
                    # Сохраняем сигнал
                    signals.append((k1, l))
                    # Сохраняем сумму
                    moneySpread.append(money)
                    k1 = k
                    l = k1
                l += 1
                # Используем тип первого экстремума и правило чётности экстремумов
                if len(signals) % 2 == maximin:
                    countS1 -= 1
                    countS2 += 1
                else:
                    countS1 += 1
                    countS2 -= 1
            # Закрываем все позиции
            money += S1[l] * countS1 + S2[l] * countS2
        except:
            pass
    return money, moneySpread

# Ищем пары на обучающей выборке
pairs = kagi_pairs(new_df, days_between('2020-01-01', '2020-12-31'))
# Удаляем пары с нулевыми инверсиями
pairs = [x for x in pairs if x[3] != 0]
# Сортируем по инверсиям и волатильностям
pairs = sorted(pairs, key=lambda tup: (tup[3], tup[4] / tup[3]))
pairs.reverse()
# Избавляемся от повторений
found = []
i = 0
while i < len(pairs):
    if pairs[i][0] in found or pairs[i][1] in found:
        del pairs[i]
    else:
        found.append(pairs[i][0])
        found.append(pairs[i][1])
        i += 1
# Берём лишь 5 лучших пар
pairs1 = pairs[:5]

# Для тестовой выборки требуется обрезать изначальную таблицу, оставить только нужные поля и "повернуть"
data2 = data[['ticker', 'date', 'close']].copy()
data2.drop(data2.loc[data2['date'] < '2021-01-01'].index, inplace=True)
data2.drop(data2.loc[data2['date'] > '2021-12-31'].index, inplace=True)
data2 = data2.pivot_table(index=['date'], columns=['ticker'], values='close')

# Запускаем торговлю на тестовой выборке
money, moneySpreadTest = kagi_trade(data2, pairs1, days_between('2021-01-01', '2021-12-31'))
print("test money", money)
dailyTest = pd.DataFrame(moneySpreadTest, columns=['return'])
print("test Sharp ratio", dailyTest['return'].mean() / dailyTest['return'].std())
print("test Accumulated profitability to maximum drawdown", money / min(moneySpreadTest))

# Для финальной выборки требуется обрезать изначальную таблицу, оставить только нужные поля и "повернуть"
data3 = data[['ticker','date', 'close']].copy()
data3.drop(data3.loc[data3['date']<'2022-01-01'].index, inplace=True)
data3.drop(data3.loc[data3['date']>'2022-05-11'].index, inplace=True)
data3 = data3.pivot_table(index=['date'], columns=['ticker'], values='close')

# Запускаем торговлю на финальной выборке
money, moneySpreadFinal = kagi_trade(data3, pairs1, days_between('2022-01-01', '2022-05-11'))
print("final money", money)
dailyFinal = pd.DataFrame(moneySpreadFinal, columns=['return'])
print("final Sharp ratio", dailyFinal['return'].mean() / dailyFinal['return'].std())
print("final Accumulated profitability to maximum drawdown", money / min(moneySpreadFinal))

# Строим графики
fig, (ax1, ax2) = plt.subplots(nrows =1, ncols =2, figsize=(16,6))
ax1.plot(dailyTest)
ax2.plot(dailyFinal)
ax1.hlines(dailyTest.mean(), 0, len(dailyTest), linestyles='dashed', colors = 'r')
ax2.hlines(dailyFinal.mean(), 0, len(dailyFinal), linestyles='dashed', colors = 'r')
ax1.legend(['Test'])
ax2.legend(['Final'])
ax1.set_title('Test')
ax2.set_title('Final')
plt.show()
