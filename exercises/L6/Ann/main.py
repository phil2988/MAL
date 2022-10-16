import string
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
import csv

def getAllData():
    results = []
    with open("lor_data.csv", encoding="utf8") as csvfile:   
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    return results

def getData(data = None):
    if(data == None):
        results = []
        with open("lor_data.csv", encoding="utf8") as csvfile:   
            reader = csv.reader(csvfile) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        return results[1:]
    return data[1:]

def getHeaders(data=None):
    if(data == None):
        results = []
        with open("lor_data.csv", encoding="utf8") as csvfile:   
            reader = csv.reader(csvfile) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        return results[:1]
    return data[:1]

def headerToIndex(label: string, data = None):
    if(data == None):
        return getHeaders()[0].index(label) 
    return data[0].index(label)


###########################################################3

data = getAllData()
headers = getHeaders(data)
cardData = getData(data)

# fig, axs = plt.subplots(2)

# attack = np.zeros(31)
# health = np.zeros(31)

# for i in range(len(getData(cardData))):
#     if(cardData[i][headerToIndex("health", data)] != ''):
#         health[int(cardData[i][headerToIndex("health", data)])] += 1
#     if(cardData[i][headerToIndex("attack", data)] != ''):
#         attack[int(cardData[i][headerToIndex("attack", data)])] += 1

# x = []
# for i in range(len(attack)):
#     x.append(i)

# fig.suptitle("Overview of card health and attack")

# axs[0].bar(x[1:], health[1:])
# axs[0].set(xlabel="Mana cost", ylabel="Health")

# axs[1].bar(x[1:], attack[1:])
# axs[1].set(xlabel="Mana cost", ylabel="Attack")

# plt.tight_layout()
# plt.show()

attack = []
health = []
cost = []

for i in range(len(getData(cardData))):
    if(cardData[i][headerToIndex("health", data)] != '' and int(cardData[i][headerToIndex("health", data)]) < 30):
        health.append(int(cardData[i][headerToIndex("health", data)]))
        attack.append(int(cardData[i][headerToIndex("attack", data)]))

attack = np.array(attack)
health = np.array(health)

X_train, X_test, y_train, y_test = train_test_split(
    health, 
    attack)

mlp = MLPRegressor(hidden_layer_sizes=100, max_iter=2000, random_state=10)
mlp.fit([X_train], [y_train.ravel()])

mpl_pred = mlp.predict([X_train]).ravel()

for i in range(len(X_train)):
    plt.plot(y_train[i], mpl_pred[i], 'b.')

plt.suptitle("Card attack predicted from its health")
plt.xlabel("Health")
plt.ylabel("Attack")
plt.show()
