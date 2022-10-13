import string
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
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

fig, axs = plt.subplots(2)

attack = np.zeros(31)
health = np.zeros(31)

for i in range(len(getData(cardData))):
    if(cardData[i][headerToIndex("health", data)] != ''):
        health[int(cardData[i][headerToIndex("health", data)])] += 1
    if(cardData[i][headerToIndex("attack", data)] != ''):
        attack[int(cardData[i][headerToIndex("attack", data)])] += 1

x = []
for i in range(len(attack)):
    x.append(i)

fig.suptitle("Overview of card health and attack")

axs[0].bar(x[1:], health[1:])
axs[0].set(xlabel="Mana cost", ylabel="Health")

axs[1].bar(x[1:], attack[1:])
axs[1].set(xlabel="Mana cost", ylabel="Attack")

plt.tight_layout()
plt.show()