import string
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

from modelgeneration import generateFakeLabels


def getAllData():
    results = []
    with open("lor_data.csv", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    return results

def getCardsAsDataFrame(data=None):
    if data == None:
        data = getAllData()
    cards = []
    for i in data[1:]:
        card = {}
        j = 0
        for k in i:
            card.update({data[0][j]: k})
            j += 1
        print()
        cards.append(card)
    return pd.DataFrame(data=cards)


def removeNonUnits(cards):
    assert isinstance(cards, pd.DataFrame)
    return cards.drop(cards[cards["type"] != "Unit"].index)


def getData(data=None):
    if data == None:
        results = []
        with open("lor_data.csv", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)  # change contents to floats
            for row in reader:  # each row is a list
                results.append(row)
        return results[1:]
    return data[1:]


def getHeaders(data=None):
    if data == None:
        results = []
        with open("lor_data.csv", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)  # change contents to floats
            for row in reader:  # each row is a list
                results.append(row)
        return results[:1]
    return data[:1]


def headerToIndex(label: string, data=None):
    if data == None:
        return getHeaders()[0].index(label)
    return data[0].index(label)

def onlyCostAttackAndHealth(units):
    print("Isolating cost, attack and health...")
    
    for i in units.columns:
        if i != "cost" and i != "attack" and i != "health":
            units = units.drop(i, axis=1)
    
    print("Done!")
    
    print("Converting string values to int values...")

    units["attack"] = units["attack"].astype(int)
    units["health"] = units["health"].astype(int)
    units["cost"] = units["cost"].astype(int)

    print("done")

    return units

def getTrainTestSplit_test(units):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        units, 
        generateFakeLabels(len(units))
    )

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test
