import string
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
import csv
import pandas as pd


def getAllData():
    results = []
    with open("lor_data.csv", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    return results


def getCards(data=None):
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
