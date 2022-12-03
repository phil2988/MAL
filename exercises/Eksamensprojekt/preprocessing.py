import string
from urllib.parse import quote_plus
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


def getCardsAsDataFrameByPath(path):
    data = []
    with open(path, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            data.append(row)

    cards = []
    for i in data[1:]:
        card = {}
        j = 0
        for k in i:
            card.update({data[0][j]: k})
            j += 1
        cards.append(card)
    print("Done! Returning data as DataFrame\n")
    return pd.DataFrame(data=cards)


def getCardsAsDataFrame(basePath="labels"):
    """ Reads files from folder and returns units and labels

    Parameters
    ----------
    basePath: str
        Specifies where to look for labels. Default value: "labels"

    Returns
    -------
    tuple 
        a tuple containing units and labels, units being a DataFrame and labels being a string array

    Example
    -------
    >>> units, labels = getCardsAsDataFrame()
    
    """
    import os

    print("Converting data into card objects...")
    units = []
    labels = []
    i = 0
    headers = getHeaders()
    for files in os.listdir("labels"):
        fileType = files.split(".")
        if fileType[1] == "txt":
            with open(basePath + "/" + files, "r") as labelFile:
                labels.append(labelFile.readline())

        i += 1
        if fileType[1] == "csv":
            with open("labels/" + files, "r", newline="", encoding="utf-8") as csvFile:
                unit = {}
                _i = 0
                for attr in csv.reader(csvFile):
                    unit.update({headers[0][_i]: attr[0]})
                    _i += 1
                units.append(unit)
    print("Done! Returning data as DataFrame\n")
    return (pd.DataFrame(data=units)), (labels)


def removeNonUnits(cards):
    raise NotImplementedError("This function is no longer necessary")
    assert isinstance(cards, pd.DataFrame)
    print("Removing cards which are not a unit...")

    # check if collectible
    print(cards["type"])
    units = cards.drop(cards[cards["type"] != "Unit"].index)

    print("Done!\n")
    return units


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

    print("Done!\n")

    print("Converting string values to int values...")

    units["attack"] = units["attack"].astype(int)
    units["health"] = units["health"].astype(int)
    units["cost"] = units["cost"].astype(int)

    print("Done!\n")
    return units


def getTrainTestSplit(units, labels = None):
    from sklearn.model_selection import train_test_split

    if(labels == None):
        print("No labels were given! Generating fake labels...")
        labels = generateFakeLabels(len(units))
        print("Done!\n")

    print("Splitting X and y into train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(units, labels)
    print("Done! Returning split as numpy arrays!\n")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test
