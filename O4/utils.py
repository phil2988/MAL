import string

def getAllData():
    import csv
    results = []
    with open("lor_data.csv", encoding="utf8") as csvfile:   
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    return results

def getData(data = None):
    if(data == None):
        import csv
        results = []
        with open("lor_data.csv", encoding="utf8") as csvfile:   
            reader = csv.reader(csvfile) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        return results[1:]
    return data[1:]

def getAttackHealthCost():
    import csv
    data = []
    with open("lor_data.csv", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        #data.append(reader[:1])
        for row in reader:
            data.append(row)

def getHeaders(data=None):
    if(data == None):
        import csv
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

def getTestTrainSplit(splitSize = 0.7, features = ["attack", "health", "cost", ]):
    #from pandas import pd jeg har lige udkommenteret dette og erstattet det med "import pandas as pd"
    import pandas as pd
    cards = pd.DataFrame(getData()) #Jeg har lige erstattet Dataframe med DataFrame
    fixedCards = []
        
    labels = getRandomLabels(cards)
    return (cards[:int(len(cards)*splitSize)], labels[:int(len(labels)*splitSize)]), (cards[int(len(cards)*splitSize):], labels[int(len(labels)*splitSize):])

def getRandomLabels(X, labels = ["aggro", "control", "midrange"]):
    y = []
    for i in range(len(X)):
        from random import randrange
        y.append(labels[randrange(2)])
    return y


