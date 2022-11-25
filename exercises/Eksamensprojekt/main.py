from preprocessing import *
import pandas as pd

rawData = getAllData()
cards = getCards(rawData)

# Remove entries not being a unit
units = removeSpells(cards)

# only keep health, attack and cost
for i in units.columns:
    if i != "cost" and i != "attack" and i != "health":
        units = units.drop(i, axis=1)

print(units)
