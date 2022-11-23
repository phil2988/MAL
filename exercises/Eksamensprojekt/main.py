from utils import *
import pandas as pd

rawData = getAllData()
cards = getCards(rawData)

minions = removeSpells(cards)

healthCostManaCards = [cards["cost"], cards["attack"], cards["health"]]

print()
