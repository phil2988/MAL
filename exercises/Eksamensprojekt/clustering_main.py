import clustering_model_generation
import clustering_model_training
from preprocessing import *
import numpy

for i in range(0, 99, 1):
    units, labels = getCardsAsDataFrame("sigurdslabels_done")

    units = onlyCostAttackAndHealth(units) #Prøv md alle i stedet?

    X_train, X_test, y_train, y_test = getTrainTestSplit(units, labels)

    model = clustering_model_generation.CreateSpectralClustering()

    print(model)

    model = clustering_model_training.TrainSpectralClustering(model, X_train, y_train)

    labled_test = model.labels_
    named = []
    counter = 0

    UnitsToUse = X_train#units[:435]
    A_n = 0
    T_n = 0
    C_n = 0
    save_these_cards = []
    for statline in UnitsToUse:#.iterrows():
        #print(statline)
        currentLabel = int(labled_test[counter])
        #print(outputEnumNumberConvert(currentLabel))
        if currentLabel == 0:
            save_these_cards.append("Control: ")
            save_these_cards.append(statline)
            C_n += 1
        elif currentLabel == 2:
            T_n += 1
            save_these_cards.append("Tempo: ")
            save_these_cards.append(statline)
        elif currentLabel == 1:
            save_these_cards.append("Aggro: ")
            save_these_cards.append(statline)
            A_n += 1
        counter += 1

    newlist = save_these_cards[:30]
    if C_n > 100 and A_n > 100 and T_n > 100:
        for card in newlist:
            print()
            print(card)
        print("Total Control: ", C_n)
        print("Total Tempo: ", T_n)
        print("Total Aggro: ", A_n)
    #print(save_these_cards)

    #model = outputEnumNumberConvert(model)
    #print(model)
    #print(X_test)


