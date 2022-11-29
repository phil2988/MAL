from utils import *

# Load dataen fra utils på samme måde som Phillip (kun cost-attack-health)
# Sæt træningssæt op
# Opret alle modellerne
# -Mean shift
# -KMeans
# -Hierarchical Clustering
# -BIRCH
# -Spectral Clustering
# Kør parameter search for at tjekke nogle forskellige parametre
# -Sørg for at der KUN er 3 kategorier der passer til "aggro", "tempo", "control"
# Træn modellerne
# Plot dataen

tmp = getData()
print("Loading data...")
(X_train, y_train), (X_test, y_test) = getTestTrainSplit()
print("Done!")

print(tmp)