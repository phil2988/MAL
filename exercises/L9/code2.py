from code1 import *
# Setup data
X_train, X_test, y_train, y_test = LoadAndSetupData(
    'iris')  # 'iris', 'moon', or 'mnist'

# Setup search parameters
model = SGDClassifier()  

tuning_parameters = {
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha" : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
    "max_iter": [50000, 20000, 10000, 5000, 1000], 
    "shuffle": [True, False],
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "l1_ratio": [0.1, 0.2, 0.5, 0.8, 1],
    "epsilon": [0, 1, 2, 10, 50, 100, 200, 1000]
}

CV = 5
VERBOSE = 1

# Run GridSearchCV for the model
# grid_tuned = GridSearchCV(model,
#                           tuning_parameters,
#                           cv=CV,
#                           scoring='f1_micro',
#                           verbose=VERBOSE,
#                           n_jobs=-1)

# start = time()
# grid_tuned.fit(X_train, y_train)
# t = time() - start

# # Report result
# b0, m0 = FullReport(grid_tuned, X_test, y_test, t)


# Run RandomSearchCV for the model

random_tuned = RandomizedSearchCV(
    model, 
    tuning_parameters, 
    n_iter=20, 
    random_state=42, 
    cv=CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1
)

start = time()
random_tuned.fit(X_train, y_train)
t = time() - start

# Report result
b0, m0 = FullReport(random_tuned, X_test, y_test, t)

print('OK(grid-search)')

# model = SGDClassifier()  

# model = SGDClassifier(alpha=0.005,epsilon=200,l1_ratio=0.1,learning_rate='optimal',max_iter=10000,penalty='l1',shuffle=True)

# model.fit(X_train, y_train)

# prediction = model.predict(X_test)

# print("score: ", model.score(X_test, y_test))
