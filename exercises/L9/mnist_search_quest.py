from code1 import *
from sklearn.gaussian_process import GaussianProcessClassifier
import scipy
from sklearn.gaussian_process.kernels import Kernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Setup data
X_train, X_test, y_train, y_test = LoadAndSetupData('mnist')

random_state = 42

loss_functions = [
    "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
]

alpha = []
for i in range(0, 50000, 1000):
    alpha.append(i/50000)

l1_ratio = []
for i in range(0, 50000, 1000):
    l1_ratio.append(i/50000)

tuning_parameters = {
    "n_estimators" : [100, 1000, 100],
    "verbose": [1],
    "criterion" : ["gini", "entropy", "log_loss"], 
    "min_samples_split": range(2, 50),
    "n_jobs": [-1]
}

CV = 5
VERBOSE = 1

clf = RandomForestClassifier(
    # random_state=42,
    # alpha=0.1,
    # n_jobs=-1,
    # learning_rate="optimal",
) 

clf_tuned = RandomizedSearchCV(
    clf, 
    tuning_parameters, 
    n_iter=10, 
    random_state=42, 
    cv=CV, 
    scoring='f1_micro', 
    verbose=VERBOSE, 
    n_jobs=-1
)

time_sgd_start = time()
clf_tuned.fit(X_train, y_train)
time_sgd = time() - time_sgd_start

b1, m1 = FullReport(clf_tuned , X_test, y_test, time_sgd)
print(b1)

# tuning_parameters = {
#     'C': scipy.stats.expon(scale=100), 
#     'gamma': scipy.stats.expon(scale=.1),
#     'class_weight':['balanced', None]
# }

# svm_clf = svm.SVC() 
# grid_tuned_svc_clf = RandomizedSearchCV(
#     svm_clf, 
#     tuning_parameters, 
#     n_iter=20, 
#     random_state=42, 
#     cv=CV, 
#     scoring='f1_micro', 
#     verbose=VERBOSE, 
#     n_jobs=-1
# )

# time_svm_start = time()
# grid_tuned_svc_clf.fit(X_train, y_train)
# time_svm = time() - time_svm_start

# b1, m1 = FullReport(grid_tuned_svc_clf , X_test, y_test, time_svm)
# print(b1)

# gp_clf_tuning_parameters = {
#     "n_restarts_optimizer": range(1, 10)
# }

# gp_clf = GaussianProcessClassifier() 
# grid_tuned_gp_clf = RandomizedSearchCV(
#     gp_clf, 
#     gp_clf_tuning_parameters, 
#     n_iter=20, 
#     random_state=42, 
#     cv=CV, 
#     scoring='f1_micro', 
#     verbose=VERBOSE, 
#     n_jobs=-1
# )

# time_gp_clf_start = time()
# grid_tuned_gp_clf.fit(X_train, y_train)
# time_gp_clf = time() - time_gp_clf_start

# b1, m1 = FullReport(grid_tuned_gp_clf , X_test, y_test, time_gp_clf)
# print(b1)

# dt_clf_tuning_parameters = {
#     "criterion": ["gini", "entropy", "log_loss"],
#     "splitter": ["best", "random"],
#     "min_samples_split": range(1, 10),
#     "min_samples_leaf": range(2, 10),
#     "max_features": ["sqrt", "log2", None],
# }

# dt_clf = DecisionTreeClassifier() 
# grid_tuned_dt_clf = RandomizedSearchCV(
#     dt_clf, 
#     dt_clf_tuning_parameters, 
#     n_iter=20, 
#     random_state=42, 
#     cv=CV, 
#     scoring='f1_micro', 
#     verbose=VERBOSE, 
#     n_jobs=-1
# )

# time_dt_clf_start = time()
# grid_tuned_dt_clf.fit(X_train, y_train)
# time_dt_clf = time() - time_dt_clf_start

# b1, m1 = FullReport(grid_tuned_dt_clf ,X_test, y_test, time_dt_clf)
# print(b1)