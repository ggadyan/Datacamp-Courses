# =============================================================================
# Chapter 1
# =============================================================================
# Video 1

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
dt=DecisionTreeClassifier(max_depth=2, random_state=1)

dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
accuracy_score(y_test, y_pred)

# Video 2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
dt=DecisionTreeClassifier(criterion="gini", random_state=1)

# Video 3
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=3)
dt=DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)

dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)

mse_dt=MSE(y_test, y_pred)
rmse_dt=mse_dt**(1/2)
print(rmse_dt)


# =============================================================================
# Chapter 2
# =============================================================================
# Video 1
#bias_variance tradeoff

# Video 2
#Cross-Validation (CV)
# 1. K-Fold CV
# 2. Hold-Out CV

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

SEED=123
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=SEED)
dt=DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=SEED)
MSE_CV=-cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

dt.fit(X_train, y_train)
y_predict_train=dt.predict(X_train)
y_predict_test=dt.predict(X_test)

print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
print('Train MSE {:.2f}'.format(MSE(y_train, y_predict_train)))
print('Test MSE {:.2f}'.format(MSE(y_test, y_predict_test)))


# Video 3
# Classification and Regression Trees (CART)
# Ensemble learning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=SEED)
lr=LogisticRegression(random_state=SEED)
knn=KNN()
dt=DecisionTreeClassifier(random_state=SEED)
classifiers=[('Logistic Regression', lr), ('K nearest Neighbours', knn), ('Classification Tree', dt)]
for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print('{:s}:{:.3f}'.format(clf_name, accuracy_score(y_test. y_pred)))

vc=VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred=vc.predict(X_test)
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

# =============================================================================
# Chapter 3
# =============================================================================
# Video 1

# Voting Classifier (same training set, different algorithms)
# Bagging  (different subsets of training set (with replacement), one algorithm)

# Classification >> aggregates predictions by majority voting
# Regression     >> aggregates prediction through averaging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

dt=DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
bc=BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred=bc.predict(X_test)
accuracy=accuracy_score (y_test, y_pred)
print ('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# Video 2
# out of bag evaluation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

dt=DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
bc=BaggingClassifier(base_estimator=dt, n_estimators=300, obb_score=True, n_jobs=-1)

bc.fit(X_train, y_train)
y_pred=bc.predict(X_test)
test_accuracy=accuracy_score (y_test, y_pred)
obb_accuracy=bc.obb_score_

print ('Test set accuracy: {:.3f}'.format(test_accuracy))
print ('OBB accuracy: {:.3f}'.format(obb_accuracy))


# Video 3
# Random Forests  classification>> aggregates predictions by majority voting
#                     regression>> aggregates predictions through averaging

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=SEED)

rf=RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)

rmse_test=MSE(y_test, y_pred)**(1/2)
print ('Test set RMSE of fr: {:.2f}'.format(rmse_test))

#feature importance 
import pandas as pd
import matplotlib.pyplot as plt

importances_rf=pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances_rf=importances_rf.sort_values()
sorted_importances_fr.plot(kind='barh', color='lightgreen'); plt.show()


# =============================================================================
# Chapter 4
# =============================================================================
# Video 1
# AdaBoost (Adaptive Boosting) >> increases the weights of the wrongly predicted instances and descreases the ones of the correctly predicted instances
# learning rate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

dt=DecisionTreeRegressor(max_depth=4, random_state=SEED)
adb_clf=AdaBoostClassifier(base_estimator=dt, n_estimators=100)
adb_clf.fit(X_train, y_train)
y_pred_proba=adb_clf.predict_proba(X_test)[:,1]
adb_clf_roc_auc_score=roc_auc_score(y_test, y_pred_proba)
print('ROC AUC SCORE: {:.2f}'.format(adb_clf_roc_auc_score))


# Video 2
# Gradient Boosting  >> the weak learner trains on the remaining errors 
# shrinkage

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=SEED)

gbt=GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)
gbt.fit(X_train, y_train)
y_pred=gbt.predict(X_test)
rmse_test=MSE(y_test, y_pred)**(1/2)
print('Test set RMSE: {:.2f}'.format(rmse_test))

# Video 3
# Stochastic Gradient Boosting >> each CART is trained on a sample (without replacement)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED=1
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=SEED)

sgbr=GradientBoostingRegressor(max_depth=1, subsample=0.8, max_feaures=0.2, n_estmimators=300, random_state=SEED)

sgbr.fit(X_train, y_train)
y_pred=sgbr.predict(X_test)
rmse_test=MSE(y_test, y_pred)**(1/2)
print('Test set RMSE {:.2f}'.format(rmse_test))



# =============================================================================
# Chapter 4
# =============================================================================
# Video 1
# Tuning a CART's hyperparameters

# classification >> accuracy
# regression     >> R squared
# Grid Search 

# Random Search
# Bayesian Optimization
# Generic Algorithms

from sklearn.tree import DecisionTreeClassifier
SEED=1
dt=DecisonTreeClassifier(random_state=SEED)
print (dt.get_params())



from sklearn.model_selection import GridSearchCV
params_dt= {'max_dept':[3,4,5,6], 'min_samples_leaf': [0.04, 0.06, 0.08], 'max_features':[0.2, 0.2, 0.6, 0.8]}
grid_dt=GridSearchCV(estimator=dt, param_grid= params_dt,scoring='accuracy', cv=10, n_jobs=-1 )

grid_dt.fit(X_train, y_train)
best_hyperparams=grid_dt.best_params_
print('Best Hyperparameters:\n', best_hyperparams)

best_CV_Score=grid_dt.best_score_print('Best CV accuracy'.format(best_CV_score))
best_model=grid_dt.best_estimator_

est_acc=best_model.score(X_test, y_test)
print('Test set accuracy of best model: {:.3f}'.format(test_acc))


# Video 2
# tuning an RF's hyperparameters
from sklearn.ensemble import RandomForestRegressor
SEED=1
rf=RandomForestRegressor(random_state=SEED)
rf.get_params()

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

params_dt= {'n_estimators':[300,400,500], 'max_depth': [4,6,8], 'min_samples_leaf':[0.1,0.2], 'max_features':['log2', 'sqrt']}
grid_rf=GridSearchCV(estimator=rf, param_grid_rf, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

grid_rf.fit(X_train, y_train)
best_hyperparams=grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)
best_model-grid_rf.best_estimator_
y_pred=best_model.perdict(X_test)
rmse_test=MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {".2f}'.format(rmse_test))
