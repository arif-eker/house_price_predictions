import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


train_df = pd.read_pickle("datasets/house_train_df.pkl")
test_df = pd.read_pickle("datasets/house_test_df.pkl")

all_data = [train_df, test_df]
test_ıd = test_df.copy()
drop_list = ["index", "Id"]

for data in all_data:
    data.drop(drop_list, axis=1, inplace=True)

X_train = train_df.drop('SalePrice', axis=1)
y_train = np.ravel(train_df[["SalePrice"]])

X_test = test_df.drop('SalePrice', axis=1)
y_test = np.ravel(test_df[["SalePrice"]])

"""
knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
"""


# Bu hatayı manuel olarak elde edip k hiperarametresinin değişimini gözlemleyelim:
# RMSE = []

"""
for k in range(20):
    k = k + 2
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k =", k, "için RMSE değeri:", rmse)
"""
# GridSearchCV yöntemi ile optimum k'yı bulalım:
"""knn_params = {"n_neighbors": np.arange(2, 30, 1)}
knn_model = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_

# Final Model
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
"""




#Non-Linear SVR
"""svr_model = SVR()
svr_params = {"C": [0.01, 0.1, 1, 10, 100]}
# svr_params2 = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_

# Final Model
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
"""

# yapay sinir ağları
"""mlp_model = MLPRegressor().fit(X_train, y_train)
mlp_params = {"alpha": [0.1, 0.01],
              "hidden_layer_sizes": [(10, 20), (5, 5)]}
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
mlp_cv_model.best_params_

# Final Model
mlp_tuned = MLPRegressor(**mlp_cv_model.best_params_).fit(X_train, y_train)

y_pred = mlp_tuned.predict(X_test)

"""


""""# CART


# Model Tuning
cart_params = {"max_depth": [10, 20, 100],
               "min_samples_split": [15, 30, 50]}"""

cart_params = {"max_depth": [2, 3, 4, 5, 10, 20, 100, 1000],
              "min_samples_split": [2, 10, 5, 30, 50, 10],
              "criterion" : ["mse", "friedman_mse", "mae"]}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10).fit(X_train, y_train)
cart_cv_model.best_params_
print(cart_cv_model.best_params_)

# Final Model
cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
y_preds = cart_tuned.predict(X_test)


# Random Forest
"""rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Model Tuning
rf_params = {"max_depth": [5, 8, None],
             "max_features": [20, 50, 100],
             "n_estimators": [200, 500],
             "min_samples_split": [2, 5, 10]}

rf_params = {"max_depth": [3, 5, 8, 10, 15, None],
           "max_features": [5, 10, 15, 20, 50, 100],
            "n_estimators": [200, 500, 1000],
            "min_samples_split": [2, 5, 10, 20, 30, 50]}

rf_model = RandomForestRegressor()
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_
aaa = rf_cv_model.best_params_

print(aaa)
# Final Model
rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
y_preds = rf_tuned.predict(X_test)

"""

# GBM
# # Model Tuning
"""gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_params = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
              "max_depth": [3, 5, 8, 10,20,30],
              "n_estimators": [200, 500, 1000, 1500, 5000],
              "subsample": [1, 0.4, 0.5, 0.7],
              "loss": ["ls", "lad", "quantile"]}

gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model,
                            gbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_
print(gbm_cv_model.best_params_)
# Final Model
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_preds = gbm_tuned.predict(X_test)
print(gbm_cv_model.best_params_)

"""

"""xgb = XGBRegressor()

# Model Tuning
xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8],
              "n_estimators": [100, 1000],
              "colsample_bytree": [0.7, 1]}

xgb_params = {"learning_rate": [0.1, 0.01, 0.5],
              "max_depth": [5, 8, 15, 20],
              "n_estimators": [100, 200, 500, 1000],
              "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_

aaa = xgb_cv_model.best_params_
print(aaa)
# Final Model
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_preds = xgb_tuned.predict(X_test)
"""


"""lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
               "n_estimators": [200, 500, 1000, 5000],
              "max_depth": [6, 8, 10, 15, 20],
              "colsample_bytree": [1, 0.8, 0.5, 0.4]}


lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_
print(lgbm_cv_model.best_params_)

# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_preds = lgbm_tuned.predict(X_test)
"""





# Tahminleri csv yapma
C = []

for i in y_preds:
    C.append(i)

my_submission = pd.DataFrame({'Id': test_ıd.Id,'SalePrice': C})
my_submission.to_csv('datasets/submission_cart_newparams.csv', index=False)
