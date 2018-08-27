import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import sklearn.feature_selection as fs
from sklearn import tree
import pydotplus
import collections
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

train = pd.read_csv("./data/all/train.csv")
test = pd.read_csv("./data/all/test.csv")

train["test"] = 0
test["test"] = 1
varlist_misc = ["id", "test"]

outcome = pd.DataFrame(train["SalePrice"])
outcome["ln_price"] = list(map(math.log, outcome["SalePrice"]))
outcome.columns = ["saleprice", "ln_price"]

t1 = train.drop("SalePrice", axis = 1)
t2 = test

#combine train-test data
df = pd.concat([t1, t2], axis = 0, sort = False)

#list variables
vars = list(df.keys()) #create list of variables in data
df.columns = [x.lower() for x in vars] #set var names to lower case

df.columns
[type(x) for x in df.columns]

#dimension of data
df.shape

#data summary stats
df.describe()
df.count()
df.head()

##tabulate variables
#list variables with missing values
df.agg("count")[df.agg("count") < df.shape[0]]
navars = df.agg("count")[df.agg("count") < df.shape[0]].index

##drop variables
varlist_drop = ["poolqc", "fence", "miscfeature", "lotarea", "utilities"]
df.drop(varlist_drop, axis = 1)
    
##fill missing values
#fill with median (numerical)
varlist_median_fill = ["lotfrontage", 
                       "bsmtunfsf", 
                       "totalbsmtsf", 
                       "garageyrblt",
                       "garagecars", 
                       "garagearea"]

for v in varlist_median_fill:
    df[v] = df[v].fillna(df[v].median())    

#fill with most common entry (categorical)
varlist_mcommon_fill = ["mszoning",
                        "garagetype",
                        "exterior1st",
                        "exterior2nd",
                        "masvnrtype",
                        "bsmtqual",
                        "bsmtcond",
                        "bsmtexposure",
                        "electrical",
                        "bsmtfullbath",
                        "bsmthalfbath",
                        "kitchenqual",
                        "functional",
                        "garagefinish",
                        "garagequal",
                        "garagecond",
                        "saletype"]   

for v in varlist_mcommon_fill:
    df[v] = df[v].fillna(df[v].value_counts().index[0])
  
#fill with zero
varlist_zero_fill = ["masvnrarea",
                     "bsmtfinsf1",
                     "bsmtfinsf2"]    

for v in varlist_zero_fill:
    df[v] = df[v].fillna(0)
    
#fill with specific values    
df["bsmtfintype1"] = df["bsmtfintype1"].fillna("None")
df["bsmtfintype2"] = df["bsmtfintype2"].fillna("NA")

##transform variables
df["ln_lotarea"] = pd.DataFrame(list(map(math.log, df["lotarea"] + 1)))

categorical_vars = ["mssubclass", 
                    "mszoning", 
                    "street", 
                    "landcontour", 
                    "lotconfig",
                    "landslope", 
                    "neighborhood", 
                    "condition1", 
                    "condition2",
                    "bldgtype", 
                    "housestyle", 
                    "yearbuilt", 
                    "yearremodadd", 
                    "roofstyle", 
                    "roofmatl", 
                    "exterior1st", 
                    "exterior2nd",
                    "masvnrtype", 
                    "foundation",  
                    "bsmtfintype1", 
                    "bsmtfintype2", 
                    "heating",
                    "centralair", 
                    "electrical", 
                    "garagetype", 
                    "mosold", 
                    "yrsold",
                    "saletype", 
                    "salecondition"]

ordinal_vars = ["lotshape",
                "overallqual",
                "overallcond", 
                "exterqual", 
                "extercond",
                "heatingqc", 
                "bsmtfullbath", 
                "bsmthalfbath", 
                "fullbath", 
                "halfbath",
                "bedroomabvgr", 
                "kitchenabvgr", 
                "kitchenqual", 
                "functional",
                "fireplacequ", 
                "garagefinish", 
                "garagequal", 
                "garagecond", 
                "paveddrive"]

numerical_vars = ["lotfrontage", 
                  "lotarea", 
                  "masvnrarea", 
                  "bsmtfinsf1", 
                  "bsmtfinsf2",
                  "bsmtunfsf", 
                  "totalbsmtsf", 
                  "1stflrsf", 
                  "2ndflrsf", 
                  "lowqualfinsf",
                  "grlivarea", 
                  "totrmsabvgrd", 
                  "fireplaces", 
                  "garageyrblt", 
                  "garagecars", 
                  "garagearea",
                  "wooddecksf",
                  "openporchsf", 
                  "enclosedporch",
                  "3ssnporch", 
                  "screenporch", 
                  "poolarea", 
                  "miscval"]

categorized_cols = categorical_vars + ordinal_vars + numerical_vars + \
                   varlist_drop + varlist_misc

#check all variables categorized
set(categorical_vars) & set(ordinal_vars)
set(ordinal_vars) & set(numerical_vars)
list(set(df.columns) - set(categorized_cols))

print((len(categorical_vars) + len(ordinal_vars) + len(numerical_vars))/df.shape[1])

#compute unique no. of types
for c in categorical_vars:
    print(c, len(df[c].value_counts()))
    
lotshape_dic = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
df_ordinal = pd.DataFrame({})
df_ordinal["lotshape"] = df["lotshape"].map(lambda x: lotshape_dic.get(x, 0))
df_ordinal.lotshape.value_counts()

quality_dic = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
df_ordinal["exterqual"] = df["exterqual"].map(lambda x: quality_dic.get(x, 0))
df_ordinal["extercond"] = df["extercond"].map(lambda x: quality_dic.get(x, 0))
df_ordinal["heatingqc"] = df["heatingqc"].map(lambda x: quality_dic.get(x, 0))
df_ordinal["kitchenqual"] = df["kitchenqual"].map(lambda x: quality_dic.get(x, 0))
df_ordinal.exterqual.value_counts()
df_ordinal.extercond.value_counts()    
df_ordinal.heatingqc.value_counts()
df_ordinal.kitchenqual.value_counts()
        
functional_dic = {"Typ": 8, "Min1": 7, "Min2": 6, "Mod": 5, "Maj1": 4, "Maj2": 3,
                  "Sev": 2, "Sal": 1}
df_ordinal["functional"] = df["functional"].map(lambda x: functional_dic.get(x, 0))
df_ordinal.functional.value_counts()

#one hot encoding
t_categorical_oht = df[list(categorical_vars.difference(ordinal_vars))]
df_categorical_oht = pd.get_dummies(t_categorical_oht, drop_first = True)
df_categorical_oht.shape

#ch,
df_categorical = pd.merge(df_ordinal, df_categorical_oht,
                        left_index = True, right_index = True)

##plot numerical variables
#bimodal:
sns.distplot(df["lotfrontage"])
sns.distplot(df["masvnrarea"])
sns.distplot(df["bsmtfinsf1"])
sns.distplot(df["bsmtfinsf2"])
sns.distplot(df["2ndflrsf"])
sns.distplot(df["garagearea"])
sns.distplot(df["wooddecksf"])
sns.distplot(df["openporchsf"])

#right-skewed:
sns.distplot(df["lotarea"])
sns.distplot(df["totalbsmtsf"])
sns.distplot(df["1stflrsf"])
sns.distplot(df["grlivarea"])

#discrete distribution:
sns.distplot(df["overallqual"])
sns.distplot(df["overallcond"])
sns.distplot(df["bsmtfullbath"])
sns.distplot(df["bsmthalfbath"])
sns.distplot(df["fullbath"])
sns.distplot(df["halfbath"])
sns.distplot(df["bedroomabvgr"])
sns.distplot(df["totrmsabvgrd"])
sns.distplot(df["fireplaces"])
sns.distplot(df["garagecars"])

#other distribution:
sns.distplot(df["yearbuilt"])
sns.distplot(df["yearremodadd"])
sns.distplot(df["bsmtunfsf"])
sns.distplot(df["lowqualfinsf"])
sns.distplot(df["kitchenabvgr"])
sns.distplot(df["enclosedporch"])
sns.distplot(df["3ssnporch"])
sns.distplot(df["screenporch"])
sns.distplot(df["poolarea"])
sns.distplot(df["miscval"])

#combine variables
df["totalbath"] = df["fullbath"] + df["halfbath"]
df["outsidesqft"] = df["wooddecksf"] + df["openporchsf"] \
            + df["enclosedporch"] + df["3ssnporch"] + df["screenporch"]
df["totalsqft_ver1"] = df["1stflrsf"] + df["2ndflrsf"]
df["totalsqft_ver2"] = df["grlivarea"] + df["totalbsmtsf"]

sns.distplot(df["totalbath"])
sns.distplot(df["outsidesqft"])
sns.distplot(df["totalsqft_ver1"])
sns.distplot(df["totalsqft_ver2"])

#compile data for training data to use
df_X = pd.merge(df[list(numerical_vars)], df_categorical, left_index = True,
            right_index = True)
list(df_X.columns)

df_y = pd.DataFrame(list(map(math.log, train["SalePrice"])))
df_y.columns = ["ln_price"]
sns.distplot(df_y["ln_price"])

##fit models
#import train/test data
c_train = pd.read_csv("./re/cleantrain.csv")
c_test = pd.read_csv("./re/cleantest.csv")

c_train.columns = [x.lower() for x in c_train.columns]
c_test.columns = [x.lower() for x in c_test.columns]

c_train_X = c_train.drop(["saleprice", "unnamed: 0", "garageyrblt"], axis = 1)
c_train_y = pd.DataFrame(list(map(math.log, c_train["saleprice"])), c_train["id"])

kg_test_X = c_test.drop(["unnamed: 0", "garageyrblt", "id"], axis = 1)

#split data
import sklearn.model_selection as ms
from sklearn.cross_validation import train_test_split
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(c_train_X, c_train_y, 
                                   test_size = 0.2, random_state = 42)

##linear regression
#remove ids from datasets
X_train = X_train_i.loc[:, X_train_i.columns != "id"]
X_test = X_test_i.loc[:, X_test_i.columns != "id"]

#ols predictions
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
ols_train_predictions = ols.predict(X_train)
ols_test_predictions = ols.predict(X_test)

#ols results:
ols_train_rmse = mean_squared_error(y_train, ols_train_predictions) ** (1/2)
ols_train_rmse

ols_test_rmse = mean_squared_error(y_test, ols_test_predictions) ** (1/2)
ols_test_rmse

#kaggle submission
kg_saleprices = pd.DataFrame(np.exp(ols.predict(kg_test_X)))
kg_saleprices.columns = ["SalePrice"]

kg_submission_ols = pd.concat([kg_saleprices, c_test], axis = 1)
kg_submission_ols = kg_submission_ols.loc[:, ["Id", "SalePrice"]]

kg_submission_ols.to_csv("./submissions/kg_submission_ols.csv", index = False)

kg_ols = kg_submission_ols 
kg_ols.columns = ["Id", "OLS"]
#kg_ols["ols"] = list(map(math.log, kg_submission_ols["SalePrice"]))
#kg_ols = kg_ols.drop("SalePrice", axis = 1)


##ridge regression
ridge = linear_model.Ridge(alpha = 1)
ridge.fit(X_train, y_train)

ridge_train_rmse = []
ridge_test_rmse = []
alphas = np.linspace(0.001, 1, 100)
for i in range(len(alphas)):
    ridge.set_params(alpha = alphas[i])
    ridge.fit(X_train, y_train)
    ridge_train_rmse.append(mean_squared_error(y_train, ridge.predict(X_train)) ** (1/2))
    ridge_test_rmse.append(mean_squared_error(y_test, ridge.predict(X_test)) ** (1/2))
ridge_train_rmse 
ridge_test_rmse    

ridge_rmse = pd.DataFrame({"train_mse": ridge_train_rmse,
                           "test_mse": ridge_test_rmse})
ridge_rmse

#rmse by alpha
title1 = "Variation of ridge rmse with alpha"
axes = ridge_rmse.plot(logx = False, title = title1, legend = True)
axes.set_xlabel("alpha")
axes.set_ylabel("rmse")
plt.plot()

#ridge coefficients
alphas = np.linspace(0.001, 1, 100)
ridge_coef = pd.DataFrame({})
for i in alphas:
    ridge.set_params(alpha = i)
    ridge.fit(X_train, y_train)
    ridge_coef.append(pd.DataFrame(ridge.coef_))
        
df_ridge_coef = pd.DataFrame(ridge_coef, index = alphas, columns = X_train.columns)
title = "Ridge coefficients as a function of the regularization"
axes = df_ridge_coef.plot(logx = True, title = title, legend = False)
axes.set_xlabel("alpha")
axes.set_ylabel("coefficients")
plt.plot()

#ridge cv
def display_scores(scores):
    print("Mean:", scores.mean())
    print("Std:", scores.std())
    print("Scores:", scores)

#ridge grid search
ridge_param_list = [{"alpha": [x/1000.0 for x in range(1, 30, 1)]}]
ridge_model = linear_model.Ridge()
ridge_grid_search = GridSearchCV(ridge_model, ridge_param_list, cv = 10,
                                 scoring = "neg_mean_squared_error", verbose = 100)
ridge_grid_search.fit(X_train, y_train)
ridge_grid_search.best_params_
ridge_grid_search.best_estimator_

#ridge predictions
ridge_bp = linear_model.Ridge(alpha=0.029, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
ridge_bp.fit(X_train, y_train)
ridge_bp_train_predictions = ridge_bp.predict(X_train)
ridge_bp_test_predictions = ridge_bp.predict(X_test)

#ridge results
ridge_train_rmse = mean_squared_error(y_train, ridge_bp_train_predictions) ** (1/2)
ridge_train_rmse

ridge_test_rmse = mean_squared_error(y_test, ridge_bp_test_predictions) ** (1/2)
ridge_test_rmse

#kaggle submission
kg_saleprices_ridge = pd.DataFrame(np.exp(ridge_bp.predict(kg_test_X)))
kg_saleprices_ridge.columns = ["SalePrice"]

kg_submission_ridge = pd.concat([kg_saleprices_ridge, c_test], axis = 1)
kg_submission_ridge = kg_submission_ridge.loc[:, ["Id", "SalePrice"]]

kg_submission_ridge.to_csv("./submissions/kg_submission_ridge.csv", index = False)

kg_ridge = kg_submission_ridge
kg_ridge.columns = ["Id", "Ridge"]
#kg_ridge["ridge"] = list(map(math.log, kg_submission_ridge["SalePrice"]))
#kg_ridge = kg_ridge.drop("SalePrice", axis = 1)

##lasso regression
lasso = linear_model.Lasso(alpha = 1)
lasso.fit(X_train, y_train)

lasso_train_rmse = []
lasso_test_rmse = []
alphas = np.linspace(0.001, 0.5, 100)
for i in range(len(alphas)):
    lasso.set_params(alpha = alphas[i])
    lasso.fit(X_train, y_train)
    lasso_train_rmse.append(mean_squared_error(y_train, lasso.predict(X_train)) ** (1/2))
    lasso_test_rmse.append(mean_squared_error(y_test, lasso.predict(X_test)) ** (1/2))
lasso_train_rmse 
lasso_test_rmse    

lasso_rmse = pd.DataFrame({"train_mse": lasso_train_rmse,
                           "test_mse": lasso_test_rmse})
lasso_rmse = lasso_rmse.set_index(alphas)

#rmse by alpha
title1 = "Variation of lasso rmse with alpha"
axes = lasso_rmse.plot(logx = False, title = title1, legend = True)
axes.set_xlabel("alpha")
axes.set_ylabel("rmse")
plt.plot()

#lasso coefficients
alphas = np.linspace(0.001, 1, 100)
lasso_coef = pd.DataFrame({})
for i in alphas:
    lasso.set_params(alpha = i)
    lasso.fit(X_train, y_train)
    lasso_coef.append(pd.DataFrame(lasso.coef_))
        
df_lasso_coef = pd.DataFrame(lasso_coef, index = alphas, columns = X_train.columns)
title = "lasso coefficients as a function of the regularization"
axes = df_lasso_coef.plot(logx = True, title = title, legend = False)
axes.set_xlabel("alpha")
axes.set_ylabel("coefficients")
plt.plot()

#lasso grid search
lasso_param_list = [{"alpha": [x/1000.0 for x in range(1, 30, 1)]}]
lasso_model = linear_model.Lasso()
lasso_grid_search = GridSearchCV(lasso_model, lasso_param_list, cv = 10,
                                 scoring = "neg_mean_squared_error", verbose = 100)
lasso_grid_search.fit(X_train, y_train)
lasso_grid_search.best_params_
lasso_grid_search.best_estimator_

#lasso predictions
lasso_bp = linear_model.Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso_bp.fit(X_train, y_train)
lasso_bp_train_predictions = lasso_bp.predict(X_train)
lasso_bp_test_predictions = lasso_bp.predict(X_test)

#lasso results
lasso_train_rmse = mean_squared_error(y_train, lasso_bp_train_predictions) ** (1/2)
lasso_train_rmse

lasso_test_rmse = mean_squared_error(y_test, lasso_bp_test_predictions) ** (1/2)
lasso_test_rmse

#kaggle submission
kg_saleprices_lasso = pd.DataFrame(np.exp(lasso_bp.predict(kg_test_X)))
kg_saleprices_lasso.columns = ["SalePrice"]

kg_submission_lasso = pd.concat([kg_saleprices_lasso, c_test], axis = 1)
kg_submission_lasso = kg_submission_lasso.loc[:, ["Id", "SalePrice"]]

kg_submission_lasso.to_csv("./submissions/kg_submission_lasso.csv", index = False)

kg_lasso = kg_submission_lasso
kg_lasso.columns = ["Id", "Lasso"]
#kg_lasso["lasso"] = list(map(math.log, kg_submission_lasso["SalePrice"]))
#kg_lasso = kg_lasso.drop("SalePrice", axis = 1)

#decision tree model
tree_model = DecisionTreeRegressor(max_depth = 3)
tree_model.fit(X_train, y_train)
mean_squared_error(y_train, tree_model.predict(X_train)) ** (1/2)
mean_squared_error(y_test, tree_model.predict(X_test)) ** (1/2)

#tree model grid search
dt_param_grid = [{"max_depth": list(range(2, 50, 5)),
                  "max_features": list(range(2, 150, 5))}]

dt_model = DecisionTreeRegressor()

dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv = 10, verbose = 100,
                              scoring = "neg_mean_squared_error")

dt_grid_search.fit(X_train, y_train)
dt_grid_search.best_params_
dt_grid_search.best_estimator_
dt_grid_search.cv_results_

bp_tree_model = DecisionTreeRegressor(criterion='mse', max_depth=7, max_features=112,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
bp_tree_model.fit(X_train, y_train)
bp_tree_train_predictions = bp_tree_model.predict(X_train)
bp_tree_test_predictions = bp_tree_model.predict(X_test)

#decision tree results
tree_train_rmse = mean_squared_error(y_train, bp_tree_train_predictions) ** (1/2)
tree_train_rmse

tree_test_rmse = mean_squared_error(y_test, bp_tree_test_predictions) ** (1/2)
tree_test_rmse

#visualize grid search results
dt_results = pd.DataFrame(dt_grid_search.cv_results_)
dt_results
dt_results.columns

#max depth
dt_plotdata_maxdepth = dt_results[dt_results.param_max_features == 112].loc[:, ["param_max_depth", 
          "mean_train_score", "mean_test_score"]]
dt_plotdata_maxdepth["train_rmse"] = np.sqrt(-dt_plotdata_maxdepth.mean_train_score)
dt_plotdata_maxdepth["test_rmse"] = np.sqrt(-dt_plotdata_maxdepth.mean_test_score)
dt_plotdata_maxdepth = dt_plotdata_maxdepth[["param_max_depth", "train_rmse", "test_rmse"]]
dt_plotdata_maxdepth = dt_plotdata_maxdepth.set_index("param_max_depth")
title = "Decision tree regressor cross-validation rmse - no. of features = 112"
axes = dt_plotdata_maxdepth.plot(title = title)
axes.set_xlabel("Max depth")
axes.set_ylabel("RMSE")

#max features
dt_plotdata_maxfeatures = dt_results[dt_results.param_max_depth == 7].loc[:,
                                        ["param_max_features", "mean_train_score",
                                        "mean_test_score"]]
dt_plotdata_maxfeatures["train_rmse"] = np.sqrt(-dt_plotdata_maxfeatures.mean_train_score)
dt_plotdata_maxfeatures["test_rmse"] = np.sqrt(-dt_plotdata_maxfeatures.mean_test_score)
dt_plotdata_maxfeatures = dt_plotdata_maxfeatures[["param_max_features", "train_rmse", "test_rmse"]]
dt_plotdata_maxfeatures = dt_plotdata_maxfeatures.set_index("param_max_features")
title = "Decision tree regressor cross-validation rmse - max depth = 7"
axes = dt_plotdata_maxdepth.plot(title = title)
axes.set_xlabel("Max features")
axes.set_ylabel("RMSE")

dt_results.mean_test_score

#visualize tree model
tree_data = tree.export_graphviz(bp_tree_model, 
                     feature_names = X_train.columns,
                     out_file = None,
                     filled = True,
                     rounded = True)

graph = pydotplus.graph_from_dot_data(tree_data)

colors = ("turquoise", "orange")
edges = collections.defaultdict(list)                     

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png("./figures/tree.png")

#kaggle submission
kg_saleprices_tree = pd.DataFrame(np.exp(bp_tree_model.predict(kg_test_X)))
kg_saleprices_tree.columns = ["SalePrice"]

kg_submission_tree = pd.concat([kg_saleprices_tree, c_test], axis = 1)
kg_submission_tree = kg_submission_tree.loc[:, ["Id", "SalePrice"]]

kg_submission_tree.to_csv("./submissions/kg_submission_tree.csv", index = False)

kg_tree = kg_submission_tree
kg_tree.columns = ["Id", "Tree"]
#kg_tree["tree"] = list(map(math.log, kg_submission_tree["SalePrice"]))
#kg_tree = kg_tree.drop("SalePrice", axis = 1)

##random forest model
from sklearn.ensemble import RandomForestRegressor
rforest_model = RandomForestRegressor(max_depth = 10)
rforest_model.fit(X_train, y_train)

mean_squared_error(y_train, rforest_model.predict(X_train)) ** (1/2)
mean_squared_error(y_test, rforest_model.predict(X_test)) ** (1/2)

#random forest cv
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rforest_model, X_train, y_train, cv = 10,
                         scoring = "neg_mean_squared_error")
scores_rforest_train_rmse = np.sqrt(-scores)
scores_rforest_train_rmse
display_scores(scores_rforest_train_rmse)

rforest_model_train_rmse = mean_squared_error(y_train, rforest_model.predict(X_train)) ** (1/2)
rforest_model_train_rmse
rforest_model_test_rmse = mean_squared_error(y_test, rforest_model.predict(X_test)) ** (1/2)
rforest_model_test_rmse

#hyperparameter optimization
rf_param_grid = [{"n_estimators": list(range(5, 65, 5)),
               "max_features": list(range(5, 45, 5)),
               "max_depth": list(range(1, 20, 1))}]

grid_search = GridSearchCV(rforest_model, rf_param_grid, cv = 5, verbose = 100,
                           scoring = "neg_mean_squared_error")

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_

#evaluate best random forest model on test data
rforest_bp = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=15,
           max_features=40, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=45, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
rforest_bp.fit(X_train, y_train)

rforest_bp_train_predictions = rforest_bp.predict(X_train)
rforest_bp_test_predictions = rforest_bp.predict(X_test)

#random forest results
rforest_bp_train_rmse = mean_squared_error(y_train, rforest_bp_train_predictions) ** (1/2)
rforest_bp_train_rmse

rforest_bp_test_rmse = mean_squared_error(y_test, rforest_bp_test_predictions) ** (1/2)
rforest_bp_test_rmse


results = pd.DataFrame(grid_search.cv_results_)
results.shape
results.describe()
results.columns
results["param_n_estimators"]
(- results["mean_test_score"]) ** (1/2)

gs_chart_data_estimators = results[(results.param_max_features == 40) & (results.param_max_depth == 9)].loc[:, ["param_n_estimators", "mean_test_score", "mean_train_score"]]
gs_chart_data_estimators.head()
t = gs_chart_data_estimators.reset_index().drop(["index", "param_n_estimators"], axis = 1)
t.plot()
gs_chart_data_features.plot()

#kaggle submission
kg_saleprices_rforest = pd.DataFrame(np.exp(rforest_bp.predict(kg_test_X)))
kg_saleprices_rforest.columns = ["SalePrice"]

kg_submission_rforest = pd.concat([kg_saleprices_rforest, c_test], axis = 1)
kg_submission_rforest = kg_submission_rforest.loc[:, ["Id", "SalePrice"]]

kg_submission_rforest.to_csv("./submissions/kg_submission_rforest.csv", index = False)

kg_rforest = kg_submission_rforest
kg_rforest.columns = ["Id", "Rforest"]
#kg_rforest["rforest"] = list(map(math.log, kg_submission_rforest["SalePrice"]))
#kg_rforest = kg_rforest.drop("SalePrice", axis = 1)

##ensemble 1
ens_data = pd.merge(kg_ols, kg_ridge, on = "Id").merge(kg_lasso, on = "Id") \
            .merge(kg_tree, on = "Id").merge(kg_rforest, on = "Id")
ens_data = ens_data.set_index("Id")
title = "Comparison of model predictions for test data"
axes = ens_data.plot(title = title)
axes.set_xlabel("Observation ID")

ens_data["SalePrice"] = (ens_data.OLS + ens_data.Ridge + ens_data.Lasso +
        ens_data.Tree + ens_data.Rforest)/5
ens_data = ens_data.loc[:, "SalePrice"]

ens_data.to_csv("./submissions/kg_submission_avg1.csv", index = True)

#ensemble 2
ens_data = pd.merge(kg_ols, kg_ridge, on = "Id").merge(kg_lasso, on = "Id") 
ens_data = ens_data.set_index("Id")

ens_data["SalePrice"] = (ens_data.OLS + ens_data.Ridge + ens_data.Lasso)/3
ens_data = ens_data.loc[:, "SalePrice"]

ens_data.to_csv("./submissions/kg_submission_avg2.csv", index = True)

#ensemble 3
ens_data = pd.merge(kg_ridge, kg_lasso, on = "Id")
ens_data = ens_data.set_index("Id")

ens_data["SalePrice"] = (ens_data.Ridge + ens_data.Lasso)/2
ens_data = ens_data.loc[:, "SalePrice"]

ens_data3 = pd.DataFrame(ens_data)
ens_data.to_csv("./submissions/kg_submission_avg3.csv", index = True)

#ensemble 4
ens_data = pd.merge(kg_ols, kg_ridge, on = "Id").merge(kg_lasso, on = "Id") 
ens_data["OLS"].mean()
ens_data["SalePrice"] = (ens_data.Ridge + ens_data.Lasso)/2

ens_data = ens_data.set_index("Id")
ens_data = ens_data.loc[:, "SalePrice"]

ens_data.to_csv("./submissions/kg_submission_avg3.csv", index = True)

#ensemble 5 - group
kdata = pd.read_csv("./submissions/kevin_predictions.csv")
kdata = kdata.loc[:, ["Id", "Lasso", "Random_Forest"]]
kdata

mdata = pd.read_csv("./submissions/lgb_model_predict.csv")
mdata.columns = ["Id", "Lgb_boost"]

edata = pd.merge(kg_ols, kg_ridge, on = "Id").merge(kg_tree, on = "Id") 
edata

gdata = pd.merge(kdata, mdata, on = "Id").merge(edata, on = "Id")
gdata["SalePrice"] = (gdata.OLS + gdata.Ridge + gdata.Lasso + gdata.Tree +
     gdata.Random_Forest + gdata.Lgb_boost)/6

gdata.to_csv("./submissions/kg_submission_group_avg.csv", index = True)


#ensemble 6 - group
kdata = pd.read_csv("./submissions/kevin_predictions.csv")
kdata = kdata.loc[:, ["Id", "Lasso", "Random_Forest"]]
kdata

mdata = pd.read_csv("./submissions/lgb_model_predict.csv")
mdata.columns = ["Id", "Lgb_boost"]

edata = pd.merge(kg_ols, kg_ridge, on = "Id").merge(kg_tree, on = "Id") 
edata

gdata = pd.merge(kdata, mdata, on = "Id").merge(edata, on = "Id")
gdata["SalePrice"] = (gdata.Lasso + gdata.Lgb_boost)/2

gdata.to_csv("./submissions/kg_submission_group_avg2.csv", index = True)

##summary of variables
chart_data_train = df[df.test == 0]
chart_data_train = pd.concat([chart_data_train, outcome], axis = 1)
df.test
#fig1: count of obs by year
with sns.axes_style("white"):
    g = sns.factorplot("yrsold", data = df, aspect = 2, kind = "count",
                       color = "steelblue")
#note: limited data for 2010

#fig2: count of obs by month
with sns.axes_style("white"):
    g = sns.factorplot("mosold", data = df, aspect = 2, kind = "count",
                       color = "steelblue")
#note: most sales during summer months

#fig3: 
with sns.axes_style("white"):
    sns.jointplot("yrsold", "mosold", data = df, kind = "hex")
#notes: pattern of sales across months different in last year (also some in first)

#fig4:
sns.distplot(outcome["saleprice"])
#notes: sale price skewed right

#fig5:
sns.distplot(outcome["ln_price"])
#notes: log sale price normally distributed

#fig6
with sns.axes_style(style = "ticks"):
    g = sns.factorplot("yrsold", "ln_price", data = chart_data_train, kind = "box")
#notes: similar price distribution across all years

#fig7
with sns.axes_style("white"):
    g = sns.factorplot("yearbuilt", data = df, aspect = 2, kind = "count",
                    color = "steelblue")
    g.set_xticklabels(step = 10)
#notes: sales concentrated among newly built homes

#fig8
sns.lmplot(x = "1stflrsf", y = "ln_price", col = "yrsold", hue = "yrsold", 
          col_wrap = 2, data = chart_data_train)
#notes: correlation of price with 1st floor sqft similar across years

#fig9
sns.lmplot(x = "2ndflrsf", y = "ln_price", col = "yrsold", hue = "yrsold", 
          col_wrap = 2, data = chart_data_train)
#notes: correlation of price with 2nd floor sqft similar across years,
#need to create a new variable for total sqft

#fig10
sns.lmplot(x = "totalbsmtsf", y = "ln_price", col = "yrsold", hue = "yrsold", 
          col_wrap = 2, data = train)
#notes: correlation of price with bsmt sqft similar across years,
#no zero values as in 2nd floor sqft

#fig11
sns.lmplot(x = "fullbath", y = "ln_price", col = "yrsold", hue = "yrsold", 
          col_wrap = 2, data = train)
#notes: correlation of price with full bath is similar across years,
#need to create new bathroon variable (full + half baths)

#fig12
sns.lmplot(x = "yearbuilt", y = "ln_price", data = chart_data_train)
#notes: fairly strong relationship between price and year built

#fig13
sns.boxplot(x = "overallqual", y = "ln_price", data = chart_data_train)
#notes: upward relationship between price and condition for bottom half, 
#flat for top half

#fig14
sns.boxplot(x = "salecondition", y = "ln_price", data = chart_data_train)
#notes: partial appears to have higher sale price

sns.boxplot(x = "yearbuilt", y = "ln_price", data = chart_data_train[chart_data_train.yearbuilt > 2000])


##tabulate variables
#mssubclass (categorical): dwelling type
df.mssubclass.count()
df.mssubclass.dtype
df.mssubclass.value_counts().sort_index()
df.mssubclass.isnull().sum() #none
#notes: most 1946 or newer

#mszoning (categorical): zoning classification
df.mszoning.count()
df.mzzoning.dtype
df.mszoning.value_counts().sort_index()
df.mszoning.isnull().sum() #has missing values
df.mszoning.value_counts().index[0]
#notes: most low or medium density residential

#lotfrontage (numeric): length of street connected to property
df.lotfrontage.count()
df.lotfrontage.dtype
df.lotfrontage.describe()
df.lotfrontage.isnull().sum()
sns.distplot(df["lotfrontage"][df["lotfrontage"].isnull() == False])
df.lotfrontage.isnull().sum()
sns.distplot(df["lotfrontage"])
#notes: convert missing values

#lotarea (numeric): lot size
df.lotarea.count()
df.lotarea.describe()
df.lotarea.isnull().sum() #none
sns.distplot(df["lotarea"])
sns.distplot(pd.DataFrame(list(map(math.log, df["lotarea"]))))
#notes: need to transform

#street (categorical): type of road access
df.street.count()
df.street.dtype
df.street.value_counts().sort_index()
df.street.isnull().sum() #none
#notes: vast majority pavement

#alley (categorical): type of alley access
df.alley.count()
df.alley.dtype
df.alley.value_counts().sort_index()
#notes: equal breakdown, but few obs

#lotshape (categorical): property shape
df.lotshape.count()
df.lotshape.dtype
df.lotshape.isnull().sum() #none
df.lotshape.value_counts().sort_index()
#notes: majority among two categories

#landcontour (categorical): flatness of property
df.landcontour.count()
df.landcontour.dtype
df.landcontour.isnull().sum() #none
df.landcontour.value_counts().sort_index()
#notes: most concentrated among one category

#utilities (categorical): type of utilities available
df.utilities.count()
df.utilities.dtype
df.utilities.isnull().sum()
df.utilities.value_counts().sort_index()
#notes: concentrated among one category, drop

#lotconfig (categorical): lot configuration
df.lotconfig.count()
df.lotconfig.dtype
df.lotconfig.isnull().sum() #none
df.lotconfig.value_counts().sort_index()
#notes: concentrated among two categories

#landslope (categorical): slope of property
df.landslope.count()
df.landslope.dtype
df.landslope.isnull().sum() #none
df.landslope.value_counts().sort_index()
#concentrated among one category

#neighborhood (categorical): location with city
df.neighborhood.count()
df.neighborhood.dtype
df.neighborhood.isnull().sum() #none
df.neighborhood.value_counts().sort_index()
#notes: some concentration among neighborhoods

#condition1 (categorical): proximity to various conditions
df.condition1.count()
df.condition1.dtype
df.condition1.isnull().sum() #none
df.condition1.value_counts().sort_index()
#notes: concentrated among one category

#condition2 (categorical): proximity to various conditions (if multiple)
df.condition2.count()
df.condition2.dtype
df.condition2.isnull().sum() #none
df.condition2.value_counts().sort_index()
#notes: vast majority among one category

#bldgtype (categorical): dwelling type
df.bldgtype.count()
df.bldgtype.dtype
df.bldgtype.isnull().sum() #none
df.bldgtype.value_counts().sort_index()
#notes: concentrated among one category

#housestyle (categorical): style of dwelling
df.housestyle.isnull().sum() #none
df.housestyle.value_counts().sort_index()
#notes: concentration among a few categories

#overallqual (ordinal): overall quality 
df.overallqual.isnull().sum() #none
df.overallqual.value_counts().sort_index()
#notes: distributed across range of qualities

#overallcond (ordinal): overall condition
df.overallcond.isnull().sum() #none
df.overallcond.value_counts().sort_index()
#notes: broad distribution, missing top category

#yearbuilt (categorical): construction year
df.yearbuilt.isnull().sum() #none
df.yearbuilt.value_counts().sort_index()
#notes: a lot of years

#yearremodadd (categorical): remodel date
df.yearremodadd
df.yearremodadd.value_counts().sort_index()
#notes: a lot of years

#roofstyle (categorical): roof type
df.roofstyle.isnull().sum() #none
df.roofstyle.value_counts().sort_index()
#notes: most in two categories

#roofmatl (categorical): roof material
df.roofmatl.isnull().sum() #none
df.roofmatl.value_counts().sort_index()
#notes: most in one category

#exterior1st (categorical): exterior covering
df.exterior1st.isnull().sum() 
df.exterior1st.value_counts().sort_index()
#notes: broad distribution across categories

#exterior2nd (categorical): exterior covering (if multiple)
df.exterior2nd.isnull().sum()
df.exterior2nd.value_counts().sort_index()
df.exterior2nd.isnull().sum()
#notes: broad distribution across categories

#masvnrtype (categorical): veneer type
#np.unique(train.masvnrtype)
df.masvnrtype.isnull().sum()
df.masvnrtype.value_counts()
#notes: broad distribution across categories

#masvnrarea (numeric): veneer area
df.masvnrarea.describe()
df.masvnrarea.isnull().sum()
sns.distplot(df["masvnrarea"]

#exterqual (ordinal): quality of external material
df.exterqual.isnull().sum() #none
df.exterqual.value_counts().sort_index()
#notes: mainly distributed across two categories

#extercond (ordinal): present condition of external material
df.extercond.isnull().sum() #none
df.extercond.value_counts().sort_index()
#notes: most in one category

#foundation (categorical): foundation type
df.foundation.isnull().sum() #none
df.foundation.value_counts().sort_index()
#notes: concentrated amond three categories

#bsmtqual (categorical): basement height
df.bsmtqual.isnull().sum()
df.bsmtqual.value_counts().sort_index()
#notes: concentrated amond three categories

#bsmtcond (categorical): basement condition
df.bsmtcond.isnull().sum()
df.bsmtcond.value_counts().sort_index()
#notes: mostly one category

#bsmtexposure (categorical): wall type
df.bsmtexposure.isnull().sum()
df.bsmtexposure.value_counts().sort_index()
#notes: broad distribution across categories

#bsmtfintype1 (categorical): basement quality
df.bsmtfintype1.isnull().sum()
df.bsmtfintype1.value_counts().sort_index()
#notes: broad distribution across categories

#bsmtfinsf1 (numeric): "type1" square feet
df.bsmtfinsf1.isnull().sum()
sns.distplot(df["bsmtfinsf1"])
#notes: bimodal distribution, large density at zero

#bsmtfintype2 (categorical): basement quality (if multiple)
df.bsmtfintype2.isnull().sum()
df.bsmtfintype2.value_counts().sort_index()
#notes: concentrated among one category

#bsmtfinsf2 (numerical): "type2" square feet
sns.distplot(df["bsmtfinsf2"])
#notes: largely distributed at zero

#bsmtunfsf (numerical): unfinished square feet
sns.distplot(df["bsmtunfsf"])
#notes: right skewed, more mass as approach zero

#totalbsmtsf (numerical): total basement square feet
sns.distplot(df["totalbsmtsf"])
#notes: approximately normally distributed, some right skew and mass at zero

#heating (categorical): heating type
df.heating.isnull().sum() #none
df.heating.value_counts().sort_index()
#notes: mostly in one category

#heatingqc (ordinal): heating quality and condition
df.heatingqc.isnull().sum() #none
df.heatingqc.value_counts().sort_index()
#notes: broad distribution across categories

#centralair (categorical): central air conditioning
df.centralair.isnull().sum() #none
df.centralair.value_counts().sort_index()
#notes: mostly in one category

#electrical (categorical): type of electrical system
df.electrical.isnull().sum()
df.electrical.value_counts().sort_index()
#notes: mostly in one category

#1stflrsf (numerical): first floor square feet
df["1stflrsf"].isnull().sum() #none
sns.distplot(df["1stflrsf"])
#notes: close to normal distribution, slight right skew

#2ndflrsf (numerical): second floor square feet
df["2ndflrsf"].isnull().sum() #none
sns.distplot(df["2ndflrsf"])
#notes: bimodal distribution, mass at zero, normal distribution for non-zero values

#lowqualfinsf (numerical): low quality finished square feet
df.lowqualfinsf.isnull().sum() #none
sns.distplot(df["lowqualfinsf"])
#notes: density concentrated mainly at zero

#grlivarea (numerical): above ground square feet
df.grlivarea.isnull().sum() #none
sns.distplot(df["grlivarea"])
#notes: close to normal distribution, some right skew

#bsmtfullbath (ordinal): basement full bathrooms
df.bsmtfullbath.isnull().sum()
df.bsmtfullbath.value_counts().sort_index()
#notes: mostly in two categories

#bsmthalfbath (ordinal): basement half bathrooms
df.bsmthalfbath.isnull().sum()
df.bsmthalfbath.value_counts().sort_index()
#notes: mostly in one category

#fullbath (ordinal): "above grade" full bathrooms
df.fullbath.isnull().sum() #none
df.fullbath.value_counts().sort_index()

#halfbath (ordinal): "above grade" half bathrooms
df.halfbath.isnull().sum() #none
df.halfbath.value_counts().sort_index()
#notes: mostly in two categories

#bedroomabvgr (ordinal): "above grade" bedrooms
df.bedroomabvgr.isnull().sum() #none
df.bedroomabvgr.value_counts().sort_index()
#notes: concentrated among a few categories

#kitchenabvgr (ordinal): "above grade" kitchens
df.kitchenabvgr.isnull().sum() #none
df.kitchenabvgr.value_counts().sort_index()
#notes: mostly in one category

#kitchenqual (ordinal): kitchen quality
df.kitchenqual.isnull().sum()
df.kitchenqual.value_counts().sort_index()
#notes: broad distribution across categories

#totrmsabvgrd (numeric): total "above grade" room (excl. bathrooms)
df.totrmsabvgrd.isnull().sum() #none
df.totrmsabvgrd.value_counts().sort_index()
#notes: broad distribution across categories

#functional (ordinal): home "functionality"
df.functional.isnull().sum()
df.functional.value_counts().sort_index()
#notes: mainly in one category

#fireplaces (numerical): no. of fireplaces
df.fireplaces.isnull().sum() #none
df.fireplaces.value_counts().sort_index()
#notes: mainly concentrated among two categories

#fireplacequ (ordinal): fireplace quality 
df.fireplacequ.isnull().sum()
df.fireplacequ.value_counts().sort_index()
#notes: mainly concentrated among two categories

#garagetype (categorical): garage location
df.garagetype.isnull().sum()
df.garagetype.value_counts()
#notes: mainly concentrated among two categories

#garageyrblt (numerical): year garage was built
df.garageyrblt.isnull().sum()
df.garageyrblt.describe()
sns.distplot(df["garageyrblt"])
#notes: distributed across many years

#garagefinish (ordinal): garage interior finish
df.garagefinish.isnull().sum()
df.garagefinish.value_counts()
#notes: distributed across three categories

#garagecars (numeric): garage size (no. of cars)
df.garagecars.isnull().sum()
df.garagecars.value_counts()
#notes: distributed across three main categories

#garagearea (numeric): garage size (in sq. feet)
sns.distplot(df["garagearea"])
#notes: close to normally distributed, mass at zero

#garagequal (ordinal): garage quality
df.garagequal.isnull().sum()
df.garagequal.value_counts()
#notes: concentrated within one category

#garagecond (ordinal): garage cond
df.garagecond.isnull().sum()
df.garagecond.value_counts()
#notes: concentrated within one category

#paveddrive (ordinal): paved driveway
df.paveddrive.isnull().sum() #none
df.paveddrive.value_counts()
#notes: concentrated among one category

#wooddecksf (numeric): wood deck area (in sq. feet)
df.wooddecksf.isnull().sum() #none
df.wooddecksf.describe()
sns.distplot(df["wooddecksf"])
#notes: large mass at zero, right skewed for non-zero values

#openporchsf (numerical): open poarch area (in sq. feet)
df.openporchsf.isnull().sum()
df.openporchsf.describe()
sns.distplot(df["openporchsf"])
#notes: large mass at zero, right skewed for non-zero values

#enclosedporch (numerical): enclosed porch area (in sq. feet)
df.enclosedporch.isnull().sum() #none
df.enclosedporch.describe()
sns.distplot(df["enclosedporch"])
#notes: mostly distributed at zero

#3ssnporch (numerical): three season porch area in sqft
df["3ssnporch"].isnull().sum() #none
df["3ssnporch"].describe()
sns.distplot(df["3ssnporch"])
#notes: mostly distributed at zero

#screenporch (numerical): screen porch area in sqft
df["screenporch"].isnull().sum() #none
df["screenporch"].describe()
sns.distplot(df["screenporch"])
#notes: mostly distributed at zero

#poolarea (numerical): pool area (in sq. feet)
df.poolarea.isnull().sum() #none
df.poolarea.describe()
sns.distplot(df["poolarea"])
#notes: mostly distributed at zero

#poolqc (ordinal): pool quality
df.poolqc.isnull().sum()
df.poolqc.value_counts()
#notes: mostly missing

#fence (categorical): fench quality
df.fence.isnull().sum()
df.fence.value_counts()
#notes: mostly missing

#miscfeature: other features
df.miscfeature.isnull().sum()
df.miscfeature.value_counts()
#notes: mostly missing

#miscval (numerical): value of other features
df.miscval.isnull().sum() #none
df.miscval.describe()
sns.distplot(df["miscval"])
#notes: mostly zero

#mosold (categorical): month of sale
df.mosold.isnull().sum() #none
df.mosold.value_counts()

#yrsold (categorical): year of sale
df.yrsold.isnull().sum() #none
df.yrsold.value_counts()

#saletype: sale type
df.saletype.isnull().sum()
df.saletype.value_counts()

#salecondition: sale condition
df.salecondition.isnull().sum() #none
df.salecondition.value_counts()