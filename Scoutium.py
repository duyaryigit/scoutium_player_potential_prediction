################################################################
# CLASSIFICATION OF TALENT HUNTING WITH ARTIFICIAL LEARNING
################################################################

################################################################
# Business Problem:
################################################################

# Predicting which class (average, highlighted) players are based on the scores given to the characteristics of the footballers tracked by scouts.

################################################################
# Dataset Story:
################################################################

# The data set consists of information from Scoutium, which includes the features and scores of the football players evaluated by the scouts according to the characteristics of the footballers observed in the matches.

# attributes: It contains the points that the users who evaluate the players give to the characteristics of each player they watch and evaluate in a match. (Independent variables)

#potential_labels: Contains potential tags from users who rate players, with their final opinions about the players in each match. (target variable)

# 9 Variables, 10730 Observations

################################################################
# Variables:
################################################################

# task_response_id: The set of a scout's assessments of all players on a team's roster in a match.

# match_id: The id of the relevant match.

# evaluator_id: The id of the evaluator(scout).

# player_id: The id of the respective player.

# position_id: The id of the position played by the relevant player in that match.
# 1- Goalkeeper
# 2- Stopper
# 3- rfb (right fullback)
# 4- lt (left tackle)
# 5- Defensive midfielder
# 6- Central midfielder
# 7- Right wing
# 8- Left wing
# 9- Offensive midfielder
# 10- Striker

# analysis_id: A set containing a scout's attribute evaluations of a player in a match.

# attribute_id: The id of each attribute the players were evaluated for.

# attribute_value: The value (points) given to a player's attribute of a scout.

# potential_label: Label indicating the final decision of a scout regarding a player in a match. (target variable)


import itertools
#pip install xgboost
#pip install catboost
#pip install --upgrade dask
import xgboost
import catboost
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 200)
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')

################################################################
# Task 1: Read scoutium_attributes.csv and scoutium_potential_labels.csv files.
################################################################

df = pd.read_csv("datasets/scoutium_attributes.csv", sep=";")
df.head()
df.shape
df2 = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=";")
df2.head()
df2.shape

################################################################
# Task 2: Let's combine the csv files we have read using the merge function. ("task_response_id", 'match_id', 'evaluator_id' "player_id" perform the merge operation over 4 variables.)
################################################################

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

################################################################
# Task 3: Remove the Goalkeeper (1) class in position_id from the dataset.
################################################################

dff = dff[dff["position_id"] != 1]

################################################################
# Task 4: Remove the below_average class in the potential_label from the dataset (the below_average class makes up 1% of the entire dataset)
################################################################

dff["potential_label"].value_counts()/dff["potential_label"].count()

dff = dff[dff["potential_label"] != "below_average"]

################################################################
#Task 5: Create a table from the data set you created using the "pivot_table" function. Manipulate this pivot table with one player per row.
################################################################

# Step 1: Create the pivot table with “player_id”, “position_id” and “potential_label” in the index, “attribute_id” in the columns, and the score given by the scouts to the players “attribute_value” in the values.

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])
pt.head()

#Step 2: Get rid of the index error by using the "reset_index" function and convert the names of the "attribute_id" columns to strings.

pt = pt.reset_index(drop=False)
pt.head()
pt.columns = pt.columns.map(str)

################################################################
# Task 3: Assign numeric variable columns to a list with the name “num_cols”.
################################################################

num_cols = pt.columns[3:]

##################################
# TASK 4: Exploratory Data Analysis
##################################


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1],numeric_only=True).T)

check_df(pt)

##################################
# Step 2: Examine the numeric and categorical variables.
##################################

##################################
# Analysis of Categorical Variables
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)

##################################
# Analysis of Numeric Variables
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pt, col, plot=True)


##################################
# Step 3: Perform target variable analysis with numerical variables.
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)

##################################
# Step 4: Examine the correlation.
##################################

pt[num_cols].corr()

# Correlation Matrix

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# TASK 5: Apply Feature Extraction.
##################################

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt.head()
pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")


for i in pt.columns[3:-6]:
    threshold = pt[i].mean() + pt[i].std()

    lst = pt[i].apply(lambda x: 0 if x < threshold else 1)
    pt[str(i) + "_FLAG"] = lst


flagCols = [col for col in pt.columns if "_FLAG" in col]

pt["counts"] = pt[flagCols].sum(axis=1)

pt["countRatio"] = pt["counts"] / len(flagCols)

pt.head()

pt[pt["counts"] == 0]["potential_label"].value_counts()

pt[pt["counts"] != 0]["potential_label"].value_counts()

################################################################
# Task 6: Express the “potential_label” categories (average, highlighted) numerically using the Label Encoder function.
################################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label","mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)


################################################################
# Task 7: Apply standardScaler to scale the data in all the "num_cols" variables you save.
################################################################

pt.head()
lst = ["counts", "countRatio","min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()

################################################################
# Task 8: Develop a machine learning model that predicts the potential tags of football players with minimum error from the data set we have.
################################################################

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier(verbose=-1))]


for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

# Bonus
def sorting_score():
    list_cvs = []
    for name, model in models:
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
            list_cvs+=[[name,score,cvs]]
    cvs_df=pd.DataFrame(list_cvs,columns=["name","score","cvs"]).sort_values(by="cvs",ascending=False)
    return cvs_df

sorting_score()

################################################################
# Task 9: Perform Hyperparameter Optimization.
################################################################

lgbm_model = LGBMClassifier(random_state=46,force_col_wise=True)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# normal y cv duration: 16.2s
# with scaled y: 13.8s

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
    cvs = cross_val_score(final_model, X, y, scoring=score, cv=10).mean()
    print(score + " score:" + str(cvs))


################################################################
# Task 10: Draw the order of the features using the feature_importance function, which indicates the importance of the variables.
################################################################

# feature importance

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)


for name, model in models[3:]:
    model=model.fit(X,y)
    plot_importance(model, X)