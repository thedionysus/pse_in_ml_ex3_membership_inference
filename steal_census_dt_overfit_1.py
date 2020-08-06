import pandas as pd

from shadow_data_census import generate_census_shadow, predict_shadow_data, create_in_out_prediction_set_census, create_attack_model
from sklearn.model_selection import train_test_split
from decision_tree import decision_tree_overfit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("data/census-data.csv")

dataset.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
                   'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'over_50']

dataset = dataset.dropna()
dataset.astype({'over_50': 'str'})

Y = dataset['over_50']
dataset = dataset.drop(['over_50'], axis=1)

dataset = pd.get_dummies(dataset)
dataset['over_50'] = Y

X = dataset.drop(['over_50'], axis=1)



# x_train and y_train will not be used. We split the dataset so that we can use some of the examples for predictions later on
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model_overfit = decision_tree_overfit(X, Y)

# Generating the shadow data
# -----------------------------------------------------------------------------------------
shadow_data_train_1 = generate_census_shadow(dataset, 1000)
shadow_data_test_1 = generate_census_shadow(dataset, 1000)
# -----------------------------------------------------------------------------------------

# split the data into x and y for using in the model
# -----------------------------------------------------------------------------------------
shadow_data_train_labels_1 = shadow_data_train_1['over_50']
shadow_data_train_1 = shadow_data_train_1.drop(['over_50'], axis=1)

shadow_data_test_labels_1 = shadow_data_test_1['over_50']
shadow_data_test_1 = shadow_data_test_1.drop(['over_50'], axis=1)

shadow_model_1 = decision_tree_overfit(shadow_data_train_1, shadow_data_train_labels_1)
# -----------------------------------------------------------------------------------------


# Predict the shadow data and create the prediction set that is used as a dataset for the attack model
# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------
shadow_data_train_predict_1, shadow_data_test_predict_1 = predict_shadow_data(shadow_data_train_1, shadow_data_test_1,
                                                                              shadow_model_1, 1000)

in_prediction_set_1, out_prediction_set_1 = create_in_out_prediction_set_census(shadow_data_train_predict_1,
                                                                                shadow_data_train_labels_1,
                                                                                shadow_data_test_predict_1,
                                                                                shadow_data_test_labels_1)

# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

# Join the in and out data and train the model
# -----------------------------------------------------------------------------------------
prediction_set = pd.concat([in_prediction_set_1, out_prediction_set_1])
prediction_set_over = prediction_set.loc[prediction_set['class_label'].str.contains(">50K")]
prediction_set_under = prediction_set.loc[prediction_set['class_label'].str.contains("<=50K")]


y_prediction_set_over = prediction_set_over['in/out']
x_prediction_set_over = prediction_set_over.drop(['in/out'], axis=1)
#
attack_model_over = DecisionTreeClassifier()
attack_model_over = create_attack_model(attack_model_over, x_prediction_set_over,
                                             y_prediction_set_over)
#
y_prediction_set_under = prediction_set_under['in/out']
x_prediction_set_under = prediction_set_under.drop(['in/out'], axis=1)
#
attack_model_under = DecisionTreeClassifier()
attack_model_under = create_attack_model(attack_model_under, x_prediction_set_under, y_prediction_set_under)
# -----------------------------------------------------------------------------------------

no_test_samples = y_test.shape[0]
real_data_predictions = model_overfit.predict(x_test)
y_test = y_test.to_numpy()

in_set = pd.DataFrame({"prediction": real_data_predictions, "class_label": y_test})
in_set["in/out"] = "in"

out_shadow = generate_census_shadow(dataset, no_test_samples)
y_out_shadow = out_shadow['over_50']
x_out_shadow = out_shadow.drop(['over_50'], axis=1)


shadow_data_predictions = pd.DataFrame(data=model_overfit.predict(x_out_shadow), index=range(0, no_test_samples),
                                       columns=["prediction"])
out_set = pd.concat([shadow_data_predictions, y_out_shadow], axis=1, sort=False)
out_set["in/out"] = "out"
out_set = out_set.rename(columns={"prediction": "prediction", "over_50": "class_label", "in/out": "in/out"})


final_set = pd.concat([in_set, out_set])
final_set_over = final_set.loc[final_set['class_label'].str.contains(">50K")]
final_set_under = final_set.loc[final_set['class_label'].str.contains("<=50K")]
print(final_set.shape)


y_final_set_over = final_set_over['in/out']
x_final_set_over = final_set_over.drop(['in/out'], axis=1)
x_final_set_over = pd.get_dummies(x_final_set_over)

y_final_set_under = final_set_under['in/out']
x_final_set_under = final_set_under.drop(['in/out'], axis=1)
x_final_set_under = pd.get_dummies(x_final_set_under)

pred_final_set_over = attack_model_over.predict(x_final_set_over)
pred_final_set_under = attack_model_under.predict(x_final_set_under)


print("Results for over 50 class")
print("-----------------------------")
print(accuracy_score(y_final_set_over, pred_final_set_over))
print(classification_report(y_final_set_over, pred_final_set_over))
print("-----------------------------")


print("Results for under 50 class")
print("-----------------------------")
print(accuracy_score(y_final_set_under, pred_final_set_under))
print(classification_report(y_final_set_under, pred_final_set_under))
print("-----------------------------")
