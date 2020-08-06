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

shadow_data_train_2 = generate_census_shadow(dataset, 1000)
shadow_data_test_2 = generate_census_shadow(dataset, 1000)

shadow_data_train_3 = generate_census_shadow(dataset, 1000)
shadow_data_test_3 = generate_census_shadow(dataset, 1000)

shadow_data_train_4 = generate_census_shadow(dataset, 1000)
shadow_data_test_4 = generate_census_shadow(dataset, 1000)

shadow_data_train_5 = generate_census_shadow(dataset, 1000)
shadow_data_test_5 = generate_census_shadow(dataset, 1000)

shadow_data_train_6 = generate_census_shadow(dataset, 1000)
shadow_data_test_6 = generate_census_shadow(dataset, 1000)

shadow_data_train_7 = generate_census_shadow(dataset, 1000)
shadow_data_test_7 = generate_census_shadow(dataset, 1000)

shadow_data_train_8 = generate_census_shadow(dataset, 1000)
shadow_data_test_8 = generate_census_shadow(dataset, 1000)

shadow_data_train_9 = generate_census_shadow(dataset, 1000)
shadow_data_test_9 = generate_census_shadow(dataset, 1000)

shadow_data_train_10 = generate_census_shadow(dataset, 1000)
shadow_data_test_10 = generate_census_shadow(dataset, 1000)
# -----------------------------------------------------------------------------------------

# split the data into x and y for using in the model
# -----------------------------------------------------------------------------------------
shadow_data_train_labels_1 = shadow_data_train_1['over_50']
shadow_data_train_1 = shadow_data_train_1.drop(['over_50'], axis=1)

shadow_data_train_labels_2 = shadow_data_train_2['over_50']
shadow_data_train_2 = shadow_data_train_2.drop(['over_50'], axis=1)

shadow_data_train_labels_3 = shadow_data_train_3['over_50']
shadow_data_train_3 = shadow_data_train_3.drop(['over_50'], axis=1)

shadow_data_train_labels_4 = shadow_data_train_4['over_50']
shadow_data_train_4 = shadow_data_train_4.drop(['over_50'], axis=1)

shadow_data_train_labels_5 = shadow_data_train_5['over_50']
shadow_data_train_5 = shadow_data_train_5.drop(['over_50'], axis=1)

shadow_data_train_labels_6 = shadow_data_train_6['over_50']
shadow_data_train_6 = shadow_data_train_6.drop(['over_50'], axis=1)

shadow_data_train_labels_7 = shadow_data_train_7['over_50']
shadow_data_train_7 = shadow_data_train_7.drop(['over_50'], axis=1)

shadow_data_train_labels_8 = shadow_data_train_8['over_50']
shadow_data_train_8 = shadow_data_train_8.drop(['over_50'], axis=1)

shadow_data_train_labels_9 = shadow_data_train_9['over_50']
shadow_data_train_9 = shadow_data_train_9.drop(['over_50'], axis=1)

shadow_data_train_labels_10 = shadow_data_train_10['over_50']
shadow_data_train_10 = shadow_data_train_10.drop(['over_50'], axis=1)

# ----------------------------------------------------------------------------------------FF-

shadow_data_test_labels_1 = shadow_data_test_1['over_50']
shadow_data_test_1 = shadow_data_test_1.drop(['over_50'], axis=1)

shadow_data_test_labels_2 = shadow_data_test_2['over_50']
shadow_data_test_2 = shadow_data_test_2.drop(['over_50'], axis=1)

shadow_data_test_labels_3 = shadow_data_test_3['over_50']
shadow_data_test_3 = shadow_data_test_3.drop(['over_50'], axis=1)

shadow_data_test_labels_4 = shadow_data_test_4['over_50']
shadow_data_test_4 = shadow_data_test_4.drop(['over_50'], axis=1)

shadow_data_test_labels_5 = shadow_data_test_5['over_50']
shadow_data_test_5 = shadow_data_test_5.drop(['over_50'], axis=1)

shadow_data_test_labels_6 = shadow_data_test_6['over_50']
shadow_data_test_6 = shadow_data_test_6.drop(['over_50'], axis=1)

shadow_data_test_labels_7 = shadow_data_test_7['over_50']
shadow_data_test_7 = shadow_data_test_7.drop(['over_50'], axis=1)

shadow_data_test_labels_8 = shadow_data_test_8['over_50']
shadow_data_test_8 = shadow_data_test_8.drop(['over_50'], axis=1)

shadow_data_test_labels_9 = shadow_data_test_9['over_50']
shadow_data_test_9 = shadow_data_test_9.drop(['over_50'], axis=1)

shadow_data_test_labels_10 = shadow_data_test_10['over_50']
shadow_data_test_10 = shadow_data_test_10.drop(['over_50'], axis=1)

# -----------------------------------------------------------------------------------------
shadow_model_1 = decision_tree_overfit(shadow_data_train_1, shadow_data_train_labels_1)
shadow_model_2 = decision_tree_overfit(shadow_data_train_2, shadow_data_train_labels_2)
shadow_model_3 = decision_tree_overfit(shadow_data_train_3, shadow_data_train_labels_3)
shadow_model_4 = decision_tree_overfit(shadow_data_train_4, shadow_data_train_labels_4)
shadow_model_5 = decision_tree_overfit(shadow_data_train_5, shadow_data_train_labels_5)
shadow_model_6 = decision_tree_overfit(shadow_data_train_6, shadow_data_train_labels_6)
shadow_model_7 = decision_tree_overfit(shadow_data_train_7, shadow_data_train_labels_7)
shadow_model_8 = decision_tree_overfit(shadow_data_train_8, shadow_data_train_labels_8)
shadow_model_9 = decision_tree_overfit(shadow_data_train_9, shadow_data_train_labels_9)
shadow_model_10 = decision_tree_overfit(shadow_data_train_10, shadow_data_train_labels_10)
# -----------------------------------------------------------------------------------------


# Predict the shadow data and create the prediction set that is used as a dataset for the attack model
# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------
shadow_data_train_predict_1, shadow_data_test_predict_1 = predict_shadow_data(shadow_data_train_1, shadow_data_test_1,
                                                                              shadow_model_1, 1000)

shadow_data_train_predict_2, shadow_data_test_predict_2 = predict_shadow_data(shadow_data_train_2, shadow_data_test_2,
                                                                              shadow_model_2, 1000)

shadow_data_train_predict_3, shadow_data_test_predict_3 = predict_shadow_data(shadow_data_train_3, shadow_data_test_3,
                                                                              shadow_model_3, 1000)

shadow_data_train_predict_4, shadow_data_test_predict_4 = predict_shadow_data(shadow_data_train_4, shadow_data_test_4,
                                                                              shadow_model_4, 1000)

shadow_data_train_predict_5, shadow_data_test_predict_5 = predict_shadow_data(shadow_data_train_5, shadow_data_test_5,
                                                                              shadow_model_5, 1000)

shadow_data_train_predict_6, shadow_data_test_predict_6 = predict_shadow_data(shadow_data_train_6, shadow_data_test_6,
                                                                              shadow_model_6, 1000)

shadow_data_train_predict_7, shadow_data_test_predict_7 = predict_shadow_data(shadow_data_train_7, shadow_data_test_7,
                                                                              shadow_model_7, 1000)

shadow_data_train_predict_8, shadow_data_test_predict_8 = predict_shadow_data(shadow_data_train_8, shadow_data_test_8,
                                                                              shadow_model_8, 1000)

shadow_data_train_predict_9, shadow_data_test_predict_9 = predict_shadow_data(shadow_data_train_9, shadow_data_test_9,
                                                                              shadow_model_9, 1000)

shadow_data_train_predict_10, shadow_data_test_predict_10 = predict_shadow_data(shadow_data_train_10, shadow_data_test_10,
                                                                              shadow_model_10, 1000)

# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

in_prediction_set_1, out_prediction_set_1 = create_in_out_prediction_set_census(shadow_data_train_predict_1,
                                                                                shadow_data_train_labels_1,
                                                                                shadow_data_test_predict_1,
                                                                                shadow_data_test_labels_1)

in_prediction_set_2, out_prediction_set_2 = create_in_out_prediction_set_census(shadow_data_train_predict_2,
                                                                                shadow_data_train_labels_2,
                                                                                shadow_data_test_predict_2,
                                                                                shadow_data_test_labels_2)
in_prediction_set_3, out_prediction_set_3 = create_in_out_prediction_set_census(shadow_data_train_predict_3,
                                                                                shadow_data_train_labels_3,
                                                                                shadow_data_test_predict_3,
                                                                                shadow_data_test_labels_3)

in_prediction_set_4, out_prediction_set_4 = create_in_out_prediction_set_census(shadow_data_train_predict_4,
                                                                                shadow_data_train_labels_4,
                                                                                shadow_data_test_predict_4,
                                                                                shadow_data_test_labels_4)

in_prediction_set_5, out_prediction_set_5 = create_in_out_prediction_set_census(shadow_data_train_predict_5,
                                                                                shadow_data_train_labels_5,
                                                                                shadow_data_test_predict_5,
                                                                                shadow_data_test_labels_5)

in_prediction_set_6, out_prediction_set_6 = create_in_out_prediction_set_census(shadow_data_train_predict_6,
                                                                                shadow_data_train_labels_6,
                                                                                shadow_data_test_predict_6,
                                                                                shadow_data_test_labels_6)

in_prediction_set_7, out_prediction_set_7 = create_in_out_prediction_set_census(shadow_data_train_predict_7,
                                                                                shadow_data_train_labels_7,
                                                                                shadow_data_test_predict_7,
                                                                                shadow_data_test_labels_7)
in_prediction_set_8, out_prediction_set_8 = create_in_out_prediction_set_census(shadow_data_train_predict_8,
                                                                                shadow_data_train_labels_8,
                                                                                shadow_data_test_predict_8,
                                                                                shadow_data_test_labels_8)

in_prediction_set_9, out_prediction_set_9 = create_in_out_prediction_set_census(shadow_data_train_predict_9,
                                                                                shadow_data_train_labels_9,
                                                                                shadow_data_test_predict_9,
                                                                                shadow_data_test_labels_9)

in_prediction_set_10, out_prediction_set_10 = create_in_out_prediction_set_census(shadow_data_train_predict_10,
                                                                                shadow_data_train_labels_10,
                                                                                shadow_data_test_predict_10,
                                                                                shadow_data_test_labels_10)

# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

# Join the in and out data and train the model
# -----------------------------------------------------------------------------------------
prediction_set = pd.concat([in_prediction_set_1, out_prediction_set_1, in_prediction_set_2,
                            out_prediction_set_2, in_prediction_set_3, out_prediction_set_3,
                            in_prediction_set_4, out_prediction_set_4, in_prediction_set_5,
                            out_prediction_set_5, in_prediction_set_6, out_prediction_set_6,
                            in_prediction_set_7, out_prediction_set_7, in_prediction_set_8,
                            out_prediction_set_8, in_prediction_set_9, out_prediction_set_9,
                            in_prediction_set_10, out_prediction_set_10])

prediction_set_over = prediction_set.loc[prediction_set['class_label'].str.contains(">50K")]
prediction_set_under = prediction_set.loc[prediction_set['class_label'].str.contains("<=50K")]


y_prediction_set_over = prediction_set_over['in/out']
x_prediction_set_over = prediction_set_over.drop(['in/out'], axis=1)

attack_model_over = decision_tree_overfit(pd.get_dummies(x_prediction_set_over), y_prediction_set_over)

y_prediction_set_under = prediction_set_under['in/out']
x_prediction_set_under = prediction_set_under.drop(['in/out'], axis=1)


attack_model_under = decision_tree_overfit(pd.get_dummies(x_prediction_set_under), y_prediction_set_under)
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
