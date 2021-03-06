import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from decision_tree import decision_tree_overfit
from shadow_data_cancer import generate_cancer_shadow, predict_shadow_data, create_in_out_prediction_set_cancer, \
    create_attack_model

dataset = pd.read_csv("data/cancer-data.csv")
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

Y = dataset['diagnosis']
X = dataset.drop(['diagnosis'], axis=1)

# x_train and y_train will not be used. We split the dataset so that we can use some of the examples for predictions later on
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model_overfit = decision_tree_overfit(X, Y)

# Generating the shadow data
# -----------------------------------------------------------------------------------------
shadow_data_train_1 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_1 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_2 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_2 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_3 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_3 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_4 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_4 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_5 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_5 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_6 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_6 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_7 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_7 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_8 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_8 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_9 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_9 = generate_cancer_shadow(dataset, 1000)

shadow_data_train_10 = generate_cancer_shadow(dataset, 1000)
shadow_data_test_10 = generate_cancer_shadow(dataset, 1000)
# -----------------------------------------------------------------------------------------

# split the data into x and y for using in the model
# -----------------------------------------------------------------------------------------
shadow_data_train_labels_1 = shadow_data_train_1['diagnosis']
shadow_data_train_1 = shadow_data_train_1.drop(['diagnosis'], axis=1)

shadow_data_train_labels_2 = shadow_data_train_2['diagnosis']
shadow_data_train_2 = shadow_data_train_2.drop(['diagnosis'], axis=1)

shadow_data_train_labels_3 = shadow_data_train_3['diagnosis']
shadow_data_train_3 = shadow_data_train_3.drop(['diagnosis'], axis=1)

shadow_data_train_labels_4 = shadow_data_train_4['diagnosis']
shadow_data_train_4 = shadow_data_train_4.drop(['diagnosis'], axis=1)

shadow_data_train_labels_5 = shadow_data_train_5['diagnosis']
shadow_data_train_5 = shadow_data_train_5.drop(['diagnosis'], axis=1)

shadow_data_train_labels_6 = shadow_data_train_6['diagnosis']
shadow_data_train_6 = shadow_data_train_6.drop(['diagnosis'], axis=1)

shadow_data_train_labels_7 = shadow_data_train_7['diagnosis']
shadow_data_train_7 = shadow_data_train_7.drop(['diagnosis'], axis=1)

shadow_data_train_labels_8 = shadow_data_train_8['diagnosis']
shadow_data_train_8 = shadow_data_train_8.drop(['diagnosis'], axis=1)

shadow_data_train_labels_9 = shadow_data_train_9['diagnosis']
shadow_data_train_9 = shadow_data_train_9.drop(['diagnosis'], axis=1)

shadow_data_train_labels_10 = shadow_data_train_10['diagnosis']
shadow_data_train_10 = shadow_data_train_10.drop(['diagnosis'], axis=1)

# -----------------------------------------------------------------------------------------

shadow_data_test_labels_1 = shadow_data_test_1['diagnosis']
shadow_data_test_1 = shadow_data_test_1.drop(['diagnosis'], axis=1)

shadow_data_test_labels_2 = shadow_data_test_2['diagnosis']
shadow_data_test_2 = shadow_data_test_2.drop(['diagnosis'], axis=1)

shadow_data_test_labels_3 = shadow_data_test_3['diagnosis']
shadow_data_test_3 = shadow_data_test_3.drop(['diagnosis'], axis=1)

shadow_data_test_labels_4 = shadow_data_test_4['diagnosis']
shadow_data_test_4 = shadow_data_test_4.drop(['diagnosis'], axis=1)

shadow_data_test_labels_5 = shadow_data_test_5['diagnosis']
shadow_data_test_5 = shadow_data_test_5.drop(['diagnosis'], axis=1)

shadow_data_test_labels_6 = shadow_data_test_6['diagnosis']
shadow_data_test_6 = shadow_data_test_6.drop(['diagnosis'], axis=1)

shadow_data_test_labels_7 = shadow_data_test_7['diagnosis']
shadow_data_test_7 = shadow_data_test_7.drop(['diagnosis'], axis=1)

shadow_data_test_labels_8 = shadow_data_test_8['diagnosis']
shadow_data_test_8 = shadow_data_test_8.drop(['diagnosis'], axis=1)

shadow_data_test_labels_9 = shadow_data_test_9['diagnosis']
shadow_data_test_9 = shadow_data_test_9.drop(['diagnosis'], axis=1)

shadow_data_test_labels_10 = shadow_data_test_10['diagnosis']
shadow_data_test_10 = shadow_data_test_10.drop(['diagnosis'], axis=1)

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

in_prediction_set_1, out_prediction_set_1 = create_in_out_prediction_set_cancer(shadow_data_train_predict_1,
                                                                                shadow_data_train_labels_1,
                                                                                shadow_data_test_predict_1,
                                                                                shadow_data_test_labels_1)

in_prediction_set_2, out_prediction_set_2 = create_in_out_prediction_set_cancer(shadow_data_train_predict_2,
                                                                                shadow_data_train_labels_2,
                                                                                shadow_data_test_predict_2,
                                                                                shadow_data_test_labels_2)
in_prediction_set_3, out_prediction_set_3 = create_in_out_prediction_set_cancer(shadow_data_train_predict_3,
                                                                                shadow_data_train_labels_3,
                                                                                shadow_data_test_predict_3,
                                                                                shadow_data_test_labels_3)

in_prediction_set_4, out_prediction_set_4 = create_in_out_prediction_set_cancer(shadow_data_train_predict_4,
                                                                                shadow_data_train_labels_4,
                                                                                shadow_data_test_predict_4,
                                                                                shadow_data_test_labels_4)

in_prediction_set_5, out_prediction_set_5 = create_in_out_prediction_set_cancer(shadow_data_train_predict_5,
                                                                                shadow_data_train_labels_5,
                                                                                shadow_data_test_predict_5,
                                                                                shadow_data_test_labels_5)

in_prediction_set_6, out_prediction_set_6 = create_in_out_prediction_set_cancer(shadow_data_train_predict_6,
                                                                                shadow_data_train_labels_6,
                                                                                shadow_data_test_predict_6,
                                                                                shadow_data_test_labels_6)

in_prediction_set_7, out_prediction_set_7 = create_in_out_prediction_set_cancer(shadow_data_train_predict_7,
                                                                                shadow_data_train_labels_7,
                                                                                shadow_data_test_predict_7,
                                                                                shadow_data_test_labels_7)
in_prediction_set_8, out_prediction_set_8 = create_in_out_prediction_set_cancer(shadow_data_train_predict_8,
                                                                                shadow_data_train_labels_8,
                                                                                shadow_data_test_predict_8,
                                                                                shadow_data_test_labels_8)

in_prediction_set_9, out_prediction_set_9 = create_in_out_prediction_set_cancer(shadow_data_train_predict_9,
                                                                                shadow_data_train_labels_9,
                                                                                shadow_data_test_predict_9,
                                                                                shadow_data_test_labels_9)

in_prediction_set_10, out_prediction_set_10 = create_in_out_prediction_set_cancer(shadow_data_train_predict_10,
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

prediction_set_malignant = prediction_set.loc[prediction_set['class_label'] == "M"]
prediction_set_benign = prediction_set.loc[prediction_set['class_label'] == "B"]

y_prediction_set_malignant = prediction_set_malignant['in/out']
x_prediction_set_malignant = prediction_set_malignant.drop(['in/out'], axis=1)

attack_model_malignant = decision_tree_overfit(pd.get_dummies(x_prediction_set_malignant), y_prediction_set_malignant)

y_prediction_set_benign = prediction_set_benign['in/out']
x_prediction_set_benign = prediction_set_benign.drop(['in/out'], axis=1)

attack_model_benign = decision_tree_overfit(pd.get_dummies(x_prediction_set_benign), y_prediction_set_benign)
# -----------------------------------------------------------------------------------------

# Predict the test real data stored and store the results as in
# Predict some more shadow data and store the results as out
# -----------------------------------------------------------------------------------------
no_test_samples = y_test.shape[0]
real_data_predictions = model_overfit.predict(x_test)
y_test = y_test.to_numpy()

in_set = pd.DataFrame({"prediction": real_data_predictions, "class_label": y_test})
in_set["in/out"] = "in"

out_shadow = generate_cancer_shadow(dataset, no_test_samples)
y_out_shadow = out_shadow['diagnosis']
x_out_shadow = out_shadow.drop(['diagnosis'], axis=1)

shadow_data_predictions = pd.DataFrame(data=model_overfit.predict(x_out_shadow), index=range(0, no_test_samples),
                                       columns=["prediction"])
out_set = pd.concat([shadow_data_predictions, y_out_shadow], axis=1, sort=False)
out_set["in/out"] = "out"
out_set = out_set.rename(columns={"prediction": "prediction", "diagnosis": "class_label", "in/out": "in/out"})

final_set = pd.concat([in_set, out_set])
final_set_benign = final_set.loc[final_set['class_label'] == "B"]
final_set_malignant = final_set.loc[final_set['class_label'] == "M"]
print(final_set.shape)

y_final_set_benign = final_set_benign['in/out']
x_final_set_benign = final_set_benign.drop(['in/out'], axis=1)
x_final_set_benign = pd.get_dummies(x_final_set_benign)

y_final_set_malignant = final_set_malignant['in/out']
x_final_set_malignant = final_set_malignant.drop(['in/out'], axis=1)
x_final_set_malignant = pd.get_dummies(x_final_set_malignant)


pred_final_set_benign = attack_model_benign.predict(x_final_set_benign)
pred_final_set_malignant = attack_model_malignant.predict(x_final_set_malignant)

print("Results for benign class")
print("-----------------------------")
print(accuracy_score(y_final_set_benign, pred_final_set_benign))
print(classification_report(y_final_set_benign, pred_final_set_benign))
print("-----------------------------")


print("Results for malignant class")
print("-----------------------------")
print(accuracy_score(y_final_set_malignant, pred_final_set_malignant))
print(classification_report(y_final_set_malignant, pred_final_set_malignant))
print("-----------------------------")