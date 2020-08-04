import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from decision_tree import decision_tree_overfit
from shadow_data_utils import generate_cancer_shadow, predict_shadow_data, create_in_out_prediction_set_cancer, create_attack_model



dataset = pd.read_csv("data/cancer-data.csv")
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

Y = dataset['diagnosis']
X = dataset.drop(['diagnosis'], axis=1)


#x_train and y_train will not be used. We split the dataset so that we can use some of the examples for predictions later on
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model = decision_tree_overfit(X, Y)

#Generating the shadow data
#-----------------------------------------------------------------------------------------
shadow_data_train = generate_cancer_shadow(dataset, 100)
shadow_data_test = generate_cancer_shadow(dataset, 100)
#-----------------------------------------------------------------------------------------

#split the data into x and y for using in the model
#-----------------------------------------------------------------------------------------
shadow_data_train_labels = shadow_data_train['diagnosis']
shadow_data_train = shadow_data_train.drop(['diagnosis'], axis=1)

shadow_data_test_labels = shadow_data_test['diagnosis']
shadow_data_test = shadow_data_test.drop(['diagnosis'], axis=1)
#-----------------------------------------------------------------------------------------


#Predict the shadow data and create the prediction set that is used as a dataset for the attack model
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
shadow_data_train_predict, shadow_data_test_predict = predict_shadow_data(shadow_data_train, shadow_data_test, model, 100)

in_prediction_set, out_prediction_set = create_in_out_prediction_set_cancer(shadow_data_train_predict, shadow_data_train_labels,
                                                                            shadow_data_test_predict, shadow_data_test_labels)

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

#Join the in and out data and train the model
#TO BE FIXED: create a model for each class
#-----------------------------------------------------------------------------------------
prediction_set = pd.concat([in_prediction_set, out_prediction_set])

y_prediction_set = prediction_set['in/out']
x_prediction_set = prediction_set.drop(['in/out'], axis=1)

attack_model = DecisionTreeClassifier()
attack_model = create_attack_model(attack_model, x_prediction_set, y_prediction_set)
#-----------------------------------------------------------------------------------------

#Predict the test real data stored and store the results as in
#Predict some more shadow data and store the results as out
#-----------------------------------------------------------------------------------------
no_test_samples = y_test.shape[0]
real_data_predictions =model.predict(x_test)
y_test = y_test.to_numpy()

in_set = pd.DataFrame({"prediction": real_data_predictions, "class_label": y_test})
in_set["in/out"] = "in"


out_shadow = generate_cancer_shadow(dataset, no_test_samples)
y_out_shadow = out_shadow['diagnosis']
x_out_shadow = out_shadow.drop(['diagnosis'], axis=1)

shadow_data_predictions = pd.DataFrame(data=model.predict(x_out_shadow), index=range(0,no_test_samples), columns=["prediction"])
out_set = pd.concat([shadow_data_predictions, y_out_shadow], axis=1, sort=False)
out_set["in/out"] = "out"
out_set = out_set.rename(columns={"prediction": "prediction", "diagnosis":"class_label", "in/out":"in/out"})


final_set = pd.concat([in_set, out_set])
print(final_set.shape)

y_final_set = final_set['in/out']
x_final_set = final_set.drop(['in/out'], axis=1)
x_final_set = pd.get_dummies(x_final_set)

pred_final_set = attack_model.predict(x_final_set)

print(accuracy_score(y_final_set, pred_final_set))
print(classification_report(y_final_set, pred_final_set))
