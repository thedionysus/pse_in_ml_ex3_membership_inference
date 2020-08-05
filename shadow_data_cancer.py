import numpy as np
import pandas as pd

dataset = pd.read_csv("data/cancer-data.csv")
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

# print(dataset.describe())

def generate_cancer_shadow(dataset, no_samples):
    y = dataset['diagnosis']
    x = dataset.drop(['diagnosis'], axis=1)
    columns = dataset.columns
    malignant_cases = y[y == "M"].count()
    benign_cases = y[y == "B"].count()
    malignant_prob = malignant_cases / (malignant_cases + benign_cases)
    benign_prob = benign_cases / (malignant_cases + benign_cases)
    statistics = x.describe()
    new_data = pd.DataFrame(index=range(0,no_samples), columns=columns)
    for index in range(0,no_samples):
        for column in columns:
            if (column != 'diagnosis'):
                new_data[column][index] = np.random.normal(statistics[column]['mean'], statistics[column]['std'], 1)[0]
            else:
                new_data['diagnosis'][index] = np.random.choice(["M", "B"], p=[malignant_prob, benign_prob])

    return new_data


def predict_shadow_data(shadow_train, shadow_test, model, no_samples):
    shadow_data_train_predict = pd.DataFrame(data=model.predict(shadow_train), index=range(0, no_samples),
                                             columns=["prediction"])
    shadow_data_test_predict = pd.DataFrame(data=model.predict(shadow_test), index=range(0, no_samples),
                                            columns=["prediction"])
    return shadow_data_train_predict, shadow_data_test_predict

def create_in_out_prediction_set_cancer(shadow_data_train_predict, shadow_data_train_labels, shadow_data_test_predict, shadow_data_test_labels):
    in_prediction_set = pd.concat([shadow_data_train_predict, shadow_data_train_labels], axis=1, sort=False)
    in_prediction_set['in/out'] = "in"
    in_prediction_set = in_prediction_set.rename(
        columns={"prediction": "prediction", "diagnosis": "class_label", "in/out": "in/out"})

    out_prediction_set = pd.concat([shadow_data_test_predict, shadow_data_test_labels], axis=1, sort=False)
    out_prediction_set['in/out'] = "out"
    out_prediction_set = out_prediction_set.rename(
        columns={"prediction": "prediction", "diagnosis": "class_label", "in/out": "in/out"})
    return in_prediction_set, out_prediction_set

def create_attack_model(model, x_train, y_train):
    x_prediction_set = pd.get_dummies(x_train)
    attack_model = model.fit(x_prediction_set, y_train)
    return attack_model
