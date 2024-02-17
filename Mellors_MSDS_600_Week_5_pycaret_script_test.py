import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(C):
      churn_predictions_df = pd.read_csv('C:/Users/kmell/Regis_MSDS/MSDS_600_Intro_to_DS/MSDS_600_Wk5/new_churn_data.csv', index_col='customerID')
      return churn_predictions_df


def make_predictions(churn_df):
    classifier = ClassificationExperiment()
    model = classifier.load_model('C:/Users/kmell/Regis_MSDS/MSDS_600_Intro_to_DS/MSDS_600_Wk5/churn_pycaret_model')
    predictions = classifier.predict_model(model, data=churn_df)
    predictions.rename({'prediction_label': 'Prediction'}, axis=1, inplace=True)
    predictions['Prediction'].replace({'Yes': 'Churn', 'No': 'No Churn'}, inplace=True)
    return predictions[['Prediction']]


if __name__ == "__main__":
    churn_predictions_df = load_data('../new_churn_data.csv')
    predictions = make_predictions(churn_predictions_df)
    print('Churn Predictions:')
    print(predictions)

