from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import shap
import uvicorn
import re

app = FastAPI()

# === Fonctions de prétraitement ===
def clean_columns(df):
    df.columns = [re.sub(r'\W+', '_', col) for col in df.columns]
    return df

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

# --- Chargement du modèle et des données ---
model = joblib.load('pipeline_complete.pkl')
print("Type du modèle :", type(model))

data = pd.read_csv('data_test.csv', sep=',')
data_train = pd.read_csv('data_train.csv', sep=',')

# Nettoyage cohérent avec l'entraînement
data = encode_categorical_columns(clean_columns(data))
data_train = encode_categorical_columns(clean_columns(data_train))

# Extract the final classifier from pipeline
classifier = model.named_steps['model']

# Explainer SHAP (adapter selon ton modèle, TreeExplainer si arbre)
explainer = shap.Explainer(classifier)

@app.get('/')
def welcome():
    return {"message": "Welcome to the API"}

@app.get('/{client_id}')
def check_client_id(client_id: int):
    # Vérifie si client_id existe dans les données
    return client_id in data['SK_ID_CURR'].values.tolist()

@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    try:
        client_data = data[data['SK_ID_CURR'] == client_id]
        if client_data.empty:
            return {"error": "Client ID not found"}

        info_client = client_data.drop(columns=['TARGET'], errors='ignore')
        prediction = model.predict_proba(info_client)[0][1]

        return {
            "client_id": client_id,
            "probability_default": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}

@app.get('/interpretabilite/{client_id}')
def get_shap_values(client_id: int):
    try:
        client_data = data[data['SK_ID_CURR'] == client_id]
        if client_data.empty:
            return {"error": "Client ID not found"}

        individual = client_data.drop(columns=['TARGET'], errors='ignore')

        # SHAP local
        shap_values_individual = explainer(individual)
        shap_values_df = pd.DataFrame(shap_values_individual.values, columns=individual.columns)
        shap_values_dict = shap_values_df.to_dict(orient='records')[0]
        shap_values_dict.pop("SK_ID_CURR", None)

        # SHAP global (moyenne des valeurs absolues sur tout le jeu data)
        data_no_target = data.drop(columns=['TARGET'], errors='ignore')
        shap_values_all = explainer(data_no_target)
        shap_values_all_array = np.abs(shap_values_all.values)
        mean_shap_global = shap_values_all_array.mean(axis=0)
        shap_global_dict = dict(zip(data_no_target.columns, mean_shap_global))
        shap_global_dict.pop("SK_ID_CURR", None)

        return {
            "shap_values": shap_values_dict,
            "shap_global": shap_global_dict
        }

    except Exception as e:
        return {"error": str(e)}

@app.get('/feature_importance')
def get_feature_importance():
    try:
        X_train = data_train.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

        shap_values = explainer(X_train)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        features = X_train.columns

        importance = dict(sorted(zip(features, mean_abs_shap), key=lambda x: x[1], reverse=True)[:20])
        return {"feature_importance": importance}

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
