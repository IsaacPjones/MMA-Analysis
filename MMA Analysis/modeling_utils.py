import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def prepare_features(df):

    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    df['Target'] = (df['Winner'] == 'Red').astype(int)
    
    df['AgeDiff'] = df['RedAge'] - df['BlueAge']
    df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']
    df['HeightDiff'] = df['RedHeightCms'] - df['BlueHeightCms']
    df['RedFights'] = df['RedWins'] + df['RedLosses']
    df['BlueFights'] = df['BlueWins'] + df['BlueLosses']
    df['ExpDiff'] = df['RedFights'] - df['BlueFights']
    
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    else:
        df['Year'] = np.nan
    
    if 'Finish' in df.columns:
        df['Finish'] = df['Finish'].replace({
            'M-DEC': 'DEC',
            'S-DEC': 'DEC',
            'U-DEC': 'DEC'
        })
    else:
        df['Finish'] = np.nan
    
    if 'FinishRound' not in df.columns:
        df['FinishRound'] = np.nan  
    
    numeric_features = ['AgeDiff', 'ReachDiff', 'HeightDiff', 'ExpDiff', 'FinishRound', 'Year']
    categorical_features = ['WeightClass', 'Gender', 'Finish']
    
    df = df.dropna(subset=numeric_features + categorical_features + ['Target'])
    

    X = df[numeric_features + categorical_features].copy()
    y = df['Target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def build_model_pipeline(model):

    numeric_features = ['AgeDiff', 'ReachDiff', 'HeightDiff', 'ExpDiff', 'FinishRound', 'Year']
    categorical_features = ['WeightClass', 'Gender', 'Finish']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, model_name="Model"):
    # generate the classification report and ROC curve and AUC score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n==== {model_name} Classification Report ====")
    print(classification_report(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/roc_{model_name}.png')
    plt.close()
    
    return auc_score

def run_modeling(df):

    X_train, X_test, y_train, y_test = prepare_features(df)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    auc_scores = {}
    for name, clf in models.items():
        pipeline = build_model_pipeline(clf)
        pipeline.fit(X_train, y_train)
        auc = evaluate_model(pipeline, X_test, y_test, name)
        auc_scores[name] = auc
    
    print("\n=== Model AUC Comparison ===")
    for name, auc in auc_scores.items():
        print(f"{name}: AUC = {auc:.4f}")
