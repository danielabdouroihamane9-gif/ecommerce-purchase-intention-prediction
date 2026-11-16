from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'raw' / 'online_shoppers_intention.csv'
MODEL_PATH = ROOT / 'models' / 'purchase_intention_model.joblib'
METADATA_PATH = ROOT / 'models' / 'purchase_intention_model_metadata.json'
TARGET_COLUMN = 'Revenue'
EXCLUDED_FEATURES = ['PageValues']

FEATURE_COLUMNS = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'SpecialDay',
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend',
]


def build_pipeline() -> Pipeline:
    numeric_features = [
        'Administrative',
        'Administrative_Duration',
        'Informational',
        'Informational_Duration',
        'ProductRelated',
        'ProductRelated_Duration',
        'BounceRates',
        'ExitRates',
        'SpecialDay',
        'OperatingSystems',
        'Browser',
        'Region',
        'TrafficType',
    ]
    categorical_features = ['Month', 'VisitorType', 'Weekend']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features),
    ])

    model = HistGradientBoostingClassifier(random_state=42)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    return pipeline


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'{TARGET_COLUMN} column is missing from the dataset.')

    for feature in EXCLUDED_FEATURES:
        if feature in df.columns:
            print(f'Excluding leakage-prone feature: {feature}')

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    param_grid = {
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.03, 0.05, 0.08],
        'model__max_iter': [200, 400],
        'model__min_samples_leaf': [10, 20],
        'model__l2_regularization': [0.0, 0.1],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }

    print('Best params:', search.best_params_)
    print(json.dumps({
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_metrics': metrics,
    }, indent=2))
    print(classification_report(y_test, y_pred, target_names=['No purchase', 'Purchase']))

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        'target_column': TARGET_COLUMN,
        'excluded_features': EXCLUDED_FEATURES,
        'feature_columns': FEATURE_COLUMNS,
        'best_params': search.best_params_,
        'test_metrics': metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Model saved to {MODEL_PATH}')
    print(f'Metadata saved to {METADATA_PATH}')


if __name__ == '__main__':
    main()
