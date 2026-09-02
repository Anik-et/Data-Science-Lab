from src.components.model_evaluator import ModelEvaluator

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineer
from src.components.model_trainer import ModelTrainer


def main():

    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------

    config = read_yaml("config.yaml")


    # --------------------------------------------------
    # 2. Data Ingestion
    # --------------------------------------------------

    ingestion = DataIngestion(config)

    data = ingestion.load_data()

    print(f"Data loaded: {data.shape}")


    # --------------------------------------------------
    # 3. Data Validation
    # --------------------------------------------------

    required_columns = config["validation"]["required_columns"]
    expected_dtypes = config["validation"]["expected_dtypes"]

    validator = DataValidation(
        required_columns,
        expected_dtypes
    )

    validation_report = validator.validate_data(data)

    print("\nValidation Report:")
    print(
        f"Missing columns: "
        f"{validation_report['missing_columns']}"
    )

    print(
        f"Duplicate rows: "
        f"{validation_report['duplicate_count']}"
    )

    print(
        f"Incorrect dtypes: "
        f"{validation_report['incorrect_dtypes']}"
    )


    # --------------------------------------------------
    # 4. Data Transformation
    # --------------------------------------------------

    transformation = DataTransformation()

    data = transformation.transform(data)

    print(
        f"\nData after transformation: "
        f"{data.shape}"
    )


    # --------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------

    X = data.drop(columns=["Churn"])

    y = data["Churn"].map({ "No": 0, "Yes": 1 })


    # --------------------------------------------------
    # 6. Train / Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print( f"\nTraining data: {X_train.shape}" )

    print( f"Testing data: {X_test.shape}" )


    # --------------------------------------------------
    # 7. Feature Engineering / Preprocessing
    # --------------------------------------------------

    feature_engineer = FeatureEngineer()

    X_train_transformed = (
        feature_engineer.fit_transform(X_train)
    )

    X_test_transformed = (
        feature_engineer.transform(X_test)
    )

    print(
        f"\nTransformed training data: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Transformed testing data: "
        f"{X_test_transformed.shape}"
    )


    # --------------------------------------------------
    # 8. Model Training
    # --------------------------------------------------

    trainer = ModelTrainer(config)

    trained_models = trainer.train_all(
        X_train_transformed,
        y_train
    )

    print("\nModels trained:")

    for model_name in trained_models:
        print(f"- {model_name}")


    # --------------------------------------------------
    # 9. Model Evaluation
    # --------------------------------------------------

    evaluator = ModelEvaluator()

    results = evaluator.evaluate_all(
    trained_models,
    X_test_transformed,
    y_test
    )

    print("\nModel Evaluation:")
    print(results.to_string(index=False))

    best_model_row = results.loc[
    results["ROC-AUC"].idxmax()
    ]

    best_model_name = best_model_row["Model"]

    print(
        f"\nBest model based on ROC-AUC: "
        f"{best_model_name}"
    )

    # --------------------------------------------------
    # 10. Save artifacts
    # --------------------------------------------------

    from pathlib import Path
    import joblib

    artifact_directory = Path(
        config["artifacts"]["directory"]
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save best model
    trainer.save_model(
        best_model_name,
        config["artifacts"]["best_model"]
    )

    # Save fitted preprocessor
    joblib.dump(
        feature_engineer.preprocessor,
        config["artifacts"]["preprocessor"]
    )

    # Save feature names
    joblib.dump(
        feature_engineer.get_feature_names(),
        config["artifacts"]["feature_names"]
    )

    print("\nArtifacts saved successfully.")


    # --------------------------------------------------
    # 11. SHAP Analysis
    # --------------------------------------------------

    from src.components.shap_analyzer import SHAPAnalyzer

    if best_model_name in ["random_forest", "xgboost"]:

        shap_analyzer = SHAPAnalyzer()

        best_model = trained_models[best_model_name]

        shap_analyzer.calculate_shap_values(
            best_model,
            X_test_transformed
        )

        shap_analyzer.plot_summary(
            X_test_transformed,
            feature_engineer.get_feature_names(),
            config["shap"]["output_path"]
        )

        print("\nSHAP analysis completed.")


if __name__ == "__main__":
    main()