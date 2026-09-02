from src.utils.config import read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


# Load configuration
config = read_yaml("config.yaml")

# Load data
ingestion = DataIngestion(config)
data = ingestion.load_data()

required_columns = config["validation"]["required_columns"]
expected_dtypes = config["validation"]["expected_dtypes"]

# # Define required columns
# required_columns = [
#     "customerID",
#     "gender",
#     "SeniorCitizen",
#     "Partner",
#     "Dependents",
#     "tenure",
#     "PhoneService",
#     "MultipleLines",
#     "InternetService",
#     "OnlineSecurity",
#     "OnlineBackup",
#     "DeviceProtection",
#     "TechSupport",
#     "StreamingTV",
#     "StreamingMovies",
#     "Contract",
#     "PaperlessBilling",
#     "PaymentMethod",
#     "MonthlyCharges",
#     "TotalCharges",
#     "Churn"
# ]

# Create validator
validator = DataValidation(required_columns, expected_dtypes)

# Run validation
report = validator.validate_data(data)

# Print results
print("Validation completed successfully.\n")

print("\nIncorrect Data Types:")
print(report["incorrect_dtypes"])

print("Missing Columns:")
print(report["missing_columns"])

print("\nDuplicate Rows:")
print(report["duplicate_count"])

print("\nMissing Values:")
print(report["missing_values"])