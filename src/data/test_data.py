import sys
import pandas as pd

from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import os

preprocessed_dir = "data/preprocessed/air"
reference_dir = "data/reference/air"
report_dir = "reports/tests"
os.makedirs(reference_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

all_tests_passed = True

# Foreach station Load the reference and current data
for filename in os.listdir(preprocessed_dir):
    if filename.endswith(".csv"):
        station_id = os.path.splitext(filename)[0]
        current_path = os.path.join(preprocessed_dir, filename)
        reference_path = os.path.join(reference_dir, filename)
        report_path = os.path.join(report_dir, f"data_testing_report_{station_id}.html")

        current = pd.read_csv(current_path)

        if not os.path.exists(reference_path):
            print(f"[{station_id}] Reference file not found. Copying from current data to {reference_path}.")
            current.to_csv(reference_path, index=False)

        reference = pd.read_csv(reference_path)

        # Remove 'Date_to' column if present
        for df in [reference, current]:
            if "Date_to" in df.columns:
                del df["Date_to"]
                
        # Skip if all values are missing
        if current.dropna(how="all").empty:
            print(f"[{station_id}] Skipping file: all values are missing.")
            continue

        # Check if the reference and current data have the same columns
        report = Report(
            [DataSummaryPreset(), DataDriftPreset()],
            include_tests=True
        )

        # Run the report on the reference and current data
        result = report.run(reference_data=reference, current_data=current)
        
        # Save the report to an HTML file
        result.save_html(report_path)

        # Check if the report contains any tests and if all tests passed
        result_dict = result.dict()
        if "tests" in result_dict:
            for test in result_dict["tests"]:
                if "status" in test and test["status"] != "SUCCESS":
                    print(f"[{station_id}] Data tests failed.")
                    all_tests_passed = False
                    break
        else:
            print(f"[{station_id}] No tests found in result.")
            all_tests_passed = False

        if all_tests_passed:
            print(f"[{station_id}] Data tests passed. Updating reference.")
            # Replace the reference data with the current data
            current.to_csv(reference_path, index=False)

if not all_tests_passed:
    print("At least one station failed data validation.")
    sys.exit(1)
else:
    print("All station data tests passed.")
    sys.exit(0)