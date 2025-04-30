import sys
import os
import great_expectations as gx

# Debug: print current working directory
print("Current working directory:", os.getcwd())

context = gx.get_context()

# Debug: list available datasources
print("Available datasources:", context.list_datasources())

datasource_name = "air_quality"
data_asset_name = "air_quality_data"

# Load the data asset
datasource = context.get_datasource(datasource_name)

# Debug: print base_directory of the datasource
try:
    print("Datasource base_directory:", datasource.base_directory)
except AttributeError:
    print("Could not access base_directory directly from datasource.")

asset = datasource.get_asset(data_asset_name)

# Debug: check actual files in the datasource base_directory
try:
    data_path = datasource.base_directory
    print("Files in data directory:", os.listdir(data_path))
except Exception as e:
    print("Error accessing data directory:", e)

# Load checkpoint
checkpoint_name = "air_quality_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

# Run the checkpoint
run_id = "air_quality_run"
checkpoint_result = checkpoint.run(run_id=run_id)

# Build data docs
context.build_data_docs()

# Check if the checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)
