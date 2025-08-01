"""
Simple wrapper script to run the cloud embeddings test and save the output to a file
"""

import os
import subprocess
import sys
import datetime

# Get the current timestamp for the log file name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"test_results_{timestamp}.log"

# Run the test with output redirected to the log file
print(f"Running test and saving output to {log_file}...")

# Use the python executable from the current environment
python_exe = sys.executable
cmd = [python_exe, "test_cloud_embeddings.py"]

with open(log_file, "w") as f:
    # Run the process and capture output
    process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

# Print the result
exit_code = process.returncode
if exit_code == 0:
    print(f"Test completed successfully. See {log_file} for details.")
else:
    print(f"Test failed with exit code {exit_code}. See {log_file} for details.")

# Display the log file content
print("\nLog file content:")
with open(log_file, "r") as f:
    print(f.read())
