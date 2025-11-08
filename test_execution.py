
import sys
import os
from dsstar import DS_STAR_Agent, DSConfig
import shutil

def main():
    """
    Test script to diagnose file access issues in _execute_code.
    """
    # --- Setup ---
    run_id = 'test_execution_run'
    project_root = '/Users/jules/Documents/Projets/DS_Star_impl'
    
    # Ensure we are in the project root, as dsstar.py expects to find config files there.
    os.chdir(project_root)
    print(f"Current working directory: {os.getcwd()}")

    # Clean up previous test run artifacts if they exist
    run_dir = os.path.join('runs', run_id)
    if os.path.exists(run_dir):
        print(f"Removing existing test run directory: {run_dir}")
        shutil.rmtree(run_dir)

    config = DSConfig(run_id=run_id)
    agent = DS_STAR_Agent(config)

    # --- Test 1: Code with relative path ---
    print("\n--- Test 1: Executing code with a relative path ('data/Fichier INPUT.xlsx') ---")
    
    relative_path_code = """
import pandas as pd
import os
print(f"CWD inside executed script: {os.getcwd()}")
file_path = 'data/Fichier INPUT.xlsx'
try:
    df = pd.read_excel(file_path)
    print(f"SUCCESS: File read successfully using relative path: {file_path}")
except FileNotFoundError:
    print(f"ERROR: File not found using relative path: {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
"""
    # We pass the base filename to _execute_code for validation purposes
    result, error = agent._execute_code(relative_path_code, data_files=['Fichier INPUT.xlsx'])

    if error:
        print(f"Test 1 Result: FAILED. Execution returned an error.")
        print(f"Stderr: {error}")
    else:
        print("Test 1 Result: PASSED. Execution was successful.")
        print("Stdout:")
        print(result)
        if "ERROR" in result:
            print("Note: Execution succeeded, but the script inside reported a FileNotFoundError. This confirms the pathing problem.")


    # --- Test 2: Code with absolute path ---
    print("\n--- Test 2: Executing code with an absolute path ---")
    
    # Construct an absolute path to the data file
    data_dir = os.path.abspath(config.data_dir)
    file_name = 'Fichier INPUT.xlsx'
    absolute_path = os.path.join(data_dir, file_name).replace('\\\\', '/') # Normalize for cross-platform robustness

    absolute_path_code = f"""
import pandas as pd
import os
print(f"CWD inside executed script: {os.getcwd()}")
file_path = r'{absolute_path}'
print(f"Attempting to read absolute path: {{file_path}}")
try:
    df = pd.read_excel(file_path)
    print(f"SUCCESS: File read successfully using absolute path: {{file_path}}")
except FileNotFoundError:
    print(f"ERROR: File not found using absolute path: {{file_path}}")
except Exception as e:
    print(f"An unexpected error occurred: {{e}}")
"""
    result, error = agent._execute_code(absolute_path_code, data_files=['Fichier INPUT.xlsx'])

    if error:
        print(f"Test 2 Result: FAILED. Execution returned an error.")
        print(f"Stderr: {error}")
    else:
        print("Test 2 Result: PASSED. Execution was successful.")
        print("Stdout:")
        print(result)
        if "SUCCESS" in result:
            print("Note: The script inside successfully read the file. This confirms the absolute path solution works.")


if __name__ == "__main__":
    main()
