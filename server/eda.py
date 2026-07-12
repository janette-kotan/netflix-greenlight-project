import pandas as pd
import numpy as np
import ast
import os

def parse_json_column(val):
    # Safely converts stringified JSON lists into Python lists.
    try:
        if pd.isna(val):
            return []
        parsed_list = ast.literal_eval(val)
        return [item['name'] for item in parsed_list if 'name' in item]
    except (ValueError, SyntaxError):
        return []

def run_data_audit():
    # Dynamic Absolute Path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "movies.csv")

    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return

    df = pd.read_csv(data_path)

    print(f"Total Movies/Rows: {df.shape[0]}")
    print(f"Total Attributes/Columns: {df.shape[1]}")

    print("\n Parsing nested text arrays (Genres and Keywords)...")
    for col in ['genres', 'keywords']:
        if col in df.columns:
            df[col] = df[col].apply(parse_json_column)

    print("\n MISSING DATA AND NULL AUDIT")
    null_counts = df.isnull().sum()
    missing_data = pd.DataFrame({'Null Count': null_counts, 'Percentage (%)': (null_counts / len(df)) * 100})
    print(missing_data[missing_data['Null Count'] > 0].sort_values(by='Null Count', ascending=False))

if __name__ == "__main__":
    run_data_audit()