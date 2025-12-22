import os
import pandas as pd
import re

def count_python_lines(directory="."):
    total_lines = 0
    file_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        total_lines += line_count
                        file_count += 1
                        print(f"{file_path}: {line_count} lines")
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    print("\nSummary:")
    print(f"Total Python files: {file_count}")
    print(f"Total lines of code: {total_lines}")

if __name__ == "__main__":
    count_python_lines("/Users/hosseinsamaei/phd/epidemiqs/user-files/")
    FOLDER_PATH = "/Users/hosseinsamaei/phd/epidemiqs/user-files/flue-experiment"

def count_words(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))

total_words = 0

for file in os.listdir(FOLDER_PATH):
    if file.endswith((".xlsx", ".xls")):
        file_path = os.path.join(FOLDER_PATH, file)
        print(f"Processing: {file}")

        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        for sheet in excel_file.sheet_names:
            df = excel_file.parse(sheet)

            for col in df.columns:
                total_words += df[col].apply(count_words).sum()



total_tokens = 0

for f in os.listdir(FOLDER_PATH):
    if f.endswith((".xlsx", ".xls", ".csv")):
        full = os.path.join(FOLDER_PATH, f)
        size_bytes = os.path.getsize(full)

        estimated_tokens = size_bytes // 4   # rough estimate
        total_tokens += estimated_tokens

        print(f"{f}  -->  {size_bytes} bytes  (~{estimated_tokens} tokens)")

print("\n------------------------------")
print(f"TOTAL ESTIMATED TOKENS: {total_tokens}")
print("------------------------------")