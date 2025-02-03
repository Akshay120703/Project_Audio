import os
import pandas as pd
import shutil

def retrieve_audio_files(csv_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if there are at least 7 columns
    if df.shape[1] < 7:
        print("Error: CSV file must have at least 7 columns.")
        return
    
    # User input for keyword
    keyword = input("Enter the keyword to search for: ")
    
    # Filter rows containing the keyword in the 7th column
    filtered_rows = df[df.iloc[:, 6].str.contains(keyword, na=False, case=False)]
    
    if filtered_rows.empty:
        print(f"No audio files found for keyword: {keyword}")
        return
    
    # Copy the matching audio files to the output folder
    for index, row in filtered_rows.iterrows():
        audio_path = row.iloc[0]  # First column contains audio paths
        if os.path.exists(audio_path):
            shutil.copy(audio_path, os.path.join(output_folder, os.path.basename(audio_path)))
            print(f"Copied: {audio_path}")
        else:
            print(f"Warning: File not found - {audio_path}")
    
    print(f"Retrieved audio files are stored in '{output_folder}'")

if __name__ == "__main__":
    csv_file_path = input("Enter the path to the CSV file: ")
    output_directory = "retrieved_audio_files"
    retrieve_audio_files(csv_file_path, output_directory)
