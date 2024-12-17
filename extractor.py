def filter_transcriptions(file1_path, file2_path, output_file_path):
    try:
        # Read the content of the first input file
        with open(file1_path, 'r') as file1:
            file1_lines = file1.readlines()

        # Read the content of the second input file
        with open(file2_path, 'r') as file2:
            file2_lines = file2.readlines()

        # Extract the audio labels from file2
        file2_labels = {line.split(":")[0].strip() for line in file2_lines}

        # Filter lines from file1 where the audio label matches those in file2
        filtered_lines = [line for line in file1_lines if line.split(":")[0].strip() in file2_labels]

        # Write the filtered lines to the output file
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(filtered_lines)

        print(f"Filtered transcriptions have been written to: {output_file_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    file1_path = r"C:/Users/aksha/Downloads/input1.txt"
    file2_path = r"C:/Users/aksha/Downloads/input2.txt"
    output_file_path = r"C:/Users/aksha/Downloads/extractor_output.txt"

    filter_transcriptions(file1_path, file2_path, output_file_path)
