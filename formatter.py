def combine_lines(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            lines = infile.readlines()
            for i in range(0, len(lines), 2):
                combined_line = lines[i].strip()
                if i + 1 < len(lines):
                    combined_line += " " + lines[i + 1].strip()
                outfile.write(combined_line + '\n')

        print(f"Combined lines have been written to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file at {input_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_file_path = r"C:/Users/aksha/Downloads/sample_input.txt"
    output_file_path = r"C:/Users/aksha/Downloads/sample_output.txt"
    combine_lines(input_file_path, output_file_path)
