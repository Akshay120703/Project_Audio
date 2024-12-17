import json
from IPython.display import FileLink

# Function to parse the file and extract key-value pairs based on the colon
def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    result = {}
    lines = data.splitlines()

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()

    return result

# Function to save the dictionary to a file
def save_dict_to_file(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, indent=4)
    print(f"Data saved to {filename}")

# Function to save user-requested key-value pairs to a new file without inverted commas and commas
def save_user_selected_pairs(data_dict, selected_keys, filename):
    selected_dict = {key: data_dict[key] for key in selected_keys if key in data_dict}

    # Manually format the dictionary as a string
    formatted_data = "updates = {\n"
    for key, value in selected_dict.items():
        formatted_data += f"    {key}: {value}\n"  # No commas and no inverted commas
    formatted_data += "}\n"

    # Write the formatted data to the file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(formatted_data)
    
    print(f"Selected key-value pairs saved to {filename} without inverted commas or commas")

# Path of the file
file_path = r'C:\Users\aksha\OneDrive\Desktop\Real_Data.txt'

# Process the file
data_dict = parse_file(file_path)

if data_dict:
    print("Parsed data:", data_dict)

    # Saving the dictionary to a new text file
    output_filename = 'parsed_data.json'
    save_dict_to_file(data_dict, output_filename)

    # Let the user input a list of keys to retrieve
    selected_keys = input("Enter the keys to retrieve, separated by commas: ").split(',')
    selected_keys = [key.strip() for key in selected_keys]

    # Save selected key-value pairs to 'updates.json' without inverted commas or commas
    updates_filename = 'updates.txt'
    save_user_selected_pairs(data_dict, selected_keys, updates_filename)

    # Provide download links for the generated files
    display(FileLink(output_filename))
    display(FileLink(updates_filename))
else:
    print("No data parsed! Please check the input file format.")
