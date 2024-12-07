# Define the old and new paths
old_path = "/media/Storage2/yyz/"
new_path = "/your/new/path/"

# File paths
input_file = "noise_files.txt"  # Replace with the full path if not in the same directory
output_file = "updated_noise_files.txt"

# Open the file, replace the text, and save to a new file
with open(input_file, "r") as file:
    data = file.read()

# Replace the old path with the new path
updated_data = data.replace(old_path, new_path)

# Write the updated data to a new file
with open(output_file, "w") as file:
    file.write(updated_data)

print(f"Path updated and saved to {output_file}")
