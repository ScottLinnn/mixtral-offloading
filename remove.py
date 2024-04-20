def remove_double_newlines(log_file_path):
    # Read the entire file
    with open(log_file_path, "r") as file:
        content = file.read()

    # Replace double newlines with single newlines
    modified_content = content.replace("\n\n", "\n")

    # Write the modified content back to the file
    with open(log_file_path, "w") as file:
        file.write(modified_content)


if __name__ == "__main__":
    log_file_path = "cache-data/expert_cache_log2.txt"  # Path to the log file
    remove_double_newlines(log_file_path)
