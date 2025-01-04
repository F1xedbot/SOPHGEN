import os
import glob
import argparse

def find_files(base_name):
    # Normalize paths to use correct separators for the OS
    base_file = os.path.normpath(f"./input/{base_name}_nonvul.c")
    gen_files = glob.glob(os.path.normpath(f"./generated/{base_name}_gen_*.c"))
    
    # Ensure base file exists
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file {base_file} not found.")
    
    # Ensure there are generated files
    if not gen_files:
        raise FileNotFoundError(f"No generated files found for {base_name}.")
    
    return base_file, gen_files


def run_gumtree_diff(file1, file2):
    try:
        # Construct the command to run gumtree and save the output to a temp file
        command = f"gumtree textdiff {file1} {file2} > generated/temp.diff"
        
        # Debugging: Print the command being executed
        print(f"Running command: {command}")
        
        # Use os.system() to run the command and capture output
        result = os.system(command)
        
        # Check if the command was successful
        if result != 0:
            print(f"Error running gumtree diff: {result}")
            return None
        
        # Read the generated diff file
        with open("generated/temp.diff", "r") as diff_file:
            diff_output = diff_file.read()
        
        # Return the diff output
        return diff_output

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def handle_delete(file_before, offsets):

    start_offset, end_offset = offsets
    start_line, end_line, deleted_lines = get_line_from_diff(file_before, start_offset, end_offset)

    # Construct the output in the required format
    result = "DELETE: " + " ".join([line.strip() for line in deleted_lines]) + f"!@#$[{start_line}:{end_line}]"
    print(result)
    
    # Write the result to the predictions_diff.txt file
    with open("outputs/predictions_diff.txt", "a", encoding="utf-8") as f:
        f.write(result + "\n")  # Add a new line after each result


def handle_insert(file_before, file_after, offsets):
    start_offset, end_offset = offsets

    before_start_line, before_end_line, before_lines = get_line_from_diff(file_before, start_offset, end_offset)
    _, _, after_lines = get_line_from_diff(file_after, start_offset, end_offset)

    # Construct the output in the required format
    insert_content = f"LINE[{before_start_line}:{before_end_line}]"
    if before_start_line == before_end_line:
        insert_content = " ".join([line.strip() for line in before_lines])
    result = "INSERT: " + insert_content + ' -> ' + " ".join([line.strip() for line in after_lines]) + f"!@#$[{before_start_line}:{before_end_line}]"
    with open("outputs/predictions_diff.txt", "a", encoding="utf-8") as f:
        f.write(result + "\n")  # Add a new line after each result


def handle_update(file_before, file_after, offsets):

    start_offset, end_offset = offsets
    before_start_line, before_end_line, before_lines = get_line_from_diff(file_before, start_offset, end_offset)
    _, _, after_lines = get_line_from_diff(file_after, start_offset, end_offset)

    # Construct the output in the required format
    result = "UPDATE: " + " ".join([line.strip() for line in before_lines]) + ' -> ' + " ".join([line.strip() for line in after_lines]) + f"!@#$[{before_start_line}:{before_end_line}]"
    with open("outputs/predictions_diff.txt", "a", encoding="utf-8") as f:
        f.write(result + "\n")  # Add a new line after each result


def handle_move(file_before, file_after, before_offsets, after_offsets):
    before_start, before_end = before_offsets
    after_start, after_end = after_offsets

    _, _, before_lines = get_line_from_diff(file_before, before_start, before_end)
    after_start_line, after_end_line, _ = get_line_from_diff(file_after, after_start, after_end)

    before_result = "MOVE: " + " ".join([line.strip() for line in before_lines]) + ' -> ' + f"LINE[{after_start_line}:{after_end_line}]"
    with open("outputs/predictions_diff.txt", "a", encoding="utf-8") as f:
        f.write(before_result + "\n")  # Add a new line after each result


def process_gumtree_change(file_before, file_after, offsets, operation):
    if operation.startswith("delete"):
        handle_delete(file_before, offsets)

    elif operation.startswith("insert"):
        handle_insert(file_before, file_after, offsets)

    elif operation.startswith("update"):
        handle_update(file_before, file_after, offsets)

    elif operation.startswith("move"):
        before_offsets, after_offsets = offsets
        handle_move(file_before, file_after, before_offsets, after_offsets)


def get_line_from_diff(file_path, start_offset, end_offset):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read the file and split into lines
        lines = f.readlines()

    # Initialize variables to track byte offset
    current_offset = 0
    start_line = None
    end_line = None

    # Iterate through each line and its byte offset
    for i, line in enumerate(lines):
        line_length = len(line.encode('utf-8'))  # Get the byte length of the line
        if current_offset <= start_offset < current_offset + line_length:
            start_line = i
        if current_offset <= end_offset < current_offset + line_length:
            end_line = i
        current_offset += line_length
    return start_line, end_line, lines[start_line-1:end_line]


def find_endline(lines, start_index):
    endline = None
    for i in range(start_index, len(lines)):
        line = lines[i].strip()

        # Skip lines that are operations (insert, delete, update, move)
        if line.startswith("insert") or line.startswith("delete") or \
           line.startswith("update") or line.startswith("move"):
            continue

        # If the line is blank or an operation, take the previous line as the endline
        if not line:  # Blank line
            if endline is not None:
                return lines[endline]  # Return the full content of the last non-operation line
        else:
            # If the line is not an operation and not blank, update endline
            endline = i

    # If no blank line or operation is encountered, return the last non-operation line
    return lines[endline].strip() if endline is not None else None

def extract_first_operation(diff_output):
    lines = diff_output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("insert") or line.startswith("delete") or \
           line.startswith("update") or line.startswith("move"):
            operation = line.strip()  # Extract the operation (e.g., insert-node)
            # Look for the next line that contains the offsets (after the operation)
            if i + 2 < len(lines):
                offset_line = lines[i + 2].strip()

                # Extract the offsets as before
                offsets_str = offset_line.split('[')[1].split(']')[0]
                offsets = tuple(map(int, offsets_str.split(',')))  # Convert to tuple of integers

                # Find the endline and extract the end_offset
                endline = find_endline(lines, i + 2)
                endline_str = endline.split('[')[1].split(']')[0]
                end_offsets = list(map(int, endline_str.split(',')))  # Convert to list of integers
                
                # Get the second offset (end_offset)
                end_offset = end_offsets[1]  # This is the second value in the tuple

                # Now you can use the end_offset in combination with the offsets
                # Example: Combine the start offset with the end offset (if required)
                real_offsets = (offsets[0], end_offset)  # Assuming you want to pair the start and end offsets

                return operation, real_offsets


    return None, None

def main():
    parser = argparse.ArgumentParser(description="Process Gumtree diffs.")
    parser.add_argument('--name', required=True, help="Base file name (without suffix)")
    args = parser.parse_args()
    base_name = args.name
    try:
        # Step 1: Find the base and generated files
        base_file, gen_files = find_files(base_name)

        # Step 2: Process each generated file
        for gen_file in gen_files:
            print(f"Processing diff between {base_file} and {gen_file}")

            # Step 3: Run gumtree textdiff
            diff_output = run_gumtree_diff(base_file, gen_file)

            if diff_output:
                # Step 4: Extract the first operation and offsets
                operation, offsets = extract_first_operation(diff_output)

                if operation and offsets:
                    print(f"Found operation: {operation} with offsets {offsets}")
                    # Step 5: Process the change
                    process_gumtree_change(base_file, gen_file, offsets, operation)
                else:
                    print(f"No valid operations found in the diff between {base_file} and {gen_file}")
            else:
                print(f"Failed to generate diff between {base_file} and {gen_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
