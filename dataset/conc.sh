#!/bin/bash

# Output file name
output_file="combined.py"

# Remove the output file if it exists
rm -f "$output_file"

# Loop through all .py files
for file in *.py; do
  # Check if it's a regular file
  if [ -f "$file" ]; then
    # Add a comment with the file name
    echo "# File: $file" >> "$output_file"

    # Append the content of the file
    cat "$file" >> "$output_file"

    # Add a newline for separation
    echo "" >> "$output_file"
  fi
done

echo "All .py files have been concatenated into $output_file"
