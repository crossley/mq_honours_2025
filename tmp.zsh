#!/bin/zsh

# Root directory - adjust if needed
ROOT_DIR="./projects"

# Find every 'data' directory under the projects folder
find "$ROOT_DIR" -type d -name "data" | while read data_dir; do
  # Create a new "pilot" directory inside each data directory
  pilot_dir="${data_dir}/pilot"
  mkdir -p "$pilot_dir"

  echo "Moving data files from $data_dir to $pilot_dir"

  # Move all files (not subdirectories) into the pilot directory
  find "$data_dir" -maxdepth 1 -type f -exec mv {} "$pilot_dir" \;
done

echo "Done moving files."

