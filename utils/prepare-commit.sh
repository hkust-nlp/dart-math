#! /bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Gather files to process
modified_files=$(git ls-files --modified)
untracked_files=$(git ls-files --others --exclude-standard)
staged_files=$(git diff --name-only --cached)
all_files=$(echo -e $"$modified_files\n$untracked_files\n$staged_files")
# echo "all_files=${all_files}"
todo_nbs=$(echo "${all_files}" | grep -E '.*\.ipynb$' | sort | uniq)

# nbdev_clean
echo "Cleaning notebooks..."
echo "${todo_nbs}" | xargs -I {} sh -c 'echo "Cleaning {}"; nbdev_clean --fname "{}"'

# nbdev_prepare
echo "Running nbdev_prepare..."
nbdev_prepare

# Convert notebooks to Python scripts with `nbconvert`
# and format them with `black`
pipeline_dir="pipeline"
if [ -d "${pipeline_dir}" ]; then
    for f in ${todo_nbs}; do
        # If the notebook file and the corresponding Python script both exist
        nb_basename=$(basename "${f}" .ipynb)
        if [[ -f "${pipeline_dir}/${nb_basename}.py" ]] && [[ -f "${f}" ]]; then
            echo "Converting ${f} to Python script... Check if manual modifications are needed."
            jupyter nbconvert --to python --no-prompt --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags dev "${f}"
        fi
    done
    black "${pipeline_dir}"
fi

git status # Show the status of the working tree
