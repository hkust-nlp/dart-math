#! /bin/bash
set -e

modified_files=$(git ls-files --modified)
untracked_files=$(git ls-files --others --exclude-standard)
staged_files=$(git diff --name-only --cached)
all_files=$"$modified_files\n$untracked_files\n$staged_files"
# echo "all_files=${all_files}"
todo_nbs=$(echo "${all_files}" | grep -E '.*\.ipynb$' | sort | uniq)

echo "Cleaning notebooks..."
echo "${todo_nbs}" | xargs -I {} sh -c 'echo "Cleaning {}"; nbdev_clean --fname "{}"'

echo "Running nbdev_prepare..."
nbdev_prepare

pipeline_dir="pipeline"
if [ -d "${pipeline_dir}" ]; then
    for f in ${todo_nbs}; do
        if [[ "${f}" == "${pipeline_dir}/"*".ipynb" ]]; then
            jupyter nbconvert --to python --no-prompt --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags dev "${f}"
        fi
    done
    black "${pipeline_dir}"

fi

git status
