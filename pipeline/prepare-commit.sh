#! /bin/bash
modified_files=$(git ls-files --modified)
untracked_files=$(git ls-files --others --exclude-standard)
staged_files=$(git diff --name-only --cached)
all_files=$"$modified_files\n$untracked_files\n$staged_files"
# echo "all_files=${all_files}"
todo_nbs=$(echo "${all_files}" | grep -E '.*\.ipynb$' | sort | uniq)

echo "Cleaning notebooks..."
echo "${todo_nbs}" | xargs -I {} sh -c 'echo "Cleaning {}"; nbdev_clean --fname "{}"'

nbdev_prepare

for f in $(echo ${todo_nbs} | grep -E 'pipeline/.*?\.ipynb$'); do
    jupyter nbconvert --to python --no-prompt --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags dev "${f}"
done
black pipeline/

git status
