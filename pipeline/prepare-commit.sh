#! /bin/bash
modified_files=$(git ls-files --modified)
untracked_files=$(git ls-files --others --exclude-standard)
cached_files=$(git diff --name-only --cached)
todo_nbs=$(echo "$modified_files"$'\n'"$untracked_files"$'\n'"$cached_files" | grep ".ipynb" | sort -u)
echo "todo_nbs=${todo_nbs}"

echo "Cleaning notebooks..."
echo "${todo_nbs}" | xargs -I {} sh -c 'echo "Cleaning {}"; nbdev_clean --fname "{}"'

nbdev_prepare

for f in $(echo ${todo_nbs} | grep 'pipeline/.*.ipynb'); do
    jupyter nbconvert --to python --no-prompt --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags dev "${f}"
    black "${f%.ipynb}.py"
done

git status
