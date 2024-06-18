#! /bin/bash
echo "Cleaning notebooks..."
(
    git ls-files --modified
    git ls-files --others --exclude-standard
    git diff --name-only --cached
) | grep ".ipynb" | sort -u |
    xargs -I {} sh -c 'echo "Cleaning {}"; nbdev_clean --fname "{}"'

nbdev_prepare

git status
