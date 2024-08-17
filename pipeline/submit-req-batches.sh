#!/usr/bin/env bash
# Submit request batches to OpenAI API
set -eo pipefail
# set -x

req_fpath_pattern=$1
if [ -z "${req_fpath_pattern}" ]; then
    echo "Usage: bash $0 <req_fpath_pattern>
    E.g. bash $0 'data/oai-batch-reqs/req_*.jsonl'
    "
    exit 1
fi
check_interval=${2:-900}

# Use nullglob to handle cases where no files match the pattern
shopt -s nullglob

req_fpaths=(${req_fpath_pattern})
if [ ${#req_fpaths[@]} -eq 0 ]; then
    echo "Error: No files found matching pattern '${req_fpath_pattern}'"
    exit 1
fi

for req_fpath in "${req_fpaths[@]}"; do
    if [ ! -f "${req_fpath}" ]; then
        echo "Warning: '${req_fpath}' is not a file. Skipping."
        continue
    fi
    echo "Processing file: ${req_fpath}"
    bash pipeline/run-req-batch.sh "${req_fpath}" "${check_interval}" &
done

wait
echo "All request batches finished."
