#!/usr/bin/env bash
# c.f. https://platform.openai.com/docs/guides/batch
set -eo pipefail
# set -x

req_fpath=$1
check_interval=${2:-900}
if [ -z "${req_fpath}" ]; then
    echo "Usage: bash $0 <req_fpath>"
    exit 1
fi

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Please set the OPENAI_API_KEY environment variable."
    exit 1
fi

task_id="$(basename "${req_fpath}" | sed 's/req_\(.*\)\.jsonl/\1/')"

log_home="logs/openai/batch"
mkdir -p ${log_home}
log_path="${log_home}/${task_id}.log"

# c.f. https://platform.openai.com/docs/guides/batch/2-uploading-your-batch-input-file
curl https://api.openai.com/v1/files \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -F purpose="batch" \
    -F file="@${req_fpath}" \
    >>"${log_path}" 2>&1

# c.f. https://platform.openai.com/docs/guides/batch/3-creating-the-batch
input_file_id="$(grep -o '"id": "file-[^"]*"' "${log_path}" | tail -n 1 | sed 's/.*"\(file-.*\)".*/\1/')"
curl https://api.openai.com/v1/batches \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "input_file_id": "'"${input_file_id}"'",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }' \
    >>"${log_path}" 2>&1

# https://platform.openai.com/docs/guides/batch/4-checking-the-status-of-a-batch
batch_id="$(grep -o '"id": "batch_[^"]*"' "${log_path}" | tail -n 1 | sed 's/.*"\(batch_.*\)".*/\1/')"
while true; do
    curl "https://api.openai.com/v1/batches/${batch_id}" \
        -H "Authorization: Bearer ${OPENAI_API_KEY}" \
        -H "Content-Type: application/json" \
        >>"${log_path}" 2>&1

    if grep -qE '"status": "(failed|completed|expired|cancelled)"' "${log_path}"; then
        break
    fi
    sleep "${check_interval}"
done

# https://platform.openai.com/docs/guides/batch/5-retrieving-the-results
output_file_id="$(grep -o '"output_file_id": "file-[^"]*"' "${log_path}" | tail -n 1 | sed 's/.*"\(file-.*\)".*/\1/')"
output_home="data/oai-batch-resps"
mkdir -p "${output_home}"
curl "https://api.openai.com/v1/files/${output_file_id}/content" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    >"${output_home}/resp_${task_id}.jsonl" \
    2>>"${log_path}"
