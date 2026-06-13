#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

for source in 01_system_overview.dot \
              02_arm_execution.dot \
              03_yolo_execution.dot \
              04_ocr_execution.dot; do
  output="${source%.dot}.png"
  echo "Rendering ${source} -> ${output}"
  dot -Tpng "${source}" -o "${output}"
done

echo "Rendered 4 diagrams in ${script_dir}"
