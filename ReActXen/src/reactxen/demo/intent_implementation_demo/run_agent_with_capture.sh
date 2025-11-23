#!/bin/bash
# Script to run the agent system with comprehensive output capture
# Uses the .venv Python interpreter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
RUN_SCRIPT="${SCRIPT_DIR}/multi_agent_implementation_demo/run_with_output_capture.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/multi_agent_implementation_demo/outputs"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Output files
FULL_OUTPUT="${OUTPUT_DIR}/full_output_${TIMESTAMP}.txt"
FINAL_OUTPUT="${OUTPUT_DIR}/final_output_${TIMESTAMP}.txt"

echo "========================================"
echo "Running Hierarchical Agent System"
echo "========================================"
echo "Using Python: ${VENV_PYTHON}"
echo "Script: ${RUN_SCRIPT}"
echo "Full output: ${FULL_OUTPUT}"
echo "Final output: ${FINAL_OUTPUT}"
echo "========================================"
echo ""

# Run the script and capture all output
"${VENV_PYTHON}" "${RUN_SCRIPT}" 2>&1 | tee "${FULL_OUTPUT}"

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Execution completed with exit code: ${EXIT_CODE}"
echo "========================================"
echo "Full output saved to: ${FULL_OUTPUT}"

# Check if final output files were created by the script
if [ -f "${FINAL_OUTPUT}" ]; then
    echo "Final output saved to: ${FINAL_OUTPUT}"
fi

# Check for verification analysis
VERIFICATION_FILE="${OUTPUT_DIR}/verification_analysis_${TIMESTAMP}.json"
if [ -f "${VERIFICATION_FILE}" ]; then
    echo "Verification analysis saved to: ${VERIFICATION_FILE}"
fi

exit ${EXIT_CODE}

