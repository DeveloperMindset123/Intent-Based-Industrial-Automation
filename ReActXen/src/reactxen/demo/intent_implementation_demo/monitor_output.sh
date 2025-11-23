#!/bin/bash
# Script to monitor the agent output files
# Usage: ./monitor_output.sh [timestamp]

OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/multi_agent_implementation_demo/outputs"
TIMESTAMP="${1:-$(date +%Y%m%d)}"

echo "Monitoring output files in: ${OUTPUT_DIR}"
echo "Looking for files with timestamp: ${TIMESTAMP}"
echo ""

FULL_OUTPUT=$(ls -t "${OUTPUT_DIR}"/full_output_${TIMESTAMP}*.txt 2>/dev/null | head -1)
FINAL_OUTPUT=$(ls -t "${OUTPUT_DIR}"/final_output_${TIMESTAMP}*.txt 2>/dev/null | head -1)
VERIFICATION=$(ls -t "${OUTPUT_DIR}"/verification_analysis_${TIMESTAMP}*.json 2>/dev/null | head -1)

if [ -n "${FULL_OUTPUT}" ]; then
    echo "✅ Full Output: ${FULL_OUTPUT}"
    echo "   Size: $(wc -l < "${FULL_OUTPUT}") lines"
    echo "   Last 10 lines:"
    tail -10 "${FULL_OUTPUT}" | sed 's/^/   /'
    echo ""
else
    echo "⏳ Full output file not yet created"
    echo ""
fi

if [ -n "${FINAL_OUTPUT}" ]; then
    echo "✅ Final Output: ${FINAL_OUTPUT}"
    echo "   Size: $(wc -l < "${FINAL_OUTPUT}") lines"
    echo "   Contents:"
    cat "${FINAL_OUTPUT}"
    echo ""
else
    echo "⏳ Final output file not yet created (agent still running)"
    echo ""
fi

if [ -n "${VERIFICATION}" ]; then
    echo "✅ Verification Analysis: ${VERIFICATION}"
    if command -v jq &> /dev/null; then
        echo "   Analysis:"
        jq . "${VERIFICATION}" | sed 's/^/   /'
    else
        echo "   (Install jq for formatted JSON output)"
        cat "${VERIFICATION}"
    fi
else
    echo "⏳ Verification analysis not yet created"
    echo ""
fi

