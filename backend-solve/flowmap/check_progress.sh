#!/bin/bash
# Check pipeline progress

LOG_FILE="outputs/all_problems/pipeline.log"
SUMMARY_FILE="outputs/all_problems/summary.json"

echo "=== Pipeline Progress ==="
echo ""

# Count completed problems from log
if [ -f "$LOG_FILE" ]; then
    completed=$(grep -c "^\[.*\] 2024_odd_common_" "$LOG_FILE" || echo "0")
    echo "Problems processed: $completed / 46"

    # Show last few lines
    echo ""
    echo "Last 10 lines of log:"
    tail -10 "$LOG_FILE"
fi

echo ""
echo "=== Summary (if available) ==="
if [ -f "$SUMMARY_FILE" ]; then
    echo "Summary file exists"
    python3 -c "
import json
with open('$SUMMARY_FILE') as f:
    data = json.load(f)
print(f'Total: {len(data)} problems')
print(f'Success: {sum(1 for r in data if r.get(\"success\"))} problems')
print(f'Failed: {sum(1 for r in data if not r.get(\"success\"))} problems')
"
else
    echo "Summary not yet available"
fi

echo ""
echo "To view full log: tail -f $LOG_FILE"
