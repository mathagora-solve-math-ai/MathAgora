#!/bin/bash
# Check v4 pipeline progress

LOG_FILE="outputs/v4_all/pipeline.log"
SUMMARY_FILE="outputs/v4_all/summary.json"

echo "=== v4 Pipeline Progress ==="
echo ""

# Check if running
if pgrep -f "pipeline_v4_all.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
    echo ""
else
    echo "⚠️  Pipeline is NOT running (may have completed or stopped)"
    echo ""
fi

# Count completed problems from log
if [ -f "$LOG_FILE" ]; then
    completed=$(grep -c "^\\[.*\\] \\[.*\\] 2024_odd_" "$LOG_FILE" || echo "0")
    echo "Problems processed: $completed / 46"

    # Show last few lines
    echo ""
    echo "Last 5 log entries:"
    tail -5 "$LOG_FILE" | grep -v "cargo/env"
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

# Average stats
successful = [r for r in data if r.get('success')]
if successful:
    avg_groups = sum(r['n_groups'] for r in successful) / len(successful)
    avg_flows = sum(r['n_flows'] for r in successful) / len(successful)
    print(f'')
    print(f'Average groups: {avg_groups:.1f}')
    print(f'Average flows: {avg_flows:.1f}')
"
else
    echo "Summary not yet available"
fi

echo ""
echo "=== Commands ==="
echo "View full log:  tail -f $LOG_FILE"
echo "Check process:  ps aux | grep pipeline_v4_all"
echo "Kill process:   pkill -f pipeline_v4_all.py"
