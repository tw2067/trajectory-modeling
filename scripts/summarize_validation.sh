#!/bin/bash

# Parse validation logs and generate summary

LOGDIR="/home/gaga/tamarw1/trajectory-modeling/logs/outs/generalized"

echo "========================================================================"
echo "Validation Summary"
echo "========================================================================"
echo ""
printf "%-10s %-20s %-15s %s\n" "Backend" "Scenario" "Disease" "Status"
echo "------------------------------------------------------------------------"

for logfile in $LOGDIR/validate_*.out; do
    if [ -f "$logfile" ]; then
        BACKEND=$(grep "Backend:" "$logfile" | awk '{print $2}')
        SCENARIO=$(grep "Scenario:" "$logfile" | awk '{print $2}')
        DISEASE=$(grep "Disease:" "$logfile" | awk '{print $2}')
        
        if grep -q "✓ PASSED" "$logfile"; then
            STATUS="✓ PASSED"
        elif grep -q "❌ FAILED" "$logfile"; then
            STATUS="❌ FAILED"
        else
            STATUS="⚠ UNKNOWN"
        fi
        
        printf "%-10s %-20s %-15s %s\n" "$BACKEND" "$SCENARIO" "$DISEASE" "$STATUS"
    fi
done

echo "------------------------------------------------------------------------"
echo ""
echo "Detailed logs in: $LOGDIR/"