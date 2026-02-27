#!/bin/bash
# Merge bootstrap trajectory cohort files for eICU circulatory failure
# Run after all SLURM array jobs complete

set -e

DATA_DIR="/home/gaga/data/physionet/eicu/circulatory_failure"
N_COHORTS=5

echo "Merging eICU Circulatory Failure Bootstrap trajectory files..."

# Merge each biomarker's trajectory files
for biomarker in lactate heartrate systolic; do
    echo "  Merging ${biomarker}..."
    output_file="${DATA_DIR}/${biomarker}_trajectory_probs_bootstrap.csv"
    
    # Get header from first cohort file
    first_file="${DATA_DIR}/${biomarker}_trajectory_probs_bootstrap_cohort00.csv"
    if [[ -f "$first_file" ]]; then
        head -n 1 "$first_file" > "$output_file"
        
        # Append data from all cohort files (skip headers)
        for i in $(seq 0 $((N_COHORTS - 1))); do
            cohort_file="${DATA_DIR}/${biomarker}_trajectory_probs_bootstrap_cohort$(printf '%02d' $i).csv"
            if [[ -f "$cohort_file" ]]; then
                tail -n +2 "$cohort_file" >> "$output_file"
            else
                echo "    WARNING: Missing $cohort_file"
            fi
        done
        
        echo "    ✓ Created $output_file ($(wc -l < "$output_file") lines)"
    else
        echo "    WARNING: No cohort files found for ${biomarker}"
    fi
done

# Merge prediction dataset files
echo "  Merging prediction datasets..."
output_file="${DATA_DIR}/circulatory_failure_prediction_dataset_with_bootstrap_probs.csv"
first_file="${DATA_DIR}/circulatory_failure_prediction_dataset_with_bootstrap_probs_cohort00.csv"

if [[ -f "$first_file" ]]; then
    head -n 1 "$first_file" > "$output_file"
    
    for i in $(seq 0 $((N_COHORTS - 1))); do
        cohort_file="${DATA_DIR}/circulatory_failure_prediction_dataset_with_bootstrap_probs_cohort$(printf '%02d' $i).csv"
        if [[ -f "$cohort_file" ]]; then
            tail -n +2 "$cohort_file" >> "$output_file"
        else
            echo "    WARNING: Missing $cohort_file"
        fi
    done
    
    echo "    ✓ Created $output_file ($(wc -l < "$output_file") lines)"
else
    echo "    WARNING: No prediction dataset cohort files found"
fi

echo ""
echo "✓ Merge complete!"
