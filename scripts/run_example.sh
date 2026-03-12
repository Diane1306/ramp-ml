#!/usr/bin/env bash
set -e

# Train + save everything
ramp-ml \
  --t_path /Users/diane_wt/Downloads/RampML_test/drop/T_test.xlsx \
  --VITA_path /Users/diane_wt/Downloads/RampML_test/drop/VITA_test.xlsx \
  --mapping VITA_to_T \
  --train_T T1,T2,T3,T4,T5 \
  --event_mode drop \
  --seed 123 --deterministic --device cpu \
  --save_dir /Users/diane_wt/Downloads/RampML_test/drop/runs/run1 --save_model --save_scores \
  --plot --plot_dir /Users/diane_wt/Downloads/RampML_test/drop/plot/

# Reproduce predictions (no training)
  ramp-ml \
    --t_path /Users/diane_wt/Downloads/RampML_test/drop/T_test.xlsx \
    --VITA_path /Users/diane_wt/Downloads/RampML_test/drop/VITA_test.xlsx \
    --mapping VITA_to_T \
    --load_model /Users/diane_wt/Downloads/RampML_test/drop/runs/run1/checkpoint.pt \
    --seed 123 --deterministic --device cpu \
    --save_dir /Users/diane_wt/Downloads/RampML_test/drop/runs/run1_reproduce --save_scores \
    --plot --plot_dir /Users/diane_wt/Downloads/RampML_test/drop/plot/