#!/usr/bin/env bash
set -e

ramp-ml \
  --t_path /path/to/T_test.xlsx \
  --reset_path /path/to/VITA_test.xlsx \
  --mapping VITA_to_T \
  --train_T T1,T2,T3 \
  --test_T T4,T5 \
  --event_mode drop \
  --plot --plot_dir plots