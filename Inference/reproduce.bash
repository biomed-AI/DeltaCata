#!/bin/bash

# Settings
TEST_DATASET_PATH="../Dataset/test_dataset/"
TEST_PDBS_PATH="../Dataset/test_pdbs/"
ARTIFACTS_PATH="./artifacts/"
FEATURE_PATH="./features/reproduce_data/"

# command
python reproduce.py --task kcat --level mut --test_dataset_path "$TEST_DATASET_PATH"kcat_mut_test.csv --pdbs_path "$TEST_PDBS_PATH" --artifacts_path "$ARTIFACTS_PATH" --feature_path "$FEATURE_PATH"
python reproduce.py --task kcat --level seq --test_dataset_path "$TEST_DATASET_PATH"kcat_seq_test.csv --pdbs_path "$TEST_PDBS_PATH" --artifacts_path "$ARTIFACTS_PATH" --feature_path "$FEATURE_PATH"
python reproduce.py --task km --level mut --test_dataset_path "$TEST_DATASET_PATH"km_mut_test.csv --pdbs_path "$TEST_PDBS_PATH" --artifacts_path "$ARTIFACTS_PATH" --feature_path "$FEATURE_PATH"
python reproduce.py --task km --level seq --test_dataset_path "$TEST_DATASET_PATH"km_seq_test.csv --pdbs_path "$TEST_PDBS_PATH" --artifacts_path "$ARTIFACTS_PATH" --feature_path "$FEATURE_PATH"