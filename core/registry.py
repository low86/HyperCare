######################################
# 2026-2-2 #
# Model Registry #
######################################

from benchmarks.Task import (
    diag_prediction_mimic3_fn,
    diag_prediction_mimic4_fn,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
)

TASK_SPECS = {
    "diag": {
        "label_key": "conditions",
        "feature_keys": ["cond_hist", "procedures", "drugs"],
        "task_fn": {
            "mimic3": diag_prediction_mimic3_fn,
            "mimic4": diag_prediction_mimic4_fn,
        },
    },
    "drug": {
        "label_key": "drugs",
        "feature_keys": ["conditions", "procedures", "drugs_hist"],
        "task_fn": {
            "mimic3": drug_recommendation_mimic3_fn,
            "mimic4": drug_recommendation_mimic4_fn,
        },
    },
}
