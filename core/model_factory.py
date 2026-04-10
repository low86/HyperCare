######################################
# 2026-2-2 #
# Model Fictory #
######################################
from benchmarks.drug import drug_baselines
from benchmarks.diag import diag_baselines


def build_model(args, spec, device):
    # Task-specific dispatch; keep baselines untouched.
    if args.task == "drug":
        model_name = args.model
        model = drug_baselines.build_model(
            model_name=model_name,
            task_dataset=spec.task_dataset,
            tokenizers=spec.tokenizers,
            device=device,
            args=args,
        )
        return model

    if args.task == "diag":
        model_name = args.model
        model = diag_baselines.build_model(
            model_name=model_name,
            task_dataset=spec.task_dataset,
            tokenizers=spec.tokenizers,
            device=device,
            args=args,
        )
        return model

    raise ValueError(f"Unknown task: {args.task}")
