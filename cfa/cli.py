"""Command-line interface for the CFA pipeline."""

import argparse
import logging
import sys

from cfa.config import load_config


def setup_logging(config: dict):
    """Configure logging based on config."""
    level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_sft(args):
    """Run SFT training stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.sft import run_sft
    run_sft(config)


def cmd_generate(args):
    """Run generation + GPT scoring stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.generation import run_generation
    run_generation(config)


def cmd_calibrate(args):
    """Run conformal prediction calibration."""
    config = load_config(args.config)
    setup_logging(config)
    if args.quantile is not None:
        config["conformal"]["quantile_bars"] = [args.quantile]
    from cfa.stages.calibration import run_calibration
    run_calibration(config)


def cmd_feedback(args):
    """Run AI feedback annotation stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.feedback import run_feedback
    run_feedback(config)


def cmd_assign_weights(args):
    """Run uncertainty weight assignment stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.weights import run_assign_weights
    run_assign_weights(config)


def cmd_train(args):
    """Run weighted DPO training stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.train import run_train
    run_train(config)


def cmd_infer(args):
    """Run inference on test set."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.inference import run_inference
    run_inference(config)


def cmd_evaluate(args):
    """Run GPT-4o evaluation stage."""
    config = load_config(args.config)
    setup_logging(config)
    from cfa.stages.evaluation import run_evaluation
    run_evaluation(config)


def cmd_run_all(args):
    """Run the full pipeline end-to-end."""
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("cfa.pipeline")

    stages = [
        ("SFT Training", "cfa.stages.sft", "run_sft"),
        ("Generation + Scoring", "cfa.stages.generation", "run_generation"),
        ("Conformal Prediction Calibration", "cfa.stages.calibration", "run_calibration"),
        ("AI Feedback Annotation", "cfa.stages.feedback", "run_feedback"),
        ("Uncertainty Weight Assignment", "cfa.stages.weights", "run_assign_weights"),
        ("Weighted DPO Training", "cfa.stages.train", "run_train"),
        ("Test Inference", "cfa.stages.inference", "run_inference"),
        ("Evaluation", "cfa.stages.evaluation", "run_evaluation"),
    ]

    for i, (name, module_path, func_name) in enumerate(stages, 1):
        logger.info(f"[{i}/{len(stages)}] Starting: {name}")
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        func(config)
        logger.info(f"[{i}/{len(stages)}] Completed: {name}")

    logger.info("Full pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        prog="cfa",
        description="Conformal Feedback Alignment - RLUF with Conformal Prediction for DPO",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )

    subparsers = parser.add_subparsers(
        title="stages",
        description="Pipeline stages (run individually or use 'run-all')",
    )

    # sft
    p = subparsers.add_parser("sft", help="Stage 1a: SFT training on summarization data")
    p.set_defaults(func=cmd_sft)

    # generate
    p = subparsers.add_parser("generate", help="Stage 1b: Sample generations + GPT scoring")
    p.set_defaults(func=cmd_generate)

    # calibrate
    p = subparsers.add_parser("calibrate", help="Stage 1c: Conformal prediction calibration")
    p.add_argument("--quantile", type=float, help="Override quantile_bar (e.g., 0.2 or 0.5)")
    p.set_defaults(func=cmd_calibrate)

    # feedback
    p = subparsers.add_parser("feedback", help="Stage 2a: AI pairwise preference annotation")
    p.set_defaults(func=cmd_feedback)

    # assign-weights
    p = subparsers.add_parser("assign-weights", help="Stage 2b: Assign uncertainty weights to DPO pairs")
    p.set_defaults(func=cmd_assign_weights)

    # train
    p = subparsers.add_parser("train", help="Stage 3a: Weighted DPO training")
    p.set_defaults(func=cmd_train)

    # infer
    p = subparsers.add_parser("infer", help="Stage 3b: Generate answers on test set")
    p.set_defaults(func=cmd_infer)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Stage 3c: GPT-4o evaluation of generated answers")
    p.set_defaults(func=cmd_evaluate)

    # run-all
    p = subparsers.add_parser("run-all", help="Run the full pipeline end-to-end")
    p.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
