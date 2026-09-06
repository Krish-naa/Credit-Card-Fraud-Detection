"""Entry point to run the full training pipeline.

Usage:
    python train.py
"""
from src.fraud_detection.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    summary = TrainingPipeline().run()
    print("\n===== TRAINING SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
