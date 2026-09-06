"""Shared helper functions: reading config, saving/loading objects and arrays."""
import json
import os

import joblib
import numpy as np
import yaml

from src.fraud_detection.exception import FraudException


def read_yaml(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise FraudException(e)


def write_json(path: str, obj: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=4, default=str)
    except Exception as e:
        raise FraudException(e)


def save_object(path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)
    except Exception as e:
        raise FraudException(e)


def load_object(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        raise FraudException(e)


def save_numpy(path: str, arr: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
    except Exception as e:
        raise FraudException(e)


def load_numpy(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        raise FraudException(e)
