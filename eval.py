# eval.py
import argparse
import json
import os
import pickle

########################################################
# 평가 모델
########################################################

def evaluate(model_path: str) -> dict:
    print(f"[Eval] 모델 경로: {model_path}")

    if not os.path.isfile(model_path):
        print(f"[Eval] WARNING: 모델 파일 없음 - {model_path}, 더미 평가 수행")
        return {"accuracy": 0.0, "loss": 999.0, "pass": False}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[Eval] 모델 로드 완료: {model}")

    acc = float(model.get("accuracy", 0.0))
    loss = float(model.get("loss", 999.0))
    passed = acc >= 0.5

    return {"accuracy": acc, "loss": loss, "pass": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/hj-sample.pkl")
    args = parser.parse_args()

    result = evaluate(args.model)
    print(json.dumps(result))
