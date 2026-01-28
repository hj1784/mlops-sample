# eval.py
import json
from pathlib import Path

########################################################
# 평가 모델
########################################################

model = open("model.txt").read()

# 그냥 임의 규칙
passed = "MODEL_VERSION" in model

result = {
    "pass": passed,
    "score": 1.0 if passed else 0.0
}

OUT_DIR = Path("/eval_out")
OUT_DIR.mkdir(exist_ok=True)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(result, f)

print("🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁 EVAL RESULT:", result)
