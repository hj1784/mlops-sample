# train.py
import json
import os
import pickle


def train(epochs: int = 10):
    print("=" * 50)
    print("[Train] 학습 시작")
    print(f"[Train] 설정 epochs={epochs}")
    print("=" * 50)

    for epoch in range(epochs):
        loss = (epochs - epoch) * (1.0 / epochs)
        accuracy = epoch / float(epochs)
        print(f"[Train] Epoch {epoch + 1}/{epochs} - loss={loss:.4f}, accuracy={accuracy:.4f}")

    print("=" * 50)
    print(f"[Train] 학습 완료 - 최종 loss={loss:.4f}, accuracy={accuracy:.4f}")
    print("=" * 50)

    # 모델 산출물 저장
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hj-sample.pkl")

    model = {"accuracy": accuracy, "loss": loss, "epochs": epochs}
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[Train] 모델 저장 완료 - {model_path}")

    return json.dumps({"accuracy": accuracy, "loss": loss, "model_path": model_path})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    result = train(epochs=args.epochs)
    print(result)
