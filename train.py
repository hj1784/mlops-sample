# train.py
import mlflow
import os

########################################################
# 학습 모델
########################################################

############################
# MLflow 설정 (선택사항)
############################
# 기본값: 로컬 ./mlruns/ 폴더에 저장
mlflow.set_tracking_uri("http://192.168.2.81:30500")  # 원격 서버 사용시 MLflow 서버 주소

############################
# mlflow.start_run()
############################
# - 새로운 "실험 실행(run)"을 시작
# - 이 블록 안에서 기록한 모든 것이 하나의 run으로 묶임
# - with문 끝나면 자동으로 run 종료
# - 저장 위치: ./mlruns/0/<run_id>/
with mlflow.start_run():

    # [가짜 학습 과정]
    print("🚀 Training started...")
    epochs = 10
    for epoch in range(epochs):
        loss = (10 - epoch) * 0.1
        accuracy = epoch / 10.0

        ############################
        # mlflow.log_metric()
        ############################
        # - 숫자 값(메트릭)을 기록
        # - step 파라미터로 x축 값 지정 (보통 epoch)
        # - 나중에 그래프로 시각화 가능
        # - 저장 위치: ./mlruns/0/<run_id>/metrics/
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}")

    # [모델 생성]
    model_path = "model.txt"
    model_abs_path = os.path.abspath(model_path)
    print(f"📁 model.txt 저장 경로: {model_abs_path}")

    with open(model_path, "w") as f:
        f.write("MODEL_VERSION=1\n")
        f.write(f"FINAL_ACCURACY={accuracy}")

    ############################
    # mlflow.log_artifact()
    ############################
    # - 파일을 통째로 저장 (모델, 이미지, 설정파일 등)
    # - artifact_path: 저장할 하위 폴더 이름
    # - 저장 위치: ./mlruns/0/<run_id>/artifacts/model/model.txt
    #
    # 참고: 실제 PyTorch 모델이면 mlflow.pytorch.log_model() 사용
    #       실제 Sklearn 모델이면 mlflow.sklearn.log_model() 사용
    mlflow.log_artifact(model_path, artifact_path="model")

    print("✅✅✅✅ training & logging done ✅✅✅✅")
