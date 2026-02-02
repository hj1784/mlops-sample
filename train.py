# train.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mlflow
from mlflow.tracking import MlflowClient
import os

########################################################
# 학습 모델
########################################################

############################
# MLflow 설정 (선택사항)
############################
# 기본값: 로컬 ./mlruns/ 폴더에 저장
mlflow.set_tracking_uri("http://192.168.2.81:30500")  # 원격 서버 사용시 MLflow 서버 주소

# 새 experiment 생성 (NFS 경로 사용)
mlflow.set_experiment("poc-train-1")

############################
# mlflow.start_run()
############################
# - 새로운 "실험 실행(run)"을 시작
# - 이 블록 안에서 기록한 모든 것이 하나의 run으로 묶임
# - with문 끝나면 자동으로 run 종료
# - 저장 위치: ./mlruns/0/<run_id>/
with mlflow.start_run():
    # pipeline_run_id를 MLflow param으로 기록 (메트릭 폴링 연동용)
    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", "unknown")
    mlflow.log_param("pipeline_run_id", pipeline_run_id)

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
    #       (단, log_model()은 서버가 MLflow 3.x 이상이어야 정상 동작)
    mlflow.log_artifact(model_path, artifact_path="model")

    ############################
    # 모델 등록 (Model Registry)
    ############################
    # - 학습된 모델을 Model Registry에 등록
    # - Staging 스테이지로 전환하여 배포 준비 상태로 설정
    artifact_path = "model"
    model_name = "test-model"

    mlflow_run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{mlflow_run_id}/{artifact_path}"
    print(f"[register_model] model_uri={model_uri}, name={model_name}")

    # MlflowClient를 직접 사용 (클라이언트 3.x + 서버 2.x 호환성 문제 우회)
    client = MlflowClient()

    # 등록된 모델이 없으면 새로 생성
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    # 모델 버전 생성
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=mlflow_run_id,
    )

    ############################
    # Staging 스테이지 전환
    ############################
    client.transition_model_version_stage(
        name=model_name, version=mv.version,
        stage="Staging", archive_existing_versions=False,
    )

    print(f"Model registered: {model_name} v{mv.version} -> Staging")
    print("training & logging done")



# logged-models API는 MLflow 3.x에서 추가된 기능이라 2.x 서버에서 404가 발생 -> log_model() 방식이 아닌 log_artifact + register_model 방식으로 사용
# mlflow.register_model() 대신 MlflowClient().create_model_version()을 직접 사용