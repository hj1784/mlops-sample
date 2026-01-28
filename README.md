# MLOps Sample

MLOps 파이프라인 샘플 프로젝트입니다.

## 파일 구조

```
├── train.py         # 학습 코드 (MLflow 연동)
├── eval.py          # 평가 코드 (pass/fail 판정)
├── serve.py         # 서빙 코드 (eval 결과에 따라 실행)
└── requirements.txt # 의존성
```

## 파이프라인 흐름

```
train.py → model.txt 생성 → eval.py → results.json 생성 → serve.py
```

1. **train.py**: 학습 실행, MLflow에 메트릭/아티팩트 기록
2. **eval.py**: model.txt 검증 후 pass/fail 판정
3. **serve.py**: eval 결과가 pass면 서빙 시작

## 사용법

```bash
# 의존성 설치
pip install -r requirements.txt

# 1. 학습
python train.py

# 2. 평가
python eval.py

# 3. 서빙
python serve.py
```

## MLflow 설정

train.py에서 MLflow 서버 주소 설정:

```python
mlflow.set_tracking_uri("http://IP:PORT")
```

또는 환경변수:

```bash
export MLFLOW_TRACKING_URI=http://IP:PORT
```

## 산출물

| 파일 | 설명 |
|------|------|
| model.txt | 학습 결과 (LoRA/checkpoint 자리) |
| /eval_out/results.json | 평가 결과 (pass/score) |
