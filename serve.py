from fastapi import FastAPI
import uvicorn
import os
import datetime

app = FastAPI()


@app.get("/")
def root():
    # 현재 시간을 포함한 로그 출력
    print(f"[{datetime.datetime.now()}] INFO: / root 엔드포인트에 접근함")
    return {"msg": "model server running"}


@app.get("/health")
def health():
    print(f"[{datetime.datetime.now()}] INFO: Health Check 수행 중...")
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    print("-" * 30)
    print(f"서버를 시작합니다! 포트 번호: {port}")
    print(f"문서 확인: http://localhost:{port}/docs")
    print("-" * 30)

    uvicorn.run(app, host="0.0.0.0", port=port)