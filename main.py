import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="LOOP 서버")

# 프론트(5173번 주소)에서 이 서버를 부를 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
CASES = {"pressure_vessel", "desk"}  # 지금 결과 파일이 있는 케이스


def check_case(case_id: str):
    if case_id not in CASES:
        raise HTTPException(status_code=404, detail="알 수 없는 케이스예요")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cases/{case_id}/result")
def get_result(case_id: str):
    check_case(case_id)
    path = DATA_DIR / f"{case_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="결과 파일이 없어요")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/cases/{case_id}/image")
def get_image(case_id: str):
    check_case(case_id)
    path = DATA_DIR / f"{case_id}_stress.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="그림이 없어요")
    return FileResponse(path, media_type="image/png")