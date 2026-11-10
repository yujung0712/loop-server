import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="LOOP 서버")

# 프론트(5173번 주소)에서 이 서버를 부를 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"      # 저장해 둔 결과
CASES_DIR = BASE_DIR / "cases"    # 해석 코드와 입력 파일

# 실행 폴더는 서버 폴더 밖에 둬요 (--reload가 해석 중에 서버를 재시작하지 않게)
RUNS_DIR = Path("C:/Temp/loop_runs")

# 해석 코드를 돌릴 파이썬 (ANSYS용 도구가 설치된 쪽)
SOLVER_PYTHON = r"C:\Users\ghddb\AppData\Local\Programs\Python\Python310\python.exe"

# 저장된 결과가 있는 케이스
SAVED_CASES = {"pressure_vessel", "desk", "flange", "beam"}


# ---------- 케이스별 입력값 검사 (problem.json을 덮어쓴 뒤에 실행해요) ----------

def post_flange(problem):
    g = problem["geometry_mm"]
    # 플랜지 안지름은 파이프 안지름과 같게 맞춰요
    g["flange_inner_d"] = g["pipe_inner_d"]

    thk = problem["design_var"]["pipe_wall_thk_mm"]
    if thk["min"] >= thk["max"]:
        raise HTTPException(status_code=400, detail="두께 최소는 최대보다 작아야 해요")
    if g["pipe_inner_d"] + 2 * thk["max"] >= g["flange_outer_d"]:
        raise HTTPException(
            status_code=400,
            detail="파이프 바깥지름(내경 + 2 x 최대 두께)이 플랜지 외경보다 작아야 해요",
        )


def post_beam(problem):
    g = problem["geometry_mm"]
    lc = problem["mesh_mm"]["lc_global"]

    if problem["load"]["F_tip_total_N"] == 0:
        raise HTTPException(status_code=400, detail="하중은 0이 아니어야 해요")
    if lc > min(g["beam_width"], g["beam_height"]) / 2:
        raise HTTPException(
            status_code=400,
            detail="메시 크기는 보 폭과 높이 중 작은 쪽의 절반 이하여야 해요",
        )
    # 노드 수 대략 계산 (1000x50x50 보, 메시 10 mm 일 때 22,219개를 기준으로 비례)
    volume = g["beam_length"] * g["beam_width"] * g["beam_height"]
    est_nodes = 22219 * (volume / 2.5e6) * (10 / lc) ** 3
    if est_nodes > 100000:
        raise HTTPException(
            status_code=400,
            detail=f"메시가 너무 커요 (예상 노드 약 {int(est_nodes):,}개). 메시 크기를 키우거나 보 크기를 줄여 주세요. 학생용 ANSYS는 노드 수에 제한이 있어요",
        )


# 실제로 해석을 돌릴 수 있는 케이스
# params: 화면 입력 이름 -> (problem 파일 안의 위치, 최솟값, 최댓값)
# 최솟값/최댓값은 엉뚱한 입력을 막으려고 임시로 정한 안전 범위예요 (필요하면 바꿔도 돼요)
RUNNERS = {
    "flange": {
        "script": "run_FLAN_FINAL_no.py",
        "problem_file": "problem.json",
        "out_dir": "outputs_FLANGED_PIPE_CASE1_NOCLI/final",
        "result": "summary.json",
        "image": "stress.png",
        "int_keys": {"bolt_holes"},
        "post": post_flange,
        "params": {
            "pipe_thk_min": (["design_var", "pipe_wall_thk_mm", "min"], 2, 40),
            "pipe_thk_max": (["design_var", "pipe_wall_thk_mm", "max"], 2, 40),
            "stress_limit_MPa": (["constraint", "stress_limit_MPa"], 1, 1000),
            "tol_mm": (["optimizer", "thk_bisect_tol_mm"], 0.1, 5),
            "length": (["geometry_mm", "length_total"], 200, 1500),
            "inner_d": (["geometry_mm", "pipe_inner_d"], 40, 200),
            "flange_outer_d": (["geometry_mm", "flange_outer_d"], 100, 400),
            "flange_thk": (["geometry_mm", "flange_thk"], 5, 60),
            "bolt_holes": (["geometry_mm", "n_bolt_holes"], 4, 16),
            "pressure_MPa": (["load", "internal_pressure_MPa"], 0.1, 50),
        },
    },
    "beam": {
        "script": "cantilever_tet10_physical_force.py",
        "problem_file": "problem_beam.json",
        "out_dir": "outputs_CANTILEVER_TET10_BEAM",
        "result": "result_beam.json",
        "image": "seqv.png",
        "int_keys": set(),
        "post": post_beam,
        "params": {
            "length": (["geometry_mm", "beam_length"], 100, 3000),
            "width": (["geometry_mm", "beam_width"], 10, 200),
            "height": (["geometry_mm", "beam_height"], 10, 200),
            "E_MPa": (["material_MPa", "E"], 1000, 1000000),
            "nu": (["material_MPa", "nu"], 0.1, 0.49),
            "load_N": (["load", "F_tip_total_N"], -100000, 100000),
            "lc_mm": (["mesh_mm", "lc_global"], 7, 50),
        },
    },
}

RUNS = {}  # run_id -> 실행 정보 (서버를 끄면 사라져요)
LOCK = threading.Lock()


class RunRequest(BaseModel):
    params: Dict[str, float] = {}


def check_case(case_id: str):
    if case_id not in SAVED_CASES:
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


# ---------- 해석 실행 ----------

def build_problem(case_id: str, params: Dict[str, float]):
    """서버에 둔 입력 파일을 읽어서, 화면에서 받은 값으로 덮어써요."""
    cfg = RUNNERS[case_id]
    problem = json.loads(
        (CASES_DIR / case_id / cfg["problem_file"]).read_text(encoding="utf-8")
    )
    table = cfg["params"]

    for key, value in params.items():
        if key not in table:
            raise HTTPException(status_code=400, detail=f"알 수 없는 입력 이름이에요: {key}")
        path, lo, hi = table[key]
        if not (lo <= value <= hi):
            raise HTTPException(status_code=400, detail=f"{key} 값은 {lo} ~ {hi} 사이여야 해요")
        if key in cfg["int_keys"]:
            value = int(round(value))
        node = problem
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = value

    cfg["post"](problem)
    return problem


def busy_run_id():
    for rid, r in RUNS.items():
        if r["proc"].poll() is None:
            return rid
    return None


def run_status(r):
    code = r["proc"].poll()
    if code is None:
        return "running"
    result_path = r["dir"] / r["cfg"]["out_dir"] / r["cfg"]["result"]
    return "done" if code == 0 and result_path.exists() else "failed"


# 로그에서 숨길 줄 (Gmsh 안내, 그림 도구 경고)
HIDE_PARTS = ("PyVistaFutureWarning", "self._surf_cache", "silence this warning")


def read_log_lines(run_dir: Path):
    path = run_dir / "run.log"
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("Info") or s.startswith("*") or s.startswith("="):
            continue
        if any(part in s for part in HIDE_PARTS) or s.startswith("`"):
            continue
        lines.append(line.rstrip())
    return lines


@app.post("/api/cases/{case_id}/run")
def start_run(case_id: str, req: Optional[RunRequest] = None):
    if case_id not in RUNNERS:
        raise HTTPException(status_code=400, detail="이 케이스는 아직 실행할 수 없어요")
    if not Path(SOLVER_PYTHON).exists():
        raise HTTPException(status_code=500, detail=f"해석용 파이썬을 찾을 수 없어요: {SOLVER_PYTHON}")

    params = req.params if req else {}

    with LOCK:
        busy = busy_run_id()
        if busy:
            raise HTTPException(status_code=409, detail=f"다른 해석이 실행 중이에요: {busy}")

        # 입력값 검사는 폴더를 만들기 전에 해요
        problem = build_problem(case_id, params)

        cfg = RUNNERS[case_id]
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + case_id
        run_dir = RUNS_DIR / run_id
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CASES_DIR / case_id, run_dir)

        # 실행 폴더의 입력 파일을 입력값이 반영된 것으로 바꿔요
        (run_dir / cfg["problem_file"]).write_text(
            json.dumps(problem, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        env.pop("VIRTUAL_ENV", None)

        log_file = open(run_dir / "run.log", "wb")
        proc = subprocess.Popen(
            [SOLVER_PYTHON, cfg["script"]],
            cwd=str(run_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log_file.close()

        RUNS[run_id] = {
            "case": case_id,
            "cfg": cfg,
            "dir": run_dir,
            "proc": proc,
            "params": params,
            "started": time.time(),
        }
    return {"run_id": run_id, "status": "running", "params": params}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="알 수 없는 실행이에요")
    lines = read_log_lines(r["dir"])
    return {
        "run_id": run_id,
        "case": r["case"],
        "status": run_status(r),
        "params": r["params"],
        "elapsed_s": round(time.time() - r["started"], 1),
        "evals_started": sum(1 for l in lines if l.startswith("[EVAL")),
        "results": [l for l in lines if l.startswith("[RESULT]")],
        "log": lines[-30:],
    }


@app.get("/api/runs/{run_id}/result")
def get_run_result(run_id: str):
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="알 수 없는 실행이에요")
    if run_status(r) != "done":
        raise HTTPException(status_code=409, detail="아직 끝나지 않았거나 실패했어요")
    path = r["dir"] / r["cfg"]["out_dir"] / r["cfg"]["result"]
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/image")
def get_run_image(run_id: str):
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="알 수 없는 실행이에요")
    path = r["dir"] / r["cfg"]["out_dir"] / r["cfg"]["image"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="그림이 없어요")
    return FileResponse(path, media_type="image/png")