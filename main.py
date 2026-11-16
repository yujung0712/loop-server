import json
import math
import os
import shutil
import subprocess
import sys
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
DATA_DIR = BASE_DIR / "data"      # 저장해 둔 기본 결과
CASES_DIR = BASE_DIR / "cases"    # 해석 코드와 입력 파일
LAST_DIR = BASE_DIR / "last"      # 케이스마다 마지막으로 성공한 실행 결과 (덮어써요)

# 실행 폴더는 서버 폴더 밖에 둬요 (--reload가 해석 중에 서버를 재시작하지 않게)
RUNS_DIR = Path("C:/Temp/loop_runs")


def _find_solver_python():
    """해석(ANSYS)용 파이썬을 찾아요. 환경변수 LOOP_SOLVER_PYTHON > 내 노트북 경로 > 지금 서버를 돌리는 파이썬 순서예요."""
    candidates = [
        os.environ.get("LOOP_SOLVER_PYTHON", ""),
        r"C:\Users\ghddb\AppData\Local\Programs\Python\Python310\python.exe",
        sys.executable,
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return sys.executable


# 해석용 파이썬 (ANSYS용 도구가 설치된 쪽)
SOLVER_PYTHON = _find_solver_python()

# 저장된 결과가 있는 케이스
SAVED_CASES = {"pressure_vessel", "desk", "flange", "beam", "tsb", "fatigue"}


# ---------- 케이스별 입력값 검사 (입력 파일을 덮어쓴 뒤에 실행해요) ----------

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


def post_desk(problem):
    g = problem["geometry_mm"]
    dv = problem["design_var"]
    leg = dv["leg_size_mm"]
    thk = dv["tabletop_thk_mm"]

    if leg["min"] >= leg["max"]:
        raise HTTPException(status_code=400, detail="다리 굵기 최소는 최대보다 작아야 해요")
    if thk["min"] >= thk["max"]:
        raise HTTPException(status_code=400, detail="상판 두께 최소는 최대보다 작아야 해요")
    if g["total_H"] <= thk["max"] + 100:
        raise HTTPException(
            status_code=400,
            detail="책상 높이는 상판 최대 두께보다 100 mm 이상 커야 해요",
        )
    # 다리는 모서리에서 안쪽으로 들어간 자리에 놓여서, 상판 크기에 비해 너무 굵으면 안 돼요
    room = min(g["tabletop_L"], g["tabletop_W"]) / 2 - g["leg_inset_x"]
    if leg["max"] >= room:
        raise HTTPException(
            status_code=400,
            detail=f"다리 굵기 최대는 {room:.0f} mm보다 작아야 해요 (상판 크기 기준)",
        )


def post_pv(problem):
    """압력용기: 화면에서 받은 최소~최대를 4칸으로 나눠 설계 후보 목록을 만들어요."""
    t = problem.pop("_in", None)  # 임시로 받아 둔 입력값
    g = problem["geometry_mm"]
    space = problem["design_space_mm"]

    if t:
        # (이름, 최소 입력, 최대 입력, 후보 목록 이름, 시작 설계 이름, 시작 설계로 쓸 후보 번호)
        for label, lo_key, hi_key, space_key, geom_key, idx in (
            ("주관 두께", "main_thk_min", "main_thk_max", "main_wall_thk_candidates", "main_wall_thk", 1),
            ("노즐 두께", "nozzle_thk_min", "nozzle_thk_max", "nozzle_wall_thk_candidates", "nozzle_wall_thk", 1),
            ("패드 두께", "pad_thk_min", "pad_thk_max", "pad_thk_candidates", "pad_thk", 2),
        ):
            lo = t.get(lo_key)
            hi = t.get(hi_key)
            if lo is None or hi is None:
                continue
            if lo >= hi:
                raise HTTPException(status_code=400, detail=f"{label} 최소는 최대보다 작아야 해요")
            cands = [round(lo + (hi - lo) * i / 3, 3) for i in range(4)]
            space[space_key] = cands
            g[geom_key] = cands[idx]  # 시작 설계는 후보 목록 안에 있어야 해요

    noz_out_max = g["nozzle_inner_d"] + 2 * max(space["nozzle_wall_thk_candidates"])
    if noz_out_max >= g["main_inner_d"]:
        raise HTTPException(
            status_code=400,
            detail="노즐 바깥지름(안지름 + 2 x 최대 두께)이 주관 안지름보다 작아야 해요",
        )
    if g["pad_outer_d"] <= noz_out_max + 10:
        raise HTTPException(
            status_code=400,
            detail=f"패드 바깥지름은 노즐 바깥지름({noz_out_max:.0f} mm)보다 10 mm 이상 커야 해요",
        )


def post_tsb(problem):
    g = problem["geometry_mm"]
    dv = problem["design_var"]
    lc = problem["mesh_mm"]["lc"]

    if dv["min"] >= dv["max"]:
        raise HTTPException(status_code=400, detail="지름 최소는 최대보다 작아야 해요")
    if g["L_sq"] >= g["L_total"]:
        raise HTTPException(status_code=400, detail="사각 구간 길이는 전체 길이보다 짧아야 해요")
    if dv["min"] < 3 * lc:
        raise HTTPException(
            status_code=400,
            detail=f"지름 최소는 메시 크기({lc} mm)의 3배 이상이어야 해요",
        )
    if problem["optimizer"]["tol_mm"] >= dv["max"] - dv["min"]:
        raise HTTPException(status_code=400, detail="허용오차는 지름 범위보다 작아야 해요")

    # 노드 수 대략 계산 (메시 10 mm 일 때 부피 약 650 mm3 당 노드 1개로 어림해요)
    volume = g["sq_w"] * g["sq_h"] * g["L_sq"] + math.pi * (dv["max"] / 2) ** 2 * (
        g["L_total"] - g["L_sq"]
    )
    est_nodes = volume / (650.0 * (lc / 10.0) ** 3)
    if est_nodes > 100000:
        raise HTTPException(
            status_code=400,
            detail=f"메시가 너무 커요 (예상 노드 약 {int(est_nodes):,}개). 메시 크기를 키우거나 지름 최대를 줄여 주세요. 학생용 ANSYS는 노드 수에 제한이 있어요",
        )


def post_fatigue(problem):
    """피로: 화면에서 받은 최소~최대를 4칸으로 나눠 설계 후보 목록을 만들어요."""
    t = problem.pop("_in", None)  # 임시로 받아 둔 입력값
    g = problem["geometry_mm"]
    space = problem["design_space_mm"]

    if t:
        for label, lo_key, hi_key, space_key, geom_key in (
            ("판 폭", "plate_width_min", "plate_width_max", "plate_width_candidates", "plate_width"),
            ("판 두께", "plate_thk_min", "plate_thk_max", "plate_thk_candidates", "plate_thk"),
            ("구멍 지름", "hole_d_min", "hole_d_max", "hole_d_candidates", "hole_d"),
        ):
            lo = t.get(lo_key)
            hi = t.get(hi_key)
            if lo is None or hi is None:
                continue
            if lo >= hi:
                raise HTTPException(status_code=400, detail=f"{label} 최소는 최대보다 작아야 해요")
            cands = [round(lo + (hi - lo) * i / 3, 3) for i in range(4)]
            space[space_key] = cands
            g[geom_key] = cands[1]  # 시작 설계는 후보 목록 안에 있어야 해요 (두 번째 칸)

    if g["hole_d"] >= 0.8 * g["plate_width"]:
        raise HTTPException(
            status_code=400,
            detail="시작 설계의 구멍 지름이 판 폭의 80% 이상이에요. 구멍 지름 범위를 줄이거나 판 폭 범위를 키워 주세요",
        )

    m = problem["mesh_mm"]
    if m["lc_hole"] > m["lc_global"]:
        raise HTTPException(status_code=400, detail="구멍 주변 메시 크기는 전체 메시 크기보다 작거나 같아야 해요")

    # 노드 수 대략 계산 (판 300x160x14, 메시 8/3 mm 일 때 약 4만 개로 어림해요. 구멍 주변은 빼고 계산한 값이에요)
    volume = g["plate_length"] * max(space["plate_width_candidates"]) * max(space["plate_thk_candidates"])
    est_nodes = 21705 * (volume / 360000.0) * (8.0 / m["lc_global"]) ** 3
    if est_nodes > 70000:
        raise HTTPException(
            status_code=400,
            detail=f"메시가 너무 커요 (예상 노드 약 {int(est_nodes):,}개). 메시 크기를 키우거나 판 크기를 줄여 주세요. 학생용 ANSYS는 노드 수에 제한이 있어요",
        )


# 실제로 해석을 돌릴 수 있는 케이스
# params: 화면 입력 이름 -> (입력 파일 안의 위치, 최솟값, 최댓값)
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
    "desk": {
        "script": "run_DESK_FINAL_no.py",
        "problem_file": "problem.json",
        "out_dir": "outputs_DESK_FINAL_NOCLI/final",
        "result": "summary.json",
        "image": "stress.png",
        "int_keys": {"max_eval"},
        "post": post_desk,
        "params": {
            "leg_size_mm_min": (["design_var", "leg_size_mm", "min"], 10, 150),
            "leg_size_mm_max": (["design_var", "leg_size_mm", "max"], 10, 150),
            "tabletop_thk_mm_min": (["design_var", "tabletop_thk_mm", "min"], 5, 100),
            "tabletop_thk_mm_max": (["design_var", "tabletop_thk_mm", "max"], 5, 100),
            "stress_limit_MPa": (["constraint", "stress_limit_MPa"], 0.1, 1000),
            "max_eval": (["optimizer", "max_eval"], 5, 200),
            "tabletop_length": (["geometry_mm", "tabletop_L"], 300, 3000),
            "tabletop_width": (["geometry_mm", "tabletop_W"], 300, 2000),
            "desk_height": (["geometry_mm", "total_H"], 300, 1500),
            "load_N": (["load", "total_N"], 1, 100000),
            "E_MPa": (["material_MPa", "E"], 1000, 300000),
        },
    },
    "pressure_vessel": {
        "script": "F_NSEL_P_V_VER.py",
        "problem_file": "problem.json",
        "out_dir": "outputs_CROSS_PRESSURE_VESSEL_COORD/final",
        "result": "summary.json",
        "image": "stress.png",
        "int_keys": {"max_eval"},
        "post": post_pv,
        "params": {
            # 설계 변수: 최소~최대를 입력받아 post_pv가 4칸 후보로 나눠요 ("_in"은 임시 보관함)
            "main_thk_min": (["_in", "main_thk_min"], 5, 80),
            "main_thk_max": (["_in", "main_thk_max"], 5, 80),
            "nozzle_thk_min": (["_in", "nozzle_thk_min"], 3, 60),
            "nozzle_thk_max": (["_in", "nozzle_thk_max"], 3, 60),
            "pad_thk_min": (["_in", "pad_thk_min"], 3, 60),
            "pad_thk_max": (["_in", "pad_thk_max"], 3, 60),
            "stress_limit_MPa": (["constraint", "stress_limit_MPa"], 1, 2000),
            "weight_limit_N": (["constraint", "self_weight_limit_N"], 100, 1000000),
            # 이 코드는 해석 횟수가 이 값에 닿으면 결과 없이 에러로 끝나서 넉넉하게 둬야 해요
            "max_eval": (["optimizer", "max_eval"], 20, 200),
            "main_length": (["geometry_mm", "main_length"], 500, 4000),
            "main_inner_d": (["geometry_mm", "main_inner_d"], 200, 1000),
            "nozzle_length": (["geometry_mm", "nozzle_length"], 200, 1000),
            "nozzle_inner_d": (["geometry_mm", "nozzle_inner_d"], 50, 300),
            "pad_outer_d": (["geometry_mm", "pad_outer_d"], 100, 500),
            "pressure_MPa": (["load", "internal_pressure_MPa"], 0.1, 50),
        },
    },
    "tsb": {
        "script": "run_TSB_FAST_FINAL_no.py",
        "problem_file": "problem.json",
        "out_dir": "outputs_TSB_FINAL_NOCLI/final",
        "result": "summary.json",
        "image": "deformed.png",
        "int_keys": {"max_iter"},
        "post": post_tsb,
        "params": {
            "cyl_d_min": (["design_var", "min"], 20, 150),
            "cyl_d_max": (["design_var", "max"], 20, 200),
            "u_limit_mm": (["constraint", "U_limit_mm"], 0.01, 100),
            "tol_mm": (["optimizer", "tol_mm"], 0.01, 10),
            "max_iter": (["optimizer", "max_iter"], 3, 40),
            "L_total": (["geometry_mm", "L_total"], 300, 3000),
            "L_sq": (["geometry_mm", "L_sq"], 100, 2000),
            "sq_w": (["geometry_mm", "sq_w"], 30, 300),
            "sq_h": (["geometry_mm", "sq_h"], 30, 300),
            "load_N": (["load", "total_N"], 100, 10000000),
            "E_MPa": (["material_MPa", "E"], 1000, 1000000),
            "lc_mm": (["mesh_mm", "lc"], 6, 30),
        },
    },
    "fatigue": {
        "script": "fatigue_2_sn_table.py",
        "problem_file": "problem.json",
        "out_dir": "outputs_PLATE_WITH_HOLE_FATIGUE_CASE1/final",
        "result": "summary.json",
        "image": "stress_max.png",
        "int_keys": {"max_eval"},
        "post": post_fatigue,
        "params": {
            # 설계 변수: 최소~최대를 입력받아 post_fatigue가 4칸 후보로 나눠요 ("_in"은 임시 보관함)
            "plate_width_min": (["_in", "plate_width_min"], 60, 400),
            "plate_width_max": (["_in", "plate_width_max"], 60, 400),
            "plate_thk_min": (["_in", "plate_thk_min"], 3, 60),
            "plate_thk_max": (["_in", "plate_thk_max"], 3, 60),
            "hole_d_min": (["_in", "hole_d_min"], 4, 200),
            "hole_d_max": (["_in", "hole_d_max"], 4, 200),
            "min_life_cycles": (["constraint", "min_life_cycles"], 1000, 1000000000),
            "max_mass_kg": (["constraint", "max_mass_kg"], 0.1, 100),
            # 이 코드는 해석 횟수가 이 값에 닿으면 결과 없이 에러로 끝나서 넉넉하게 둬야 해요
            "max_eval": (["optimizer", "max_eval"], 20, 200),
            "plate_length": (["geometry_mm", "plate_length"], 100, 1000),
            "ux_max_mm": (["fatigue", "ux_max_mm"], 0.01, 2),
            "E_MPa": (["material_MPa", "E"], 1000, 1000000),
            "Su_MPa": (["material_MPa", "Su"], 100, 3000),
            "lc_global_mm": (["mesh_mm", "lc_global"], 6, 30),
            "lc_hole_mm": (["mesh_mm", "lc_hole"], 2, 15),
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


# ---------- 마지막 실행 결과 저장 ----------

def last_paths(case_id: str):
    d = LAST_DIR / case_id
    return d / "result.json", d / "image.png"


def find_run_image(r):
    """실행 폴더에서 응력 그림을 찾아요. 정해진 이름이 없으면 첫 번째 png를 써요."""
    out_dir = r["dir"] / r["cfg"]["out_dir"]
    path = out_dir / r["cfg"]["image"]
    if path.exists():
        return path
    pngs = sorted(out_dir.glob("*.png"))
    return pngs[0] if pngs else None


def save_last(run_id: str, r):
    """정상으로 끝난 실행을 이 케이스의 '마지막 결과'로 저장해요 (이전 것은 덮어써요)."""
    try:
        cfg = r["cfg"]
        out_dir = r["dir"] / cfg["out_dir"]
        data = json.loads((out_dir / cfg["result"]).read_text(encoding="utf-8"))
        data["_meta"] = {
            "source": "last",
            "run_id": run_id,
            "finished_at": time.strftime("%m/%d %H:%M"),
            "params": r["params"],
        }
        result_path, image_path = last_paths(r["case"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        img = find_run_image(r)
        if img is not None:
            shutil.copyfile(img, image_path)
        elif image_path.exists():
            image_path.unlink()  # 이번 실행에 그림이 없으면 옛 그림을 남기지 않아요
    except Exception as e:
        print(f"[LAST] 마지막 결과 저장에 실패했어요: {e}")


def finalize_run(run_id: str):
    """해석이 끝나길 기다렸다가, 성공했으면 마지막 결과로 저장해요 (브라우저를 닫아도 동작해요)."""
    r = RUNS.get(run_id)
    if r is None:
        return
    r["proc"].wait()
    r["ended"] = time.time()
    time.sleep(1)
    if run_status(r) == "done":
        save_last(run_id, r)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cases/{case_id}/result")
def get_result(case_id: str):
    check_case(case_id)
    # 마지막으로 성공한 실행이 있으면 그 결과를 줘요
    last_json, _ = last_paths(case_id)
    if last_json.exists():
        return json.loads(last_json.read_text(encoding="utf-8"))
    # 없으면 처음에 저장해 둔 기본 결과를 줘요
    path = DATA_DIR / f"{case_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="결과 파일이 없어요")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_meta"] = {"source": "saved"}
    return data


@app.get("/api/cases/{case_id}/image")
def get_image(case_id: str):
    check_case(case_id)
    last_json, last_img = last_paths(case_id)
    if last_json.exists():
        if last_img.exists():
            return FileResponse(last_img, media_type="image/png")
        raise HTTPException(status_code=404, detail="그림이 없어요")
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
            node = node.setdefault(p, {})
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


# 로그에서 숨길 줄 (그림 도구 경고)
HIDE_PARTS = ("PyVistaFutureWarning", "self._surf_cache", "silence this warning")


def read_log_lines(run_dir: Path):
    path = run_dir / "run.log"
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        # Gmsh의 긴 안내 줄은 빼고 보여줘요
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
        # 직접 돌려 보며 생긴 결과 폴더(outputs_*)는 복사하지 않아요
        shutil.copytree(
            CASES_DIR / case_id,
            run_dir,
            ignore=shutil.ignore_patterns("outputs_*", "__pycache__"),
        )

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
        # 끝나면 알아서 '마지막 결과'로 저장하는 일꾼을 붙여요
        threading.Thread(target=finalize_run, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "running", "params": params}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="알 수 없는 실행이에요")
    lines = read_log_lines(r["dir"])
    # 설계 하나를 시작할 때 찍히는 줄: [EVAL ...] (최적화 코드) 또는 [ITER ...] (TSB)
    starts = ("[EVAL", "[ITER")
    return {
        "run_id": run_id,
        "case": r["case"],
        "status": run_status(r),
        "params": r["params"],
        "elapsed_s": round(r.get("ended", time.time()) - r["started"], 1),
        "evals_started": sum(1 for l in lines if l.startswith(starts)),
        "results": [l for l in lines if l.startswith("[RESULT]")],
        "eval_lines": [l for l in lines if l.startswith(starts)],
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
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_meta"] = {"source": "run", "run_id": run_id, "params": r["params"]}
    return data


@app.get("/api/runs/{run_id}/image")
def get_run_image(run_id: str):
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="알 수 없는 실행이에요")
    path = find_run_image(r)
    if path is None:
        raise HTTPException(status_code=404, detail="그림이 없어요")
    return FileResponse(path, media_type="image/png")