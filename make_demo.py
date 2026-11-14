import json
import shutil
from pathlib import Path

BASE = Path(__file__).parent
SRC = BASE / "data"                                   # 저장해 둔 기본 결과
DST = BASE.parent / "loop-app" / "public" / "demo"    # 데모용 폴더
DST.mkdir(parents=True, exist_ok=True)


def strip(obj):
    """내 컴퓨터의 폴더 경로가 들어 있는 'files' 항목을 빼요."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "files"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


for case in ["pressure_vessel", "desk", "flange", "beam"]:
    data = json.loads((SRC / f"{case}.json").read_text(encoding="utf-8"))
    (DST / f"{case}.json").write_text(
        json.dumps(strip(data), ensure_ascii=False), encoding="utf-8"
    )
    img = SRC / f"{case}_stress.png"
    if img.exists():
        shutil.copyfile(img, DST / f"{case}_stress.png")
    print("복사 완료:", case)

print("끝났어요 ->", DST)