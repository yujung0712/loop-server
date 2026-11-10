import os
import json
import time
import math
import shutil
import numpy as np
import gmsh
from ansys.mapdl.core import launch_mapdl


# ============================================================
# utils
# ============================================================
def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_copy(src: str, dst: str):
    if src and os.path.exists(src):
        shutil.copy2(src, dst)


def compress_ranges(sorted_ids: np.ndarray):
    if sorted_ids.size == 0:
        return []

    out = []
    s = int(sorted_ids[0])
    e = int(sorted_ids[0])

    for x in sorted_ids[1:]:
        x = int(x)
        if x == e + 1:
            e = x
        else:
            out.append((s, e))
            s = e = x

    out.append((s, e))
    return out


def make_component_from_nodes(mapdl, comp_name: str, node_ids: np.ndarray):
    node_ids = np.array(node_ids, dtype=int)
    node_ids.sort()

    ranges = compress_ranges(node_ids)
    if len(ranges) == 0:
        raise RuntimeError(f"{comp_name} node list is empty")

    mapdl.allsel()

    s0, e0 = ranges[0]
    mapdl.nsel("S", "NODE", "", s0, e0)

    for s, e in ranges[1:]:
        mapdl.nsel("A", "NODE", "", s, e)

    mapdl.cm(comp_name, "NODE")
    mapdl.allsel()


def frange(start: float, stop: float, step: float):
    vals = []
    x = float(start)
    step = float(step)

    if step <= 0.0:
        raise ValueError("step must be > 0")

    while x <= stop + 1e-9:
        vals.append(round(x, 6))
        x += step

    return vals


def desk_volume_mm3(tabletop_L, tabletop_W, total_H, leg_size, tabletop_thk):
    leg_h = total_H - tabletop_thk
    if leg_h <= 0.0:
        return float("inf")

    v_top = tabletop_L * tabletop_W * tabletop_thk
    v_legs = 4.0 * leg_size * leg_size * leg_h
    return float(v_top + v_legs)


def cleanup_old_lock(run_location: str, jobname_prefix: str):
    lock_path = os.path.join(run_location, f"{jobname_prefix}.lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            log(f"[LOCK] removed old lock: {lock_path}")
        except Exception:
            log(f"[LOCK] could not remove lock: {lock_path}")
            log("[LOCK] if MAPDL is still running, close it in Task Manager and run again.")


# ============================================================
# Gmsh build + extract
# ============================================================
def gmsh_build_extract_desk(iter_dir: str, cfg: dict, leg_size: float, tabletop_thk: float):
    geo = cfg["geometry_mm"]
    mesh_cfg = cfg["mesh_mm"]
    tags = cfg["physical_tags"]
    outputs = cfg["outputs"]

    tabletop_L = float(geo["tabletop_L"])
    tabletop_W = float(geo["tabletop_W"])
    total_H = float(geo["total_H"])
    inset_x = float(geo["leg_inset_x"])
    inset_y = float(geo["leg_inset_y"])

    lc = float(mesh_cfg["lc"])

    BASE = int(tags["BASE"])
    TOP = int(tags["TOP"])
    VOL = int(tags["VOL"])

    leg_size = float(leg_size)
    tabletop_thk = float(tabletop_thk)
    leg_h = total_H - tabletop_thk

    if leg_h <= 0.0:
        raise RuntimeError("tabletop_thk must be smaller than total_H")

    if 2.0 * (inset_x + leg_size) >= tabletop_L:
        raise RuntimeError("leg_size/inset_x combination is too large for tabletop_L")

    if 2.0 * (inset_y + leg_size) >= tabletop_W:
        raise RuntimeError("leg_size/inset_y combination is too large for tabletop_W")

    step_path = os.path.join(iter_dir, "model.step")
    msh_path = os.path.join(iter_dir, "mesh.msh")

    log("  [GMSH] initialize")
    gmsh.initialize()
    gmsh.model.add("DESK")
    occ = gmsh.model.occ

    # coordinate system:
    # z = vertical
    # floor at z = 0
    # tabletop bottom at z = leg_h
    # tabletop top at z = total_H
    x_min = -tabletop_L / 2.0
    y_min = -tabletop_W / 2.0

    # tabletop
    tabletop = occ.addBox(
        x_min,
        y_min,
        leg_h,
        tabletop_L,
        tabletop_W,
        tabletop_thk
    )

    # leg centers (box corner origins)
    leg_x1 = x_min + inset_x
    leg_x2 = x_min + tabletop_L - inset_x - leg_size
    leg_y1 = y_min + inset_y
    leg_y2 = y_min + tabletop_W - inset_y - leg_size

    legs = []
    for lx in [leg_x1, leg_x2]:
        for ly in [leg_y1, leg_y2]:
            leg = occ.addBox(lx, ly, 0.0, leg_size, leg_size, leg_h)
            legs.append(leg)

    log("  [GMSH] fuse solids")
    obj = [(3, tabletop)]
    tool = [(3, t) for t in legs]
    occ.fuse(obj, tool)
    occ.synchronize()

    log("  [GMSH] detect BASE/TOP surfaces")
    base_surfs = []
    top_surfs = []

    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin - 0.0) < 1e-6 and abs(zmax - 0.0) < 1e-6:
            base_surfs.append(tag)

        if abs(zmin - total_H) < 1e-6 and abs(zmax - total_H) < 1e-6:
            top_surfs.append(tag)

    if not base_surfs:
        gmsh.finalize()
        raise RuntimeError("BASE surface not found")

    if not top_surfs:
        gmsh.finalize()
        raise RuntimeError("TOP surface not found")

    gmsh.model.addPhysicalGroup(2, base_surfs, BASE)
    gmsh.model.setPhysicalName(2, BASE, "BASE")

    gmsh.model.addPhysicalGroup(2, top_surfs, TOP)
    gmsh.model.setPhysicalName(2, TOP, "TOP")

    vols = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v for _, v in vols], VOL)
    gmsh.model.setPhysicalName(3, VOL, "BODY")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    log("  [GMSH] mesh.generate(3) start")
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    log(f"  [GMSH] mesh.generate(3) done in {time.time() - t0:.2f}s")

    if outputs.get("save_step", True):
        log("  [GMSH] write STEP start")
        gmsh.write(step_path)
        log("  [GMSH] write STEP done")
    else:
        step_path = ""

    if outputs.get("save_msh", True):
        log("  [GMSH] write MSH start")
        gmsh.write(msh_path)
        log("  [GMSH] write MSH done")
    else:
        msh_path = ""

    log("  [GMSH] getNodes start")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_xyz = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    log(f"  [GMSH] getNodes done: nnode={len(node_tags)}")

    log("  [GMSH] getElements start")
    vol_entities = gmsh.model.getEntitiesForPhysicalGroup(3, VOL)
    tet_conn = []

    for ent in vol_entities:
        types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, ent)

        for etype, enodes in zip(types, elem_node_tags):
            enodes = np.asarray(enodes, dtype=np.int64)
            if enodes.size % 4 == 0:
                tet_conn.append(enodes.reshape(-1, 4))

    if not tet_conn:
        gmsh.finalize()
        raise RuntimeError("tet4 connectivity not found")

    tet_conn = np.vstack(tet_conn)
    log(f"  [GMSH] getElements done: nelem={len(tet_conn)}")

    def get_phys_surface_nodes(phys_tag: int):
        surf_entities = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
        nodes = set()

        for ent in surf_entities:
            types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, ent)
            for enodes in elem_node_tags:
                nodes.update(np.asarray(enodes, dtype=np.int64).tolist())

        return np.array(sorted(nodes), dtype=np.int64)

    log("  [GMSH] get BASE/TOP nodes start")
    base_nodes = get_phys_surface_nodes(BASE)
    top_nodes = get_phys_surface_nodes(TOP)
    log(f"  [GMSH] get BASE/TOP nodes done: nbase={len(base_nodes)}, ntop={len(top_nodes)}")

    gmsh.finalize()
    log("  [GMSH] finalize done")

    return {
        "step_path": step_path,
        "msh_path": msh_path,
        "node_tags": node_tags,
        "node_xyz": node_xyz,
        "tet_conn": tet_conn,
        "base_nodes": base_nodes,
        "top_nodes": top_nodes
    }


# ============================================================
# write local APDL build file
# ============================================================
def write_mapdl_build_inp(work_dir: str, eval_idx: int, node_tags: np.ndarray, node_xyz: np.ndarray, tet_conn: np.ndarray):
    inp_path = os.path.join(work_dir, f"build_eval_{eval_idx:03d}.inp")

    with open(inp_path, "w", encoding="ascii", errors="ignore") as f:
        f.write("/PREP7\n")

        for nid, xyz in zip(node_tags, node_xyz):
            f.write(f"N,{int(nid)},{float(xyz[0])},{float(xyz[1])},{float(xyz[2])}\n")

        for conn in tet_conn:
            n1, n2, n3, n4 = map(int, conn.tolist())
            f.write(f"E,{n1},{n2},{n3},{n4}\n")

        f.write("FINISH\n")

    return inp_path


# ============================================================
# MAPDL solve
# ============================================================
def solve_one_eval(mapdl, eval_idx: int, eval_dir: str, gm: dict, cfg: dict):
    mat = cfg["material_MPa"]
    load = cfg["load"]
    cons = cfg["constraint"]
    outputs = cfg["outputs"]

    E = float(mat["E"])
    nu = float(mat["nu"])

    F_total = float(load["total_N"])
    dof = load["direction"].upper()
    sign = float(load.get("sign", 1.0))

    stress_limit = float(cons["stress_limit_MPa"])

    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    base_nodes = gm["base_nodes"].astype(int)
    top_nodes = gm["top_nodes"].astype(int)

    mapdl_work_dir = mapdl.directory
    cdb_base_name = f"desk_eval_{eval_idx:03d}"
    cdb_work_base = os.path.join(mapdl_work_dir, cdb_base_name)
    cdb_work_path_lower = os.path.join(mapdl_work_dir, f"{cdb_base_name}.cdb")
    cdb_work_path_upper = os.path.join(mapdl_work_dir, f"{cdb_base_name}.CDB")
    cdb_eval_path = os.path.join(eval_dir, "model.cdb")
    png_path = os.path.join(eval_dir, "stress.png")

    build_inp_path = write_mapdl_build_inp(
        mapdl_work_dir,
        eval_idx,
        node_tags,
        node_xyz,
        tet_conn
    )

    log("    [MAPDL] clear")
    mapdl.clear()

    log("    [MAPDL] prep7 + material setup")
    mapdl.prep7()
    mapdl.et(1, 285)
    mapdl.type(1)
    mapdl.mat(1)
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, nu)
    mapdl.finish()

    log(f"    [MAPDL] input build file start: {build_inp_path}")
    t0 = time.time()
    mapdl.input(build_inp_path)
    log(f"    [MAPDL] input build file done in {time.time() - t0:.2f}s")

    log("    [MAPDL] remake components")
    mapdl.prep7()
    make_component_from_nodes(mapdl, "BASE", base_nodes)
    make_component_from_nodes(mapdl, "TOP", top_nodes)
    mapdl.finish()
    log("    [MAPDL] components done")

    if outputs.get("save_cdb", True):
        log("    [MAPDL] CDWRITE start")
        mapdl.prep7()
        mapdl.cdwrite("DB", cdb_work_base, "cdb")
        mapdl.finish()

        if os.path.exists(cdb_work_path_lower):
            safe_copy(cdb_work_path_lower, cdb_eval_path)
        elif os.path.exists(cdb_work_path_upper):
            safe_copy(cdb_work_path_upper, cdb_eval_path)

        log("    [MAPDL] CDWRITE done")

    log("    [MAPDL] CLEAR + CDREAD start")
    mapdl.clear()
    mapdl.prep7()
    mapdl.cdread("DB", cdb_work_base, "cdb")
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, nu)
    mapdl.finish()
    log("    [MAPDL] CLEAR + CDREAD done")

    log("    [MAPDL] SOLU start")
    mapdl.slashsolu()
    mapdl.antype("STATIC")

    mapdl.cmsel("S", "BASE")
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    mapdl.cmsel("S", "TOP")
    mapdl.run("*GET,NTOP,NODE,0,COUNT")
    n_top = int(mapdl.parameters["NTOP"])

    if n_top <= 0:
        raise RuntimeError("TOP node count is zero")

    f_each = sign * (F_total / float(n_top))
    mapdl.f("ALL", dof, f_each)
    mapdl.allsel()

    log(f"      TOP nodes={n_top}, each {dof}={f_each:.6e}, total={f_each * n_top:.6e}")

    t0 = time.time()
    mapdl.solve()
    log(f"    [MAPDL] solve done in {time.time() - t0:.2f}s")
    mapdl.finish()

    log("    [MAPDL] POST1 start")
    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()

    # nodal equivalent stress
    seqv = np.asarray(mapdl.post_processing.nodal_eqv_stress(), dtype=float).reshape(-1)
    all_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)

    if seqv.size == 0:
        raise RuntimeError("POST: nodal equivalent stress array is empty")

    if all_nodes.size != seqv.size:
        # if for some reason node count differs, still compute max stress safely
        valid = np.isfinite(seqv)
        seqv_valid = seqv[valid]
        if seqv_valid.size == 0:
            raise RuntimeError("POST: nodal equivalent stress contains no finite values")
        smax = float(np.max(seqv_valid))
        smin = float(np.min(seqv_valid))
        node_at_smax = -1
    else:
        valid = np.isfinite(seqv)
        seqv_valid = seqv[valid]
        nodes_valid = all_nodes[valid]

        if seqv_valid.size == 0:
            raise RuntimeError("POST: nodal equivalent stress contains no finite values")

        idx = int(np.argmax(seqv_valid))
        smax = float(seqv_valid[idx])
        smin = float(np.min(seqv_valid))
        node_at_smax = int(nodes_valid[idx])

    feasible = bool(smax <= stress_limit)

    png_ok = False
    if outputs.get("save_png", True):
        try:
            log("    [MAPDL] save PNG start")
            mapdl.post_processing.plot_nodal_eqv_stress(
                show_edges=True,
                off_screen=True,
                savefig=png_path
            )
            png_ok = os.path.exists(png_path)
            log(f"    [MAPDL] save PNG done: {png_ok}")
        except Exception as e1:
            try:
                log(f"    [MAPDL] eqv stress plot failed, fallback to displacement plot: {e1}")
                mapdl.post_processing.plot_nodal_displacement(
                    component="Z",
                    show_edges=True,
                    off_screen=True,
                    savefig=png_path
                )
                png_ok = os.path.exists(png_path)
                log(f"    [MAPDL] fallback PNG done: {png_ok}")
            except Exception as e2:
                log(f"    [MAPDL] save PNG failed: {e2}")
                png_ok = False

    mapdl.finish()
    log("    [MAPDL] POST1 done")

    return {
        "Smax_MPa": smax,
        "Smin_MPa": smin,
        "node_at_Smax": node_at_smax,
        "feasible": feasible,
        "n_top": n_top,
        "f_each": f_each,
        "cdb_path": cdb_eval_path if outputs.get("save_cdb", True) else "",
        "png_path": png_path if png_ok else ""
    }


# ============================================================
# evaluate one design
# ============================================================
def evaluate_design(mapdl, out_root: str, cfg: dict, eval_idx: int, leg_size: float, tabletop_thk: float):
    eval_dir = os.path.join(out_root, f"eval_{eval_idx:03d}")
    ensure_dir(eval_dir)

    tabletop_L = float(cfg["geometry_mm"]["tabletop_L"])
    tabletop_W = float(cfg["geometry_mm"]["tabletop_W"])
    total_H = float(cfg["geometry_mm"]["total_H"])
    stress_limit = float(cfg["constraint"]["stress_limit_MPa"])

    log("\n============================================================")
    log(f"[EVAL {eval_idx:03d}] leg_size = {leg_size:.6f} mm | tabletop_thk = {tabletop_thk:.6f} mm")
    log("============================================================")

    log("[1/3] Gmsh build + extract start")
    gm = gmsh_build_extract_desk(eval_dir, cfg, leg_size, tabletop_thk)
    log("[1/3] Gmsh build + extract done")

    log("[2/3] MAPDL build->CDB->CDREAD->solve->post start")
    res = solve_one_eval(mapdl, eval_idx, eval_dir, gm, cfg)
    log("[2/3] MAPDL build->CDB->CDREAD->solve->post done")

    vol_mm3 = desk_volume_mm3(tabletop_L, tabletop_W, total_H, leg_size, tabletop_thk)

    rec = {
        "eval": eval_idx,
        "leg_size_mm": leg_size,
        "tabletop_thk_mm": tabletop_thk,
        "volume_mm3": vol_mm3,
        "Smax_MPa": res["Smax_MPa"],
        "Smin_MPa": res["Smin_MPa"],
        "stress_limit_MPa": stress_limit,
        "node_at_Smax": res["node_at_Smax"],
        "feasible": res["feasible"],
        "n_top": res["n_top"],
        "f_each": res["f_each"],
        "files": {
            "step": gm["step_path"],
            "msh": gm["msh_path"],
            "cdb": res["cdb_path"],
            "png": res["png_path"]
        }
    }

    write_json(os.path.join(eval_dir, "result.json"), rec)

    log(
        f"[RESULT] Smax={rec['Smax_MPa']:.6e} MPa | "
        f"limit={rec['stress_limit_MPa']:.6e} MPa | "
        f"feasible={rec['feasible']} | "
        f"volume={rec['volume_mm3']:.6e} mm^3"
    )

    return rec


# ============================================================
# search helpers
# ============================================================
def get_cached_or_eval(mapdl, out_root, cfg, leg_size, tabletop_thk, state):
    key = (round(float(leg_size), 6), round(float(tabletop_thk), 6))
    if key in state["cache"]:
        log(f"[CACHE] hit for leg={leg_size:.6f}, thk={tabletop_thk:.6f}")
        return state["cache"][key]

    if state["eval_count"] >= state["max_eval"]:
        raise RuntimeError(f"max_eval reached: {state['max_eval']}")

    rec = evaluate_design(
        mapdl=mapdl,
        out_root=out_root,
        cfg=cfg,
        eval_idx=state["eval_count"],
        leg_size=float(leg_size),
        tabletop_thk=float(tabletop_thk)
    )

    state["cache"][key] = rec
    state["history"].append(rec)
    state["eval_count"] += 1

    return rec


def bisection_min_thk_for_leg(mapdl, out_root, cfg, leg_size, state):
    thk_min = float(cfg["design_var"]["tabletop_thk_mm"]["min"])
    thk_max = float(cfg["design_var"]["tabletop_thk_mm"]["max"])
    tol = float(cfg["optimizer"]["thk_bisect_tol_mm"])

    log(f"\n[SEARCH] leg_size = {leg_size:.6f} mm 에 대해 최소 tabletop_thk 탐색")

    # upper bound feasibility check
    rec_hi = get_cached_or_eval(mapdl, out_root, cfg, leg_size, thk_max, state)
    if not rec_hi["feasible"]:
        log(f"[SEARCH] thk_max={thk_max:.6f} mm에서도 불가능")
        return None

    # lower bound check
    rec_lo = get_cached_or_eval(mapdl, out_root, cfg, leg_size, thk_min, state)
    if rec_lo["feasible"]:
        log(f"[SEARCH] thk_min={thk_min:.6f} mm already feasible")
        return rec_lo

    lo = thk_min
    hi = thk_max
    best = rec_hi

    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        rec_mid = get_cached_or_eval(mapdl, out_root, cfg, leg_size, mid, state)

        if rec_mid["feasible"]:
            best = rec_mid
            hi = mid
        else:
            lo = mid

        log(f"[BISECTION] leg={leg_size:.6f}, lo={lo:.6f}, hi={hi:.6f}")

    return best


def choose_better_design(a, b):
    if a is None:
        return b
    if b is None:
        return a

    if not a["feasible"] and b["feasible"]:
        return b
    if a["feasible"] and not b["feasible"]:
        return a
    if not a["feasible"] and not b["feasible"]:
        return a

    if b["volume_mm3"] < a["volume_mm3"]:
        return b
    if abs(b["volume_mm3"] - a["volume_mm3"]) <= 1e-9 and b["Smax_MPa"] < a["Smax_MPa"]:
        return b
    return a


# ============================================================
# main
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = read_json(os.path.join(base_dir, "problem.json"))

    out_root = os.path.join(base_dir, "outputs_DESK_FINAL_NOCLI")
    ensure_dir(out_root)

    mapdl_cfg = cfg["mapdl"]
    run_location = mapdl_cfg["run_location"]
    jobname_prefix = mapdl_cfg["jobname_prefix"]
    override = bool(mapdl_cfg["override"])

    ensure_dir(run_location)
    cleanup_old_lock(run_location, jobname_prefix)

    max_eval = int(cfg["optimizer"]["max_eval"])
    coarse_leg_step = float(cfg["optimizer"]["coarse_leg_step_mm"])
    fine_leg_step = float(cfg["optimizer"]["fine_leg_step_mm"])

    leg_min = float(cfg["design_var"]["leg_size_mm"]["min"])
    leg_max = float(cfg["design_var"]["leg_size_mm"]["max"])

    state = {
        "cache": {},
        "history": [],
        "eval_count": 0,
        "max_eval": max_eval
    }

    log("[MAPDL] single launch start")
    mapdl = launch_mapdl(
        run_location=run_location,
        jobname=jobname_prefix,
        override=override
    )
    log(f"[MAPDL] launched: workdir = {mapdl.directory}")

    best = None

    try:
        # ---------------- coarse search ----------------
        log("\n############################################################")
        log("[PHASE 1] coarse search")
        log("############################################################")

        coarse_legs = frange(leg_min, leg_max, coarse_leg_step)

        for leg_size in coarse_legs:
            cand = bisection_min_thk_for_leg(mapdl, out_root, cfg, leg_size, state)
            best = choose_better_design(best, cand)

        if best is None:
            raise RuntimeError("No feasible design found in coarse search")

        log("\n[PHASE 1 RESULT]")
        log(
            f"best coarse: leg={best['leg_size_mm']:.6f} mm, "
            f"thk={best['tabletop_thk_mm']:.6f} mm, "
            f"Smax={best['Smax_MPa']:.6f} MPa, "
            f"volume={best['volume_mm3']:.6f} mm^3"
        )

        # ---------------- fine search ----------------
        log("\n############################################################")
        log("[PHASE 2] fine search")
        log("############################################################")

        fine_center = float(best["leg_size_mm"])
        fine_leg_lo = max(leg_min, fine_center - coarse_leg_step)
        fine_leg_hi = min(leg_max, fine_center + coarse_leg_step)

        fine_legs = frange(fine_leg_lo, fine_leg_hi, fine_leg_step)

        for leg_size in fine_legs:
            cand = bisection_min_thk_for_leg(mapdl, out_root, cfg, leg_size, state)
            best = choose_better_design(best, cand)

    finally:
        try:
            mapdl.finish()
        except Exception:
            pass

        try:
            mapdl.exit()
        except Exception:
            pass

    final_dir = os.path.join(out_root, "final")
    ensure_dir(final_dir)

    if best is not None:
        best_eval_dir = os.path.join(out_root, f"eval_{best['eval']:03d}")
        for fname in ["model.step", "mesh.msh", "model.cdb", "stress.png", "result.json"]:
            safe_copy(os.path.join(best_eval_dir, fname), os.path.join(final_dir, fname))

    summary = {
        "best": best,
        "n_evaluations": state["eval_count"],
        "history": state["history"]
    }
    write_json(os.path.join(final_dir, "summary.json"), summary)

    log("\n==============================")
    log("[DONE]")
    if best is not None:
        log(f"Best leg_size      = {best['leg_size_mm']:.6f} mm")
        log(f"Best tabletop_thk  = {best['tabletop_thk_mm']:.6f} mm")
        log(f"Best Smax          = {best['Smax_MPa']:.6e} MPa")
        log(f"Best volume        = {best['volume_mm3']:.6e} mm^3")
    log(f"Evaluations        = {state['eval_count']}")
    log(f"Outputs            = {final_dir}")
    log("==============================")


if __name__ == "__main__":
    main()