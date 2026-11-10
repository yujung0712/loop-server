import os
import json
import time
import math
import shutil
import warnings
import numpy as np
import gmsh
from ansys.mapdl.core import launch_mapdl


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*extract_surface.*", category=Warning)


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


def cleanup_old_lock(run_location: str, jobname_prefix: str):
    lock_path = os.path.join(run_location, f"{jobname_prefix}.lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            log(f"[LOCK] removed old lock: {lock_path}")
        except Exception:
            log(f"[LOCK] could not remove lock: {lock_path}")


def to_apdl_path(path: str):
    return os.path.abspath(path).replace("\\", "/")


def set_mapdl_cwd(mapdl, path: str):
    mapdl.run(f"/CWD,'{to_apdl_path(path)}'")


def compress_ranges(sorted_ids: np.ndarray):
    sorted_ids = np.asarray(sorted_ids, dtype=int).reshape(-1)
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
    """
    MAPDL에서 좌표 LOC로 면을 찾지 않는다.
    Gmsh physical group에서 추출한 node id만 사용해서 component를 만든다.
    단, CM을 만들기 위해 NODE number selection은 사용한다.
    """
    node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    if node_ids.size == 0:
        raise RuntimeError(f"{comp_name} node list is empty")

    node_ids = np.unique(node_ids)
    node_ids.sort()

    mapdl.allsel()
    mapdl.run("*GET,NNODE_NOW,NODE,0,COUNT")
    nnode_now = int(mapdl.parameters["NNODE_NOW"])
    if nnode_now <= 0:
        raise RuntimeError(f"{comp_name}: MAPDL DB has zero node")

    mapdl_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    common_ids = np.intersect1d(node_ids, mapdl_nodes)
    if common_ids.size == 0:
        raise RuntimeError(f"{comp_name}: no matching node ids in MAPDL DB")

    ranges = compress_ranges(common_ids)

    mapdl.allsel()
    s0, e0 = ranges[0]
    mapdl.nsel("S", "NODE", "", int(s0), int(e0))
    for s, e in ranges[1:]:
        mapdl.nsel("A", "NODE", "", int(s), int(e))

    mapdl.run(f"*GET,NCOUNT_{comp_name},NODE,0,COUNT")
    ncount = int(mapdl.parameters[f"NCOUNT_{comp_name}"])
    if ncount <= 0:
        raise RuntimeError(f"{comp_name}: selected node count is zero")

    mapdl.cm(comp_name, "NODE")
    mapdl.allsel()

    return {
        "requested": int(node_ids.size),
        "matched": int(common_ids.size),
        "selected": int(ncount),
    }


# ============================================================
# gmsh build + extract
# ============================================================
def get_phys_surface_nodes(phys_tag: int):
    surf_entities = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
    node_set = set()

    for ent in surf_entities:
        types, _, elem_node_tags = gmsh.model.mesh.getElements(2, ent)
        for enodes in elem_node_tags:
            enodes = np.asarray(enodes, dtype=np.int64)
            node_set.update(enodes.tolist())

    return np.array(sorted(node_set), dtype=np.int64)


def gmsh_build_extract_beam(eval_dir: str, cfg: dict):
    geo = cfg["geometry_mm"]
    mesh_cfg = cfg["mesh_mm"]
    tags = cfg["physical_tags"]
    outputs = cfg["outputs"]

    L = float(geo["beam_length"])
    B = float(geo["beam_width"])
    H = float(geo["beam_height"])

    lc_global = float(mesh_cfg["lc_global"])
    element_order = int(mesh_cfg.get("element_order", 2))

    FIXED = int(tags["FIXED"])
    LOAD = int(tags["LOAD"])
    VOL = int(tags["VOL"])

    step_path = os.path.join(eval_dir, "beam.step")
    msh_path = os.path.join(eval_dir, "beam.msh")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CANTILEVER_TET10_BEAM")
    occ = gmsh.model.occ

    # mm unit geometry
    # length: x direction, width: y direction, height: z direction
    box = occ.addBox(0.0, -0.5 * B, -0.5 * H, L, B, H)
    occ.synchronize()

    # Physical surface 지정은 Gmsh에서만 수행한다.
    # MAPDL에서는 좌표 기반 nsel,loc를 사용하지 않는다.
    surfaces = gmsh.model.getBoundary([(3, box)], oriented=False, recursive=False)
    fixed_surfs = []
    load_surfs = []

    # OCC box surface bbox can include tiny tolerance expansion.
    # Therefore classify end faces by center of mass, not by xmin==xmax exactly.
    # This is only for assigning Gmsh Physical Groups.
    # MAPDL still never uses coordinate-based NSEL/LOC.
    tol = max(1.0e-6, 1.0e-9 * L)
    for dim, stag in surfaces:
        if dim != 2:
            continue

        xcm, ycm, zcm = occ.getCenterOfMass(dim, stag)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, stag)
        xspan = xmax - xmin

        # End faces are planes normal to x direction, so their x-span is almost zero.
        if xspan < 10.0 * tol and abs(xcm - 0.0) < tol:
            fixed_surfs.append(stag)
        elif xspan < 10.0 * tol and abs(xcm - L) < tol:
            load_surfs.append(stag)

    if len(fixed_surfs) == 0:
        gmsh.finalize()
        raise RuntimeError("FIXED physical surface not found")
    if len(load_surfs) == 0:
        gmsh.finalize()
        raise RuntimeError("LOAD physical surface not found")

    gmsh.model.addPhysicalGroup(2, fixed_surfs, FIXED)
    gmsh.model.setPhysicalName(2, FIXED, "FIXED")

    gmsh.model.addPhysicalGroup(2, load_surfs, LOAD)
    gmsh.model.setPhysicalName(2, LOAD, "LOAD")

    gmsh.model.addPhysicalGroup(3, [box], VOL)
    gmsh.model.setPhysicalName(3, VOL, "BODY")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_global)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_global)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)

    t0 = time.time()
    gmsh.model.mesh.generate(3)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)
        try:
            gmsh.model.mesh.optimize("HighOrder")
        except Exception:
            pass
    mesh_time = time.time() - t0

    if outputs.get("save_step", True):
        gmsh.write(step_path)
    else:
        step_path = ""

    if outputs.get("save_msh", True):
        gmsh.write(msh_path)
    else:
        msh_path = ""

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_xyz = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)

    vol_entities = gmsh.model.getEntitiesForPhysicalGroup(3, VOL)
    tet_conn = []

    for ent in vol_entities:
        types, _, elem_node_tags = gmsh.model.mesh.getElements(3, ent)
        for etype, enodes in zip(types, elem_node_tags):
            props = gmsh.model.mesh.getElementProperties(etype)
            nnodes = int(props[3])
            enodes = np.asarray(enodes, dtype=np.int64)

            if element_order == 2 and nnodes == 10:
                tet_conn.append(enodes.reshape(-1, 10))
            elif element_order == 1 and nnodes == 4:
                tet_conn.append(enodes.reshape(-1, 4))

    if len(tet_conn) == 0:
        gmsh.finalize()
        raise RuntimeError("tet connectivity not found")

    tet_conn = np.vstack(tet_conn)

    fixed_nodes = get_phys_surface_nodes(FIXED)
    load_nodes = get_phys_surface_nodes(LOAD)

    gmsh.finalize()

    return {
        "step_path": step_path,
        "msh_path": msh_path,
        "node_tags": node_tags,
        "node_xyz": node_xyz,
        "tet_conn": tet_conn,
        "fixed_nodes": fixed_nodes,
        "load_nodes": load_nodes,
        "mesh_time_s": float(mesh_time),
        "n_nodes": int(len(node_tags)),
        "n_elems": int(len(tet_conn)),
        "nnpe": int(tet_conn.shape[1]),
        "element_order": int(element_order),
    }


# ============================================================
# MAPDL build + solve
# ============================================================
def write_mapdl_build_inp(
    work_dir: str,
    node_tags: np.ndarray,
    node_xyz: np.ndarray,
    tet_conn: np.ndarray,
):
    inp_name = "build_cantilever_beam.inp"
    inp_path = os.path.join(work_dir, inp_name)

    nnpe = int(tet_conn.shape[1])
    if nnpe not in [4, 10]:
        raise RuntimeError(f"Unsupported tetra nodes-per-element: {nnpe}")

    with open(inp_path, "w", encoding="ascii", errors="ignore") as f:
        f.write("/PREP7\n")
        f.write("TYPE,1\n")
        f.write("MAT,1\n")

        for nid, xyz in zip(node_tags, node_xyz):
            f.write(
                f"N,{int(nid)},"
                f"{float(xyz[0]):.16e},"
                f"{float(xyz[1]):.16e},"
                f"{float(xyz[2]):.16e}\n"
            )

        f.write("TYPE,1\n")
        f.write("MAT,1\n")

        if nnpe == 4:
            for conn in tet_conn:
                n1, n2, n3, n4 = map(int, conn.tolist())
                f.write(f"E,{n1},{n2},{n3},{n4}\n")

        elif nnpe == 10:
            for conn in tet_conn:
                conn = np.asarray(conn, dtype=int)

                # Gmsh Tet10 -> ANSYS SOLID187
                # 네가 말한 차이: 마지막 두 midside node 순서 swap
                conn = conn[[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]

                f.write(
                    f"E,{conn[0]},{conn[1]},{conn[2]},{conn[3]},"
                    f"{conn[4]},{conn[5]},{conn[6]},{conn[7]}\n"
                )
                f.write(f"EMORE,{conn[8]},{conn[9]}\n")

        f.write("FINISH\n")

    return inp_name, inp_path


def solve_beam(mapdl, eval_dir: str, cfg: dict, gm: dict):
    mat = cfg["material_MPa"]
    geo = cfg["geometry_mm"]
    load = cfg["load"]
    outputs = cfg["outputs"]

    E = float(mat["E"])          # MPa = N/mm^2
    nu = float(mat["nu"])

    L = float(geo["beam_length"])
    B = float(geo["beam_width"])
    H = float(geo["beam_height"])

    F_total = float(load["F_tip_total_N"])
    force_dof = str(load.get("force_dof", "FZ")).upper()

    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    nnpe = int(gm["nnpe"])

    workdir = mapdl.directory
    build_inp_name, build_inp_path = write_mapdl_build_inp(
        work_dir=workdir,
        node_tags=node_tags,
        node_xyz=node_xyz,
        tet_conn=tet_conn,
    )

    mapdl.clear()

    mapdl.prep7()
    if nnpe == 4:
        mapdl.et(1, 285)
    elif nnpe == 10:
        mapdl.et(1, 187)
    else:
        raise RuntimeError(f"Unsupported nnpe={nnpe}")

    mapdl.type(1)
    mapdl.mat(1)
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, nu)
    mapdl.finish()

    set_mapdl_cwd(mapdl, workdir)

    t0 = time.time()
    mapdl.input(build_inp_name)
    input_time = time.time() - t0

    mapdl.prep7()
    mapdl.allsel()
    mapdl.run("*GET,NNODE_BUILD,NODE,0,COUNT")
    mapdl.run("*GET,NELEM_BUILD,ELEM,0,COUNT")
    nnode_build = int(mapdl.parameters["NNODE_BUILD"])
    nelem_build = int(mapdl.parameters["NELEM_BUILD"])

    fixed_info = make_component_from_nodes(mapdl, "FIXED", gm["fixed_nodes"])
    load_info = make_component_from_nodes(mapdl, "LOAD", gm["load_nodes"])
    mapdl.finish()

    mapdl.slashsolu()
    mapdl.antype("STATIC")

    mapdl.cmsel("S", "FIXED")
    mapdl.run("*GET,NFIX,NODE,0,COUNT")
    nfix = int(mapdl.parameters["NFIX"])
    if nfix <= 0:
        raise RuntimeError("FIXED component is empty")
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    mapdl.cmsel("S", "LOAD")
    mapdl.run("*GET,NLOAD,NODE,0,COUNT")
    nload = int(mapdl.parameters["NLOAD"])
    if nload <= 0:
        raise RuntimeError("LOAD component is empty")

    f_each = F_total / float(nload)
    mapdl.f("ALL", force_dof, f_each)
    mapdl.allsel()

    t0 = time.time()
    mapdl.solve()
    solve_time = time.time() - t0
    mapdl.finish()

    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()

    # LOAD component 변위 평균/최소/최대
    mapdl.cmsel("S", "LOAD")
    load_node_ids = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    uz_vals = []
    for nid in load_node_ids:
        uz_vals.append(float(mapdl.get_value("NODE", int(nid), "U", "Z")))
    uz_vals = np.asarray(uz_vals, dtype=float)
    mapdl.allsel()

    # FIXED component reaction sum
    mapdl.cmsel("S", "FIXED")
    fixed_node_ids = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    rfz_sum = 0.0
    for nid in fixed_node_ids:
        rfz_sum += float(mapdl.get_value("NODE", int(nid), "RF", "FZ"))
    mapdl.allsel()

    seqv = np.asarray(mapdl.post_processing.nodal_eqv_stress(), dtype=float).reshape(-1)
    seqv = seqv[np.isfinite(seqv)]
    global_max_seqv_MPa = float(np.max(seqv)) if seqv.size > 0 else None

    png_path = os.path.join(eval_dir, "seqv.png")
    png_ok = False
    if outputs.get("save_png", True):
        try:
            mapdl.post_processing.plot_nodal_eqv_stress(
                show_edges=True,
                off_screen=True,
                savefig=png_path,
            )
            png_ok = os.path.exists(png_path)
        except Exception:
            png_ok = False

    mapdl.finish()

    # Euler-Bernoulli theory, mm-N-MPa system
    I = B * H ** 3 / 12.0
    c = H / 2.0
    uz_eb = F_total * L ** 3 / (3.0 * E * I)
    sigma_eb = abs(F_total) * L * c / I

    uz_avg = float(np.mean(uz_vals))
    uz_min = float(np.min(uz_vals))
    uz_max = float(np.max(uz_vals))

    if abs(uz_eb) > 0.0:
        uz_err_pct = abs((uz_avg - uz_eb) / uz_eb) * 100.0
    else:
        uz_err_pct = None

    if abs(F_total) > 0.0:
        # applied FZ=-1000이면 fixed reaction은 +1000에 가까워야 함
        reaction_err_pct = abs((rfz_sum + F_total) / F_total) * 100.0
    else:
        reaction_err_pct = None

    return {
        "case_name": cfg.get("case_name", ""),
        "units": {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "E": "MPa"
        },
        "geometry_mm": {
            "beam_length": L,
            "beam_width": B,
            "beam_height": H
        },
        "material_MPa": {
            "E": E,
            "nu": nu
        },
        "load": {
            "F_tip_total_N": F_total,
            "force_dof": force_dof,
            "load_nodes": int(nload),
            "force_per_load_node_N": float(f_each)
        },
        "mesh": {
            "mesh_nodes": int(gm["n_nodes"]),
            "mesh_elems": int(gm["n_elems"]),
            "build_nodes": int(nnode_build),
            "build_elems": int(nelem_build),
            "element_order": int(gm["element_order"]),
            "nnpe": int(gm["nnpe"]),
            "mesh_time_s": float(gm["mesh_time_s"]),
            "input_time_s": float(input_time),
            "solve_time_s": float(solve_time)
        },
        "component_info": {
            "FIXED": fixed_info,
            "LOAD": load_info
        },
        "result_mapdl": {
            "load_face_UZ_min_mm": uz_min,
            "load_face_UZ_avg_mm": uz_avg,
            "load_face_UZ_max_mm": uz_max,
            "fixed_reaction_FZ_sum_N": float(rfz_sum),
            "global_max_seqv_MPa": global_max_seqv_MPa
        },
        "result_theory_EB": {
            "I_mm4": float(I),
            "tip_UZ_mm": float(uz_eb),
            "max_bending_stress_MPa": float(sigma_eb)
        },
        "error": {
            "tip_UZ_avg_error_percent_vs_EB": None if uz_err_pct is None else float(uz_err_pct),
            "reaction_error_percent": None if reaction_err_pct is None else float(reaction_err_pct)
        },
        "files": {
            "step": gm["step_path"],
            "msh": gm["msh_path"],
            "png": png_path if png_ok else "",
            "build_inp": build_inp_path
        }
    }


# ============================================================
# main
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = read_json(os.path.join(base_dir, "problem_beam.json"))

    out_root = os.path.join(base_dir, "outputs_CANTILEVER_TET10_BEAM")
    ensure_dir(out_root)

    mapdl_cfg = cfg["mapdl"]
    run_location = mapdl_cfg["run_location"]
    jobname_prefix = mapdl_cfg["jobname_prefix"]
    override = bool(mapdl_cfg["override"])

    ensure_dir(run_location)
    cleanup_old_lock(run_location, jobname_prefix)

    log("[GMSH] build start")
    gm = gmsh_build_extract_beam(out_root, cfg)
    log(f"[GMSH] nodes={gm['n_nodes']}, elems={gm['n_elems']}, nnpe={gm['nnpe']}")
    log(f"[GMSH] FIXED nodes={len(gm['fixed_nodes'])}, LOAD nodes={len(gm['load_nodes'])}")

    log("[MAPDL] launch start")
    mapdl = launch_mapdl(
        run_location=run_location,
        jobname=jobname_prefix,
        override=override,
        cleanup_on_exit=False,
    )
    log(f"[MAPDL] launched: workdir={mapdl.directory}")

    result = None
    try:
        result = solve_beam(mapdl, out_root, cfg, gm)
    finally:
        try:
            mapdl.finish()
        except Exception:
            pass
        try:
            if not mapdl.exited:
                mapdl.exit()
        except Exception:
            pass

    result_path = os.path.join(out_root, "result_beam.json")
    write_json(result_path, result)

    final_dir = os.path.join(out_root, "final")
    ensure_dir(final_dir)
    for fname in ["beam.step", "beam.msh", "seqv.png", "result_beam.json"]:
        safe_copy(os.path.join(out_root, fname), os.path.join(final_dir, fname))

    log("\n==============================")
    log("[DONE]")
    log(f"Length x Section     = {result['geometry_mm']['beam_length']:.6f} x "
        f"{result['geometry_mm']['beam_width']:.6f} x {result['geometry_mm']['beam_height']:.6f} mm")
    log(f"Element              = order {result['mesh']['element_order']}, nnpe={result['mesh']['nnpe']}")
    log(f"Nodes / Elems        = {result['mesh']['build_nodes']} / {result['mesh']['build_elems']}")
    log(f"FIXED / LOAD nodes   = {result['component_info']['FIXED']['selected']} / "
        f"{result['component_info']['LOAD']['selected']}")
    log(f"Total FZ             = {result['load']['F_tip_total_N']:.6f} N")
    log(f"Force per LOAD node  = {result['load']['force_per_load_node_N']:.12e} N")
    log(f"Reaction FZ sum      = {result['result_mapdl']['fixed_reaction_FZ_sum_N']:.12e} N")
    log(f"Reaction error       = {result['error']['reaction_error_percent']:.6f} %")
    log("")
    log(f"MAPDL LOAD UZ avg    = {result['result_mapdl']['load_face_UZ_avg_mm']:.12e} mm")
    log(f"EB tip UZ            = {result['result_theory_EB']['tip_UZ_mm']:.12e} mm")
    log(f"UZ error vs EB       = {result['error']['tip_UZ_avg_error_percent_vs_EB']:.6f} %")
    log("")
    log(f"MAPDL max SEQV       = {result['result_mapdl']['global_max_seqv_MPa']:.6f} MPa")
    log(f"EB max bend stress   = {result['result_theory_EB']['max_bending_stress_MPa']:.6f} MPa")
    log(f"Outputs              = {out_root}")
    log("==============================")


if __name__ == "__main__":
    main()
