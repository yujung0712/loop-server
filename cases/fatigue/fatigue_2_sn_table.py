import os
import json
import time
import math
import shutil
import warnings
import numpy as np
import gmsh
from ansys.mapdl.core import launch_mapdl


# ============================================================
# warning / console cleanup
# ============================================================
warnings.filterwarnings(
    "ignore",
    message="The default value of `algorithm` for the filter",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*extract_surface.*",
    category=Warning
)


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
    return path.replace("\\", "/")


def set_mapdl_cwd(mapdl, path: str):
    apdl_path = to_apdl_path(os.path.abspath(path))
    mapdl.run(f"/CWD,'{apdl_path}'")


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


def write_mapdl_build_inp(
    work_dir: str,
    eval_idx: int,
    node_tags: np.ndarray,
    node_xyz: np.ndarray,
    tet_conn: np.ndarray
):
    inp_name = f"build_eval_{eval_idx:03d}.inp"
    inp_path = os.path.join(work_dir, inp_name)

    if tet_conn.ndim != 2:
        raise RuntimeError("tet_conn must be 2D array")

    nnpe = tet_conn.shape[1]
    if nnpe not in [4, 10]:
        raise RuntimeError(f"Unsupported tetra nodes-per-element: {nnpe}")

    with open(inp_path, "w", encoding="ascii", errors="ignore") as f:
        f.write("/PREP7\n")
        f.write("TYPE,1\n")
        f.write("MAT,1\n")

        for nid, xyz in zip(node_tags, node_xyz):
            f.write(f"N,{int(nid)},{float(xyz[0])},{float(xyz[1])},{float(xyz[2])}\n")

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
                # swap last two midside nodes
                conn = conn[[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
                conn = conn.tolist()

                f.write(
                    f"E,{conn[0]},{conn[1]},{conn[2]},{conn[3]},"
                    f"{conn[4]},{conn[5]},{conn[6]},{conn[7]}\n"
                )
                f.write(f"EMORE,{conn[8]},{conn[9]}\n")

        f.write("FINISH\n")

    return inp_name, inp_path


def make_component_from_nodes(mapdl, comp_name: str, node_ids: np.ndarray):
    node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    if node_ids.size == 0:
        raise RuntimeError(f"{comp_name} node list is empty")

    node_ids = np.unique(node_ids)
    node_ids.sort()

    mapdl.allsel()
    mapdl.run("*GET,NNODE_NOW,NODE,0,COUNT")
    nnode_now = int(mapdl.parameters["NNODE_NOW"])
    if nnode_now <= 0:
        raise RuntimeError(f"{comp_name} component creation failed: DB node count is zero")

    mapdl_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    if mapdl_nodes.size == 0:
        raise RuntimeError(f"{comp_name} component creation failed: mapdl.mesh.nnum is empty")

    common_ids = np.intersect1d(node_ids, mapdl_nodes)
    if common_ids.size == 0:
        raise RuntimeError(f"{comp_name} component creation failed: no matching node ids")

    ranges = compress_ranges(common_ids)

    mapdl.allsel()
    s0, e0 = ranges[0]
    mapdl.nsel("S", "NODE", "", int(s0), int(e0))
    for s, e in ranges[1:]:
        mapdl.nsel("A", "NODE", "", int(s), int(e))

    mapdl.run(f"*GET,NCOUNT_{comp_name},NODE,0,COUNT")
    ncount = int(mapdl.parameters[f"NCOUNT_{comp_name}"])
    if ncount <= 0:
        raise RuntimeError(f"{comp_name} component creation failed: selected node count is zero")

    mapdl.cm(comp_name, "NODE")
    mapdl.allsel()

    return {
        "requested": int(len(node_ids)),
        "matched": int(len(common_ids)),
        "selected": int(ncount),
    }


# ============================================================
# design helpers
# ============================================================
def get_initial_design(cfg: dict):
    geo = cfg["geometry_mm"]
    return {
        "plate_length": float(geo["plate_length"]),
        "plate_width": float(geo["plate_width"]),
        "plate_thk": float(geo["plate_thk"]),
        "hole_d": float(geo["hole_d"])
    }


def apply_dependent_rules(cfg: dict, design: dict):
    d = dict(design)

    L = float(d["plate_length"])
    W = float(d["plate_width"])
    T = float(d["plate_thk"])
    D = float(d["hole_d"])

    if L <= 0.0 or W <= 0.0 or T <= 0.0 or D <= 0.0:
        return None

    if D >= 0.8 * W:
        return None

    return d


def design_key(design: dict):
    return (
        round(float(design["plate_width"]), 6),
        round(float(design["plate_thk"]), 6),
        round(float(design["hole_d"]), 6)
    )


def design_change_l1(design: dict, base_design: dict):
    keys = ["plate_width", "plate_thk", "hole_d"]
    return float(sum(abs(float(design[k]) - float(base_design[k])) for k in keys))


def candidate_index_map(cands):
    return {round(float(v), 6): i for i, v in enumerate(cands)}


def generate_neighbors(cfg: dict, design: dict):
    ds = cfg["design_space_mm"]

    var_to_cands = {
        "plate_width": [float(x) for x in ds["plate_width_candidates"]],
        "plate_thk": [float(x) for x in ds["plate_thk_candidates"]],
        "hole_d": [float(x) for x in ds["hole_d_candidates"]]
    }

    neigh = []
    for var_name, cands in var_to_cands.items():
        idx_map = candidate_index_map(cands)
        cur = round(float(design[var_name]), 6)
        if cur not in idx_map:
            raise RuntimeError(f"Current value of {var_name}={cur} is not in candidate list")

        i = idx_map[cur]
        for j in [i - 1, i + 1]:
            if 0 <= j < len(cands):
                dnew = dict(design)
                dnew[var_name] = float(cands[j])
                dnew = apply_dependent_rules(cfg, dnew)
                if dnew is not None:
                    neigh.append(dnew)

    uniq = {}
    for d in neigh:
        uniq[design_key(d)] = d

    return list(uniq.values())


def plate_with_hole_volume_mm3(design: dict):
    L = float(design["plate_length"])
    W = float(design["plate_width"])
    T = float(design["plate_thk"])
    D = float(design["hole_d"])
    return (L * W - math.pi * (0.5 * D) ** 2) * T


# ============================================================
# gmsh build + extract
# ============================================================
def gmsh_build_extract_plate_hole(eval_dir: str, cfg: dict, design: dict):
    mesh_cfg = cfg["mesh_mm"]
    tags = cfg["physical_tags"]
    outputs = cfg["outputs"]

    L = float(design["plate_length"])
    W = float(design["plate_width"])
    T = float(design["plate_thk"])
    D = float(design["hole_d"])
    R = 0.5 * D

    lc_global = float(mesh_cfg["lc_global"])
    lc_hole = float(mesh_cfg["lc_hole"])
    element_order = int(mesh_cfg.get("element_order", 2))

    FIXED = int(tags["FIXED"])
    LOAD = int(tags["LOAD"])
    HOTSPOT = int(tags["HOTSPOT"])
    VOL = int(tags["VOL"])

    step_path = os.path.join(eval_dir, "model.step")
    msh_path = os.path.join(eval_dir, "mesh.msh")

    tol_x = 1.0e-8
    tol_bbox = 1.0e-6

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("PLATE_WITH_HOLE_FATIGUE")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0.0, -0.5 * W, 0.0, L, W)
    hole = occ.addDisk(0.5 * L, 0.0, 0.0, R, R)

    cut_out, _ = occ.cut([(2, rect)], [(2, hole)])
    occ.synchronize()

    if len(cut_out) != 1 or cut_out[0][0] != 2:
        gmsh.finalize()
        raise RuntimeError("2D plate cut failed")

    plate2d = cut_out[0][1]

    boundary_curves = gmsh.model.getBoundary([(2, plate2d)], oriented=False, recursive=False)
    curve_tags = [tag for dim, tag in boundary_curves if dim == 1]

    if len(curve_tags) == 0:
        gmsh.finalize()
        raise RuntimeError("No boundary curves found on 2D plate")

    fixed_curves = []
    load_curves = []
    hole_curves = []

    for ctag in curve_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, ctag)
        xspan = xmax - xmin
        yspan = ymax - ymin
        xcm, _, _ = occ.getCenterOfMass(1, ctag)

        if abs(xcm - 0.0) < tol_x and xspan < tol_bbox:
            fixed_curves.append(ctag)
            continue

        if abs(xcm - L) < tol_x and xspan < tol_bbox:
            load_curves.append(ctag)
            continue

        if abs(xspan - D) < 1.0e-4 * max(1.0, D) and abs(yspan - D) < 1.0e-4 * max(1.0, D):
            hole_curves.append(ctag)

    if len(fixed_curves) == 0:
        gmsh.finalize()
        raise RuntimeError("FIXED curve not found in 2D plate")
    if len(load_curves) == 0:
        gmsh.finalize()
        raise RuntimeError("LOAD curve not found in 2D plate")
    if len(hole_curves) == 0:
        gmsh.finalize()
        raise RuntimeError("HOLE curve not found in 2D plate")

    ext_out = occ.extrude([(2, plate2d)], 0.0, 0.0, T)
    occ.synchronize()

    vol_tags = [tag for dim, tag in ext_out if dim == 3]
    if len(vol_tags) != 1:
        gmsh.finalize()
        raise RuntimeError("Extrusion did not create exactly one volume")

    vol_tag = vol_tags[0]

    surf_tags_from_ext = [tag for dim, tag in ext_out if dim == 2]
    lateral_surface_tags = surf_tags_from_ext[1:]

    if len(lateral_surface_tags) != len(curve_tags):
        gmsh.finalize()
        raise RuntimeError(
            f"Extrusion mapping mismatch: lateral_surfaces={len(lateral_surface_tags)}, "
            f"boundary_curves={len(curve_tags)}"
        )

    curve_to_surface = {}
    for ctag, stag in zip(curve_tags, lateral_surface_tags):
        curve_to_surface[int(ctag)] = int(stag)

    fixed_surfs = [curve_to_surface[int(c)] for c in fixed_curves]
    load_surfs = [curve_to_surface[int(c)] for c in load_curves]
    hotspot_surfs = [curve_to_surface[int(c)] for c in hole_curves]

    gmsh.model.addPhysicalGroup(2, fixed_surfs, FIXED)
    gmsh.model.setPhysicalName(2, FIXED, "FIXED")

    gmsh.model.addPhysicalGroup(2, load_surfs, LOAD)
    gmsh.model.setPhysicalName(2, LOAD, "LOAD")

    gmsh.model.addPhysicalGroup(2, hotspot_surfs, HOTSPOT)
    gmsh.model.setPhysicalName(2, HOTSPOT, "HOTSPOT")

    gmsh.model.addPhysicalGroup(3, [vol_tag], VOL)
    gmsh.model.setPhysicalName(3, VOL, "BODY")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_hole, lc_global))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc_hole, lc_global))
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    field_dist = 1
    field_thr = 2

    gmsh.model.mesh.field.add("Distance", field_dist)
    gmsh.model.mesh.field.setNumbers(field_dist, "CurvesList", hole_curves)
    gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", field_thr)
    gmsh.model.mesh.field.setNumber(field_thr, "InField", field_dist)
    gmsh.model.mesh.field.setNumber(field_thr, "SizeMin", lc_hole)
    gmsh.model.mesh.field.setNumber(field_thr, "SizeMax", lc_global)
    gmsh.model.mesh.field.setNumber(field_thr, "DistMin", 0.5 * D)
    gmsh.model.mesh.field.setNumber(field_thr, "DistMax", 2.0 * D)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_thr)

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
            nnodes = props[3]
            enodes = np.asarray(enodes, dtype=np.int64)

            if element_order == 1 and nnodes == 4:
                tet_conn.append(enodes.reshape(-1, 4))

            if element_order == 2 and nnodes == 10:
                tet_conn.append(enodes.reshape(-1, 10))

    if len(tet_conn) == 0:
        gmsh.finalize()
        raise RuntimeError("tet connectivity not found for requested element order")

    tet_conn = np.vstack(tet_conn)

    def get_phys_surface_nodes(phys_tag: int):
        surf_entities = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
        node_set = set()

        for ent in surf_entities:
            types, _, elem_node_tags = gmsh.model.mesh.getElements(2, ent)
            for _, enodes in zip(types, elem_node_tags):
                enodes = np.asarray(enodes, dtype=np.int64)
                node_set.update(enodes.tolist())

        return np.array(sorted(node_set), dtype=np.int64)

    fixed_nodes = get_phys_surface_nodes(FIXED)
    load_nodes = get_phys_surface_nodes(LOAD)
    hotspot_nodes = get_phys_surface_nodes(HOTSPOT)

    gmsh.finalize()

    return {
        "step_path": step_path,
        "msh_path": msh_path,
        "node_tags": node_tags,
        "node_xyz": node_xyz,
        "tet_conn": tet_conn,
        "fixed_nodes": fixed_nodes,
        "load_nodes": load_nodes,
        "hotspot_nodes": hotspot_nodes,
        "mesh_time_s": float(mesh_time),
        "n_nodes": int(len(node_tags)),
        "n_elems": int(len(tet_conn)),
        "nnpe": int(tet_conn.shape[1]),
        "element_order": int(element_order),
    }


# ============================================================
# fatigue helpers
# ============================================================
def goodman_corrected_amplitude(sig_a: float, sig_m: float, Su: float):
    denom = 1.0 - sig_m / Su
    if denom <= 1.0e-12:
        return 1.0e30
    return sig_a / denom


def basquin_life_cycles(sig_a_eq: np.ndarray, sigma_f_prime: float, b: float):
    sig_a_eq = np.asarray(sig_a_eq, dtype=float)

    if sigma_f_prime <= 0.0:
        raise RuntimeError("basquin_sigma_f_prime_MPa must be > 0")
    if b >= 0.0:
        raise RuntimeError("basquin_b must be negative")

    out = np.empty_like(sig_a_eq, dtype=float)
    mask = sig_a_eq > 0.0

    out[~mask] = 1.0e30
    out[mask] = 0.5 * (sig_a_eq[mask] / sigma_f_prime) ** (1.0 / b)

    return out




def sn_curve_life_cycles(sig_a_eq: np.ndarray, stress_MPa, cycles):
    """
    S-N table based life calculation.

    Input stress must be alternating/equivalent stress amplitude [MPa].
    The table is interpolated in log10(S)-log10(N) space, matching the
    common S-N curve representation used by ANSYS Engineering Data.
    """
    sig = np.asarray(stress_MPa, dtype=float).reshape(-1)
    N = np.asarray(cycles, dtype=float).reshape(-1)
    x = np.asarray(sig_a_eq, dtype=float)

    if sig.size != N.size:
        raise RuntimeError("S-N curve stress_MPa and cycles must have the same length")
    if sig.size < 2:
        raise RuntimeError("S-N curve needs at least two points")
    if np.any(sig <= 0.0) or np.any(N <= 0.0):
        raise RuntimeError("S-N curve stress_MPa and cycles must be positive")

    # np.interp requires increasing x-axis.
    # ANSYS table is usually written as N increasing and stress decreasing,
    # so sorting by stress makes interpolation stable.
    order = np.argsort(sig)
    sig = sig[order]
    N = N[order]

    log_sig = np.log10(sig)
    log_N = np.log10(N)

    out = np.empty_like(x, dtype=float)
    mask = x > 0.0
    out[~mask] = 1.0e30

    # Outside the table range, clamp to end values.
    # This is safer than uncontrolled extrapolation for optimization.
    out[mask] = 10.0 ** np.interp(
        np.log10(x[mask]),
        log_sig,
        log_N,
        left=log_N[0],
        right=log_N[-1]
    )

    return out


def get_topk_mean(vals: np.ndarray, k: int = 3):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        raise RuntimeError("empty array in get_topk_mean")
    k = min(k, vals.size)
    part = np.partition(vals, vals.size - k)[-k:]
    return float(np.mean(part))


def get_component_node_stress(mapdl, comp_name: str):
    all_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    seqv = np.asarray(mapdl.post_processing.nodal_eqv_stress(), dtype=float).reshape(-1)

    if all_nodes.size == 0 or seqv.size == 0:
        raise RuntimeError("POST data is empty")
    if all_nodes.size != seqv.size:
        raise RuntimeError("node count != stress count")

    mapdl.cmsel("S", comp_name)
    selected_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    mapdl.allsel()

    if selected_nodes.size == 0:
        raise RuntimeError(f"{comp_name} component empty in POST")

    selected_set = set(selected_nodes.tolist())
    mask = np.array([nid in selected_set for nid in all_nodes], dtype=bool)

    vals = seqv[mask]
    nodes = all_nodes[mask]

    valid = np.isfinite(vals)
    vals = vals[valid]
    nodes = nodes[valid]

    if vals.size == 0:
        raise RuntimeError(f"{comp_name} stress has no finite value")

    idx = int(np.argmax(vals))

    return {
        "node_ids": nodes,
        "stress_vals_MPa": vals,
        "max_stress_MPa": float(np.max(vals)),
        "avg_stress_MPa": float(np.mean(vals)),
        "p95_stress_MPa": float(np.percentile(vals, 95.0)),
        "top3_stress_MPa": float(get_topk_mean(vals, 3)),
        "node_at_max": int(nodes[idx]),
        "n_nodes": int(len(nodes)),
    }


def stats_from_array(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    return {
        "min": float(np.min(arr)),
        "avg": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


# ============================================================
# MAPDL solve
# ============================================================
def solve_loadcase_displacement(
    mapdl,
    eval_idx: int,
    workdir: str,
    gm: dict,
    E: float,
    nu: float,
    ux_load_mm: float,
    png_path: str = ""
):
    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    nnpe = int(gm["nnpe"])

    build_inp_name, build_inp_path = write_mapdl_build_inp(
        work_dir=workdir,
        eval_idx=eval_idx,
        node_tags=node_tags,
        node_xyz=node_xyz,
        tet_conn=tet_conn
    )

    if not os.path.exists(build_inp_path):
        raise RuntimeError(f"Build inp not found: {build_inp_path}")

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
    mapdl.list_files = lambda *args, **kwargs: []
    input_out = mapdl.input(build_inp_path)
    input_time = time.time() - t0

    mapdl.prep7()
    mapdl.allsel()
    mapdl.run("*GET,NNODE_BUILD,NODE,0,COUNT")
    mapdl.run("*GET,NELEM_BUILD,ELEM,0,COUNT")
    nnode_build = int(mapdl.parameters["NNODE_BUILD"])
    nelem_build = int(mapdl.parameters["NELEM_BUILD"])
    log(
        f"    [MAPDL] after build: nodes={nnode_build}, elems={nelem_build} "
        f"(gmsh: nodes={gm['n_nodes']}, elems={gm['n_elems']})"
    )
    if nelem_build <= 0:
        raise RuntimeError(
            "MAPDL에 요소가 하나도 들어가지 않았어요. "
            f"gmsh 노드 {gm['n_nodes']}개 / 요소 {gm['n_elems']}개. "
            "학생용 라이선스의 크기 제한을 넘었을 수 있으니 메시 크기(lc_global, lc_hole)를 키워 보세요. "
            f"입력 결과 끝부분: {str(input_out)[-600:]}"
        )

    fixed_info = make_component_from_nodes(mapdl, "FIXED", gm["fixed_nodes"])
    load_info = make_component_from_nodes(mapdl, "LOAD", gm["load_nodes"])
    hotspot_info = make_component_from_nodes(mapdl, "HOTSPOT", gm["hotspot_nodes"])
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
    mapdl.d("ALL", "UX", float(ux_load_mm))
    mapdl.allsel()

    t0 = time.time()
    mapdl.solve()
    solve_time = time.time() - t0
    mapdl.finish()

    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()

    hotspot = get_component_node_stress(mapdl, "HOTSPOT")

    total_seqv = np.asarray(mapdl.post_processing.nodal_eqv_stress(), dtype=float).reshape(-1)
    valid = np.isfinite(total_seqv)
    if np.sum(valid) == 0:
        raise RuntimeError("No finite stress in whole model")

    global_max_eqv = float(np.max(total_seqv[valid]))

    png_ok = False
    if png_path:
        try:
            mapdl.post_processing.plot_nodal_eqv_stress(
                show_edges=True,
                off_screen=True,
                savefig=png_path
            )
            png_ok = os.path.exists(png_path)
        except Exception:
            png_ok = False

    mapdl.finish()

    return {
        "global_max_eqv_MPa": global_max_eqv,
        "hotspot_stats": hotspot,
        "png_ok": png_ok,
        "nnode_build": nnode_build,
        "nelem_build": nelem_build,
        "input_time_s": float(input_time),
        "solve_time_s": float(solve_time),
        "component_info": {
            "FIXED": fixed_info,
            "LOAD": load_info,
            "HOTSPOT": hotspot_info,
        },
    }


def solve_one_eval(mapdl, eval_idx: int, eval_dir: str, gm: dict, cfg: dict, design: dict):
    mat = cfg["material_MPa"]
    fatigue = cfg["fatigue"]
    cons = cfg["constraint"]

    E = float(mat["E"])
    nu = float(mat["nu"])
    rho = float(mat["density_kg_m3"])
    Su = float(mat["Su"])

    ux_max = float(fatigue["ux_max_mm"])
    ux_min = float(fatigue["ux_min_mm"])
    mean_stress_correction = str(fatigue["mean_stress_correction"]).strip().lower()
    life_method = str(fatigue.get("life_method", "basquin")).strip().lower()
    sigma_f_prime = float(fatigue.get("basquin_sigma_f_prime_MPa", 0.0))
    b = float(fatigue.get("basquin_b", -1.0))

    sn_curve = fatigue.get("sn_curve", {})
    sn_stress_MPa = sn_curve.get("stress_MPa", [])
    sn_cycles = sn_curve.get("cycles", [])

    min_life_cycles_limit = float(cons["min_life_cycles"])
    max_mass_kg = float(cons["max_mass_kg"])

    workdir = mapdl.directory
    png_path = os.path.join(eval_dir, "stress_max.png")

    res_max = solve_loadcase_displacement(
        mapdl=mapdl,
        eval_idx=2 * eval_idx,
        workdir=workdir,
        gm=gm,
        E=E,
        nu=nu,
        ux_load_mm=ux_max,
        png_path=png_path
    )

    res_min = solve_loadcase_displacement(
        mapdl=mapdl,
        eval_idx=2 * eval_idx + 1,
        workdir=workdir,
        gm=gm,
        E=E,
        nu=nu,
        ux_load_mm=ux_min,
        png_path=""
    )

    hs_max = res_max["hotspot_stats"]
    hs_min = res_min["hotspot_stats"]

    stress_max_arr = np.asarray(hs_max["stress_vals_MPa"], dtype=float)
    stress_min_arr = np.asarray(hs_min["stress_vals_MPa"], dtype=float)

    if stress_max_arr.size != stress_min_arr.size:
        raise RuntimeError("HOTSPOT stress array size mismatch between max/min loadcases")

    sigma_a_arr = 0.5 * np.abs(stress_max_arr - stress_min_arr)
    sigma_m_arr = 0.5 * (stress_max_arr + stress_min_arr)

    if mean_stress_correction == "goodman":
        sigma_a_eq_arr = np.array([
            goodman_corrected_amplitude(sa, sm, Su) for sa, sm in zip(sigma_a_arr, sigma_m_arr)
        ], dtype=float)
    else:
        sigma_a_eq_arr = sigma_a_arr.copy()

    if life_method == "basquin":
        life_arr = basquin_life_cycles(sigma_a_eq_arr, sigma_f_prime, b)
    elif life_method in ["sn_curve", "sn", "table"]:
        life_arr = sn_curve_life_cycles(sigma_a_eq_arr, sn_stress_MPa, sn_cycles)
    else:
        raise RuntimeError(f"Unknown fatigue life_method: {life_method}")

    sigma_a_stats = stats_from_array(sigma_a_arr)
    sigma_m_stats = stats_from_array(sigma_m_arr)
    sigma_a_eq_stats = stats_from_array(sigma_a_eq_arr)
    life_stats = stats_from_array(life_arr)

    # representative top-3 based fatigue
    sig_top3_max = float(hs_max["top3_stress_MPa"])
    sig_top3_min = float(hs_min["top3_stress_MPa"])
    sig_top3_a = 0.5 * abs(sig_top3_max - sig_top3_min)
    sig_top3_m = 0.5 * (sig_top3_max + sig_top3_min)

    if mean_stress_correction == "goodman":
        sig_top3_a_eq = goodman_corrected_amplitude(sig_top3_a, sig_top3_m, Su)
    else:
        sig_top3_a_eq = sig_top3_a

    if life_method == "basquin":
        life_top3 = float(basquin_life_cycles(np.array([sig_top3_a_eq]), sigma_f_prime, b)[0])
    elif life_method in ["sn_curve", "sn", "table"]:
        life_top3 = float(sn_curve_life_cycles(np.array([sig_top3_a_eq]), sn_stress_MPa, sn_cycles)[0])
    else:
        raise RuntimeError(f"Unknown fatigue life_method: {life_method}")

    vol_mm3 = plate_with_hole_volume_mm3(design)
    mass_kg = vol_mm3 * 1.0e-9 * rho

    life_ok = bool(life_top3 >= min_life_cycles_limit)
    mass_ok = bool(mass_kg <= max_mass_kg)
    feasible = bool(life_ok and mass_ok)

    return {
        "life_method": life_method,
        "global_max_eqv_MPa": float(max(res_max["global_max_eqv_MPa"], res_min["global_max_eqv_MPa"])),

        "hotspot_eqv_max_maxload_MPa": float(hs_max["max_stress_MPa"]),
        "hotspot_eqv_avg_maxload_MPa": float(hs_max["avg_stress_MPa"]),
        "hotspot_eqv_p95_maxload_MPa": float(hs_max["p95_stress_MPa"]),
        "hotspot_eqv_top3_maxload_MPa": float(hs_max["top3_stress_MPa"]),

        "hotspot_eqv_max_minload_MPa": float(hs_min["max_stress_MPa"]),
        "hotspot_eqv_avg_minload_MPa": float(hs_min["avg_stress_MPa"]),
        "hotspot_eqv_p95_minload_MPa": float(hs_min["p95_stress_MPa"]),
        "hotspot_eqv_top3_minload_MPa": float(hs_min["top3_stress_MPa"]),

        "sigma_a_min_MPa": sigma_a_stats["min"],
        "sigma_a_avg_MPa": sigma_a_stats["avg"],
        "sigma_a_p95_MPa": sigma_a_stats["p95"],
        "sigma_a_max_MPa": sigma_a_stats["max"],

        "sigma_m_min_MPa": sigma_m_stats["min"],
        "sigma_m_avg_MPa": sigma_m_stats["avg"],
        "sigma_m_p95_MPa": sigma_m_stats["p95"],
        "sigma_m_max_MPa": sigma_m_stats["max"],

        "sigma_a_eq_min_MPa": sigma_a_eq_stats["min"],
        "sigma_a_eq_avg_MPa": sigma_a_eq_stats["avg"],
        "sigma_a_eq_p95_MPa": sigma_a_eq_stats["p95"],
        "sigma_a_eq_max_MPa": sigma_a_eq_stats["max"],

        "hotspot_life_min_cycles": life_stats["min"],
        "hotspot_life_avg_cycles": life_stats["avg"],
        "hotspot_life_p95_cycles": life_stats["p95"],
        "hotspot_life_max_cycles": life_stats["max"],

        "top3_sigma_max_MPa": float(sig_top3_max),
        "top3_sigma_min_MPa": float(sig_top3_min),
        "top3_sigma_a_MPa": float(sig_top3_a),
        "top3_sigma_m_MPa": float(sig_top3_m),
        "top3_sigma_a_eq_MPa": float(sig_top3_a_eq),
        "top3_life_cycles": float(life_top3),

        "objective_life_cycles": float(life_top3),

        "hotspot_node_at_max": int(hs_max["node_at_max"]),
        "hotspot_nodes_n": int(hs_max["n_nodes"]),
        "mass_kg": float(mass_kg),
        "life_ok": life_ok,
        "mass_ok": mass_ok,
        "feasible": feasible,
        "png_path": png_path if res_max["png_ok"] else "",

        "build_nodes": int(res_max["nnode_build"]),
        "build_elems": int(res_max["nelem_build"]),
        "mesh_nodes": int(gm["n_nodes"]),
        "mesh_elems": int(gm["n_elems"]),
        "mesh_time_s": float(gm["mesh_time_s"]),
        "element_order": int(gm["element_order"]),
        "nnpe": int(gm["nnpe"]),
        "input_time_s_maxload": float(res_max["input_time_s"]),
        "input_time_s_minload": float(res_min["input_time_s"]),
        "solve_time_s_maxload": float(res_max["solve_time_s"]),
        "solve_time_s_minload": float(res_min["solve_time_s"]),
        "component_info": res_max["component_info"],
    }


# ============================================================
# evaluation
# ============================================================
def objective_score(rec: dict, cfg: dict, base_design: dict):
    cons = cfg["constraint"]
    opt = cfg["optimizer"]

    min_life_limit = float(cons["min_life_cycles"])
    max_mass = float(cons["max_mass_kg"])

    feasible_mass_coeff = float(opt["feasible_mass_coeff"])
    feasible_life_coeff = float(opt["feasible_life_coeff"])
    change_penalty_coeff = float(opt["change_penalty_coeff"])
    life_violation_coeff = float(opt["life_violation_coeff"])
    mass_violation_coeff = float(opt["mass_violation_coeff"])

    life = float(rec["objective_life_cycles"])
    mass = float(rec["mass_kg"])
    change = float(rec["design_change_l1"])

    life_violation = max(0.0, min_life_limit - life)
    mass_violation = max(0.0, mass - max_mass)

    if life_violation == 0.0 and mass_violation == 0.0:
        return feasible_mass_coeff * mass + feasible_life_coeff * life + change_penalty_coeff * change

    return (
        life_violation_coeff * (life_violation ** 2)
        + mass_violation_coeff * (mass_violation ** 2)
        + feasible_mass_coeff * mass
        + change_penalty_coeff * change
    )


def print_eval_summary(eval_idx: int, rec: dict):
    comp = rec["component_info"]

    log(f"[MESH {eval_idx:03d}] nodes={rec['build_nodes']}, elems={rec['build_elems']}, "
        f"order={rec['element_order']}, nnpe={rec['nnpe']}, "
        f"mesh={rec['mesh_time_s']:.2f}s, solve=({rec['solve_time_s_maxload']:.2f}s/{rec['solve_time_s_minload']:.2f}s)")

    log(f"           FIXED={comp['FIXED']['selected']}  LOAD={comp['LOAD']['selected']}  HOTSPOT={comp['HOTSPOT']['selected']} nodes")

    log(f"           HOTSPOT stress(maxload): "
        f"max={rec['hotspot_eqv_max_maxload_MPa']:.3f} MPa, "
        f"avg={rec['hotspot_eqv_avg_maxload_MPa']:.3f} MPa, "
        f"p95={rec['hotspot_eqv_p95_maxload_MPa']:.3f} MPa, "
        f"top3={rec['hotspot_eqv_top3_maxload_MPa']:.3f} MPa")

    log(f"           HOTSPOT life(dist): "
        f"min={rec['hotspot_life_min_cycles']:.3e}, "
        f"avg={rec['hotspot_life_avg_cycles']:.3e}, "
        f"p95={rec['hotspot_life_p95_cycles']:.3e}, "
        f"max={rec['hotspot_life_max_cycles']:.3e}")

    log(f"           TOP3 fatigue: "
        f"sigma_a_eq={rec['top3_sigma_a_eq_MPa']:.3f} MPa, "
        f"life={rec['top3_life_cycles']:.3e}")

    log(f"           life_method={rec['life_method']}")

    log(f"           objective_life(top3)={rec['objective_life_cycles']:.3e}, "
        f"mass={rec['mass_kg']:.6f} kg, feasible={rec['feasible']}")


def evaluate_design(mapdl, out_root: str, cfg: dict, eval_idx: int, design: dict, base_design: dict):
    eval_dir = os.path.join(out_root, f"eval_{eval_idx:03d}")
    ensure_dir(eval_dir)

    log("\n============================================================")
    log(
        f"[EVAL {eval_idx:03d}] "
        f"L={design['plate_length']:.3f} | "
        f"W={design['plate_width']:.3f} | "
        f"T={design['plate_thk']:.3f} | "
        f"D={design['hole_d']:.3f}"
    )
    log("============================================================")

    gm = gmsh_build_extract_plate_hole(eval_dir, cfg, design)
    res = solve_one_eval(mapdl, eval_idx, eval_dir, gm, cfg, design)

    rec = {
        "eval": eval_idx,
        "design": dict(design),
        "volume_mm3": plate_with_hole_volume_mm3(design),
        "mass_kg": res["mass_kg"],

        "mesh_nodes": res["mesh_nodes"],
        "mesh_elems": res["mesh_elems"],
        "build_nodes": res["build_nodes"],
        "build_elems": res["build_elems"],
        "mesh_time_s": res["mesh_time_s"],
        "element_order": res["element_order"],
        "nnpe": res["nnpe"],
        "input_time_s_maxload": res["input_time_s_maxload"],
        "input_time_s_minload": res["input_time_s_minload"],
        "solve_time_s_maxload": res["solve_time_s_maxload"],
        "solve_time_s_minload": res["solve_time_s_minload"],

        "life_method": res["life_method"],
        "global_max_eqv_MPa": res["global_max_eqv_MPa"],

        "hotspot_eqv_max_maxload_MPa": res["hotspot_eqv_max_maxload_MPa"],
        "hotspot_eqv_avg_maxload_MPa": res["hotspot_eqv_avg_maxload_MPa"],
        "hotspot_eqv_p95_maxload_MPa": res["hotspot_eqv_p95_maxload_MPa"],
        "hotspot_eqv_top3_maxload_MPa": res["hotspot_eqv_top3_maxload_MPa"],

        "hotspot_eqv_max_minload_MPa": res["hotspot_eqv_max_minload_MPa"],
        "hotspot_eqv_avg_minload_MPa": res["hotspot_eqv_avg_minload_MPa"],
        "hotspot_eqv_p95_minload_MPa": res["hotspot_eqv_p95_minload_MPa"],
        "hotspot_eqv_top3_minload_MPa": res["hotspot_eqv_top3_minload_MPa"],

        "sigma_a_min_MPa": res["sigma_a_min_MPa"],
        "sigma_a_avg_MPa": res["sigma_a_avg_MPa"],
        "sigma_a_p95_MPa": res["sigma_a_p95_MPa"],
        "sigma_a_max_MPa": res["sigma_a_max_MPa"],

        "sigma_m_min_MPa": res["sigma_m_min_MPa"],
        "sigma_m_avg_MPa": res["sigma_m_avg_MPa"],
        "sigma_m_p95_MPa": res["sigma_m_p95_MPa"],
        "sigma_m_max_MPa": res["sigma_m_max_MPa"],

        "sigma_a_eq_min_MPa": res["sigma_a_eq_min_MPa"],
        "sigma_a_eq_avg_MPa": res["sigma_a_eq_avg_MPa"],
        "sigma_a_eq_p95_MPa": res["sigma_a_eq_p95_MPa"],
        "sigma_a_eq_max_MPa": res["sigma_a_eq_max_MPa"],

        "hotspot_life_min_cycles": res["hotspot_life_min_cycles"],
        "hotspot_life_avg_cycles": res["hotspot_life_avg_cycles"],
        "hotspot_life_p95_cycles": res["hotspot_life_p95_cycles"],
        "hotspot_life_max_cycles": res["hotspot_life_max_cycles"],

        "top3_sigma_max_MPa": res["top3_sigma_max_MPa"],
        "top3_sigma_min_MPa": res["top3_sigma_min_MPa"],
        "top3_sigma_a_MPa": res["top3_sigma_a_MPa"],
        "top3_sigma_m_MPa": res["top3_sigma_m_MPa"],
        "top3_sigma_a_eq_MPa": res["top3_sigma_a_eq_MPa"],
        "top3_life_cycles": res["top3_life_cycles"],

        "objective_life_cycles": res["objective_life_cycles"],

        "life_ok": res["life_ok"],
        "mass_ok": res["mass_ok"],
        "feasible": res["feasible"],

        "hotspot_node_at_max": res["hotspot_node_at_max"],
        "hotspot_nodes_n": res["hotspot_nodes_n"],
        "component_info": res["component_info"],

        "design_change_l1": design_change_l1(design, base_design),
        "files": {
            "step": gm["step_path"],
            "msh": gm["msh_path"],
            "png": res["png_path"]
        }
    }

    rec["score"] = objective_score(rec, cfg, base_design)
    write_json(os.path.join(eval_dir, "result.json"), rec)

    print_eval_summary(eval_idx, rec)
    log(
        f"[RESULT] life={rec['top3_life_cycles']:.6e} cycles | "
        f"min_life={float(cfg['constraint']['min_life_cycles']):.6e} cycles | "
        f"mass={rec['mass_kg']:.6f} kg | "
        f"max_mass={float(cfg['constraint']['max_mass_kg']):.6f} kg | "
        f"feasible={rec['feasible']}"
    )

    return rec


# ============================================================
# optimization
# ============================================================
def get_cached_or_eval(mapdl, out_root, cfg, design, base_design, state):
    key = design_key(design)

    if key in state["cache"]:
        log(f"[CACHE] hit for {key}")
        return state["cache"][key]

    if state["eval_count"] >= state["max_eval"]:
        raise RuntimeError(f"max_eval reached: {state['max_eval']}")

    rec = evaluate_design(
        mapdl=mapdl,
        out_root=out_root,
        cfg=cfg,
        eval_idx=state["eval_count"],
        design=design,
        base_design=base_design
    )

    state["cache"][key] = rec
    state["history"].append(rec)
    state["eval_count"] += 1
    return rec


def optimize_case(mapdl, out_root, cfg, state):
    current_design = get_initial_design(cfg)
    current_design = apply_dependent_rules(cfg, current_design)
    if current_design is None:
        raise RuntimeError("Initial design is invalid")

    base_design = dict(current_design)

    current_rec = get_cached_or_eval(mapdl, out_root, cfg, current_design, base_design, state)
    best_rec = current_rec
    best_design = dict(current_design)

    max_iter = int(cfg["optimizer"]["max_iter"])

    for it in range(1, max_iter + 1):
        log("\n############################################################")
        log(f"[OPT ITER {it}]")
        log("############################################################")

        neighbors = generate_neighbors(cfg, current_design)
        if len(neighbors) == 0:
            log("[OPT] no neighbors available")
            break

        iter_best_rec = None
        iter_best_design = None

        for d in neighbors:
            rec = get_cached_or_eval(mapdl, out_root, cfg, d, base_design, state)
            if iter_best_rec is None or float(rec["score"]) < float(iter_best_rec["score"]):
                iter_best_rec = rec
                iter_best_design = dict(d)

        if iter_best_rec is None:
            break

        if float(iter_best_rec["score"]) + 1.0e-12 < float(current_rec["score"]):
            current_design = dict(iter_best_design)
            current_rec = iter_best_rec
            log(f"[OPT] accepted better neighbor, score={current_rec['score']:.6f}")
        else:
            log("[OPT] no improving neighbor found")
            break

        if best_rec is None or float(current_rec["score"]) < float(best_rec["score"]):
            best_rec = current_rec
            best_design = dict(current_design)

    return best_rec, best_design, base_design


# ============================================================
# main
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = read_json(os.path.join(base_dir, "problem.json"))

    out_root = os.path.join(base_dir, "outputs_PLATE_WITH_HOLE_FATIGUE_CASE1")
    ensure_dir(out_root)

    mapdl_cfg = cfg["mapdl"]
    run_location = mapdl_cfg["run_location"]
    jobname_prefix = mapdl_cfg["jobname_prefix"]
    override = bool(mapdl_cfg["override"])

    ensure_dir(run_location)
    cleanup_old_lock(run_location, jobname_prefix)

    state = {
        "cache": {},
        "history": [],
        "eval_count": 0,
        "max_eval": int(cfg["optimizer"]["max_eval"])
    }

    log("[MAPDL] launch start")
    mapdl = launch_mapdl(
        run_location=run_location,
        jobname=jobname_prefix,
        override=override,
        cleanup_on_exit=False
    )
    log(f"[MAPDL] launched: workdir={mapdl.directory}")

    best = None
    best_design = None
    base_design = None

    try:
        best, best_design, base_design = optimize_case(mapdl, out_root, cfg, state)
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

    final_dir = os.path.join(out_root, "final")
    ensure_dir(final_dir)

    if best is not None:
        best_eval_dir = os.path.join(out_root, f"eval_{best['eval']:03d}")
        for fname in ["model.step", "mesh.msh", "stress_max.png", "result.json"]:
            safe_copy(os.path.join(best_eval_dir, fname), os.path.join(final_dir, fname))

    summary = {
        "initial_design": base_design,
        "best_design": best_design,
        "best_result": best,
        "n_evaluations": state["eval_count"],
        "history": state["history"]
    }
    write_json(os.path.join(final_dir, "summary.json"), summary)

    log("\n==============================")
    log("[DONE]")
    if best is not None:
        log(f"Best plate_length = {best['design']['plate_length']:.6f} mm")
        log(f"Best plate_width  = {best['design']['plate_width']:.6f} mm")
        log(f"Best plate_thk    = {best['design']['plate_thk']:.6f} mm")
        log(f"Best hole_d       = {best['design']['hole_d']:.6f} mm")
        log(f"Best hotspot max  = {best['hotspot_eqv_max_maxload_MPa']:.6f} MPa")
        log(f"Best hotspot top3 = {best['hotspot_eqv_top3_maxload_MPa']:.6f} MPa")
        log(f"Best top3 life    = {best['top3_life_cycles']:.6e} cycles")
        log(f"Best mass         = {best['mass_kg']:.6f} kg")
        log(f"Feasible          = {best['feasible']}")
    log(f"Evaluations       = {state['eval_count']}")
    log(f"Outputs           = {final_dir}")
    log("==============================")


if __name__ == "__main__":
    main()