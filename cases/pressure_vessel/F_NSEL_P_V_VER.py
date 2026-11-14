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


def cleanup_old_lock(run_location: str, jobname_prefix: str):
    lock_path = os.path.join(run_location, f"{jobname_prefix}.lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            log(f"[LOCK] removed old lock: {lock_path}")
        except Exception:
            log(f"[LOCK] could not remove lock: {lock_path}")
            log("[LOCK] close old MAPDL process if still running.")


def to_apdl_path(path: str):
    return path.replace("\\", "/")


def set_mapdl_cwd(mapdl, path: str):
    apdl_path = to_apdl_path(os.path.abspath(path))
    mapdl.run(f"/CWD,'{apdl_path}'")
    log(f"    [MAPDL] /CWD -> {apdl_path}")


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

    with open(inp_path, "w", encoding="ascii", errors="ignore") as f:
        f.write("/PREP7\n")
        f.write("TYPE,1\n")
        f.write("MAT,1\n")

        for nid, xyz in zip(node_tags, node_xyz):
            f.write(f"N,{int(nid)},{float(xyz[0])},{float(xyz[1])},{float(xyz[2])}\n")

        f.write("TYPE,1\n")
        f.write("MAT,1\n")

        for conn in tet_conn:
            n1, n2, n3, n4 = map(int, conn.tolist())
            f.write(f"E,{n1},{n2},{n3},{n4}\n")

        f.write("FINISH\n")

    return inp_name, inp_path


def add_force_dicts(force_dict_a: dict, force_dict_b: dict):
    out = {}

    for nid, vec in force_dict_a.items():
        out[int(nid)] = np.array(vec, dtype=float)

    for nid, vec in force_dict_b.items():
        nid = int(nid)
        if nid not in out:
            out[nid] = np.zeros(3, dtype=float)
        out[nid] += np.array(vec, dtype=float)

    return out


# ============================================================
# design helpers
# ============================================================
def get_initial_design(cfg: dict):
    geo = cfg["geometry_mm"]

    design = {
        "main_length": float(geo["main_length"]),
        "main_inner_d": float(geo["main_inner_d"]),
        "main_wall_thk": float(geo["main_wall_thk"]),

        "nozzle_length": float(geo["nozzle_length"]),
        "nozzle_inner_d": float(geo["nozzle_inner_d"]),
        "nozzle_wall_thk": float(geo["nozzle_wall_thk"]),

        "pad_outer_d": float(geo["pad_outer_d"]),
        "pad_thk": float(geo["pad_thk"])
    }
    return design


def apply_dependent_rules(cfg: dict, design: dict):
    d = dict(design)

    if d["main_wall_thk"] <= 0.0:
        return None
    if d["nozzle_wall_thk"] <= 0.0:
        return None
    if d["pad_thk"] <= 0.0:
        return None

    R_main_in = 0.5 * d["main_inner_d"]
    R_main_out = R_main_in + d["main_wall_thk"]

    R_noz_in = 0.5 * d["nozzle_inner_d"]
    R_noz_out = R_noz_in + d["nozzle_wall_thk"]

    R_pad_out = 0.5 * d["pad_outer_d"]

    if R_main_in <= 0.0 or R_noz_in <= 0.0:
        return None
    if R_main_out <= R_main_in:
        return None
    if R_noz_out <= R_noz_in:
        return None
    if R_pad_out <= R_noz_out:
        return None
    if R_noz_out >= R_main_out:
        return None

    return d


def design_key(design: dict):
    return (
        round(float(design["main_wall_thk"]), 6),
        round(float(design["nozzle_wall_thk"]), 6),
        round(float(design["pad_thk"]), 6)
    )


def design_change_l1(design: dict, base_design: dict):
    keys = ["main_wall_thk", "nozzle_wall_thk", "pad_thk"]
    return float(sum(abs(float(design[k]) - float(base_design[k])) for k in keys))


def candidate_index_map(cands):
    return {round(float(v), 6): i for i, v in enumerate(cands)}


def generate_neighbors(cfg: dict, design: dict):
    ds = cfg["design_space_mm"]

    var_to_cands = {
        "main_wall_thk": [float(x) for x in ds["main_wall_thk_candidates"]],
        "nozzle_wall_thk": [float(x) for x in ds["nozzle_wall_thk_candidates"]],
        "pad_thk": [float(x) for x in ds["pad_thk_candidates"]]
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


# ============================================================
# geometry helper
# ============================================================
def cross_vessel_volume_mm3(design: dict):
    L_main = float(design["main_length"])
    R_main_in = 0.5 * float(design["main_inner_d"])
    R_main_out = R_main_in + float(design["main_wall_thk"])

    L_noz = float(design["nozzle_length"])
    R_noz_in = 0.5 * float(design["nozzle_inner_d"])
    R_noz_out = R_noz_in + float(design["nozzle_wall_thk"])

    R_pad_out = 0.5 * float(design["pad_outer_d"])
    pad_thk = float(design["pad_thk"])

    v_main_shell = math.pi * (R_main_out**2 - R_main_in**2) * L_main
    v_two_nozzle_shell = 2.0 * math.pi * (R_noz_out**2 - R_noz_in**2) * L_noz
    v_two_pads = 2.0 * math.pi * max(0.0, (R_pad_out**2 - R_noz_out**2)) * pad_thk

    return float(v_main_shell + v_two_nozzle_shell + v_two_pads)


# ============================================================
# coordinate-based cross vessel build + extract
# ============================================================
def gmsh_build_extract_cross_vessel_coord(eval_dir: str, cfg: dict, design: dict):
    mesh_cfg = cfg["mesh_mm"]
    outputs = cfg["outputs"]

    L_main = float(design["main_length"])
    R_main_in = 0.5 * float(design["main_inner_d"])
    R_main_out = R_main_in + float(design["main_wall_thk"])

    L_noz = float(design["nozzle_length"])
    R_noz_in = 0.5 * float(design["nozzle_inner_d"])
    R_noz_out = R_noz_in + float(design["nozzle_wall_thk"])

    R_pad_out = 0.5 * float(design["pad_outer_d"])
    pad_thk = float(design["pad_thk"])

    lc = float(mesh_cfg["lc"])

    step_path = os.path.join(eval_dir, "model.step")
    msh_path = os.path.join(eval_dir, "mesh.msh")

    mid_z = 0.5 * L_main

    log("  [GMSH] initialize")
    gmsh.initialize()
    gmsh.model.add("CROSS_PRESSURE_VESSEL_COORD")
    occ = gmsh.model.occ

    # outer solids
    body_main = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, L_main, R_main_out)
    body_noz_r = occ.addCylinder(0.0, 0.0, mid_z, L_noz, 0.0, 0.0, R_noz_out)
    body_noz_l = occ.addCylinder(0.0, 0.0, mid_z, -L_noz, 0.0, 0.0, R_noz_out)

    body_pad_r = occ.addCylinder(0.0, 0.0, mid_z, R_main_out + pad_thk, 0.0, 0.0, R_pad_out)
    body_pad_l = occ.addCylinder(0.0, 0.0, mid_z, -(R_main_out + pad_thk), 0.0, 0.0, R_pad_out)

    outer_fused, _ = occ.fuse(
        [(3, body_main)],
        [(3, body_noz_r), (3, body_noz_l), (3, body_pad_r), (3, body_pad_l)]
    )
    occ.synchronize()

    if len(outer_fused) != 1 or outer_fused[0][0] != 3:
        gmsh.finalize()
        raise RuntimeError("outer fuse did not return one 3D body")

    outer_tag = outer_fused[0][1]

    # inner bores
    bore_main = occ.addCylinder(0.0, 0.0, -10.0, 0.0, 0.0, L_main + 20.0, R_main_in)
    bore_noz_r = occ.addCylinder(-10.0, 0.0, mid_z, L_noz + 20.0, 0.0, 0.0, R_noz_in)
    bore_noz_l = occ.addCylinder(10.0, 0.0, mid_z, -(L_noz + 20.0), 0.0, 0.0, R_noz_in)

    bore_fused, _ = occ.fuse([(3, bore_main)], [(3, bore_noz_r), (3, bore_noz_l)])
    occ.synchronize()

    if len(bore_fused) != 1 or bore_fused[0][0] != 3:
        gmsh.finalize()
        raise RuntimeError("inner bore fuse did not return one 3D body")

    cut_out, _ = occ.cut([(3, outer_tag)], bore_fused)
    occ.synchronize()

    if len(cut_out) != 1 or cut_out[0][0] != 3:
        gmsh.finalize()
        raise RuntimeError("final cut did not return one 3D body")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    log("  [GMSH] mesh.generate(3) start")
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    log(f"  [GMSH] mesh.generate(3) done in {time.time() - t0:.2f}s")

    if outputs.get("save_step", True):
        gmsh.write(step_path)
    else:
        step_path = ""

    if outputs.get("save_msh", True):
        gmsh.write(msh_path)
    else:
        msh_path = ""

    # all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_xyz = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_tags.tolist())}

    # all tetra
    tet_conn = []
    for _, ent in gmsh.model.getEntities(3):
        types, _, elem_node_tags = gmsh.model.mesh.getElements(3, ent)
        for _, enodes in zip(types, elem_node_tags):
            enodes = np.asarray(enodes, dtype=np.int64)
            if enodes.size % 4 == 0:
                tet_conn.append(enodes.reshape(-1, 4))

    if not tet_conn:
        gmsh.finalize()
        raise RuntimeError("tet4 connectivity not found")

    tet_conn = np.vstack(tet_conn)

    # all surface triangles
    tri_list = []
    surf_node_set = set()

    for _, ent in gmsh.model.getEntities(2):
        types, _, elem_node_tags = gmsh.model.mesh.getElements(2, ent)
        for _, enodes in zip(types, elem_node_tags):
            enodes = np.asarray(enodes, dtype=np.int64)
            if enodes.size % 3 == 0:
                tris = enodes.reshape(-1, 3)
                tri_list.append(tris)
                for tri in tris:
                    surf_node_set.update(tri.tolist())

    if len(tri_list) == 0:
        gmsh.finalize()
        raise RuntimeError("surface triangles not found")

    all_surf_tris = np.vstack(tri_list)

    # coordinate-based BASE nodes
    ztol_base = max(1.0e-4, 1.0e-3 * lc)
    base_nodes = []
    for nid in surf_node_set:
        p = node_xyz[id_to_idx[int(nid)]]
        if abs(float(p[2]) - 0.0) <= ztol_base:
            base_nodes.append(int(nid))
    base_nodes = np.array(sorted(set(base_nodes)), dtype=np.int64)

    # coordinate-based INNER triangles
    inner_tri_keep = []
    rtol_main = max(1.0e-3, 0.05 * lc)
    rtol_noz = max(1.0e-3, 0.05 * lc)

    for tri in all_surf_tris:
        n1, n2, n3 = map(int, tri.tolist())
        p1 = node_xyz[id_to_idx[n1]]
        p2 = node_xyz[id_to_idx[n2]]
        p3 = node_xyz[id_to_idx[n3]]

        c = (p1 + p2 + p3) / 3.0
        x_c = float(c[0])
        y_c = float(c[1])
        z_c = float(c[2])

        area_vec = 0.5 * np.cross(p2 - p1, p3 - p1)
        anorm = float(np.linalg.norm(area_vec))
        if anorm <= 1.0e-14:
            continue

        nvec = area_vec / anorm

        # main vessel inner cylindrical surface
        r_main = math.hypot(x_c, y_c)
        if abs(r_main - R_main_in) <= rtol_main:
            if abs(float(nvec[2])) < 0.35:
                inner_tri_keep.append(tri)
                continue

        # nozzle inner cylindrical surfaces
        r_noz = math.hypot(y_c, z_c - mid_z)
        if abs(r_noz - R_noz_in) <= rtol_noz:
            if abs(float(nvec[0])) < 0.35:
                if abs(x_c) > 0.35 * R_main_out:
                    inner_tri_keep.append(tri)
                    continue

    if len(inner_tri_keep) == 0:
        gmsh.finalize()
        raise RuntimeError("INNER triangles not found by coordinate rule")

    inner_tris = np.asarray(inner_tri_keep, dtype=np.int64)

    inner_node_set = set()
    for tri in inner_tris:
        inner_node_set.update(tri.tolist())
    inner_nodes = np.array(sorted(inner_node_set), dtype=np.int64)

    log(f"  [COORD] BASE nodes count  = {len(base_nodes)}")
    log(f"  [COORD] INNER nodes count = {len(inner_nodes)}")
    log(f"  [COORD] INNER tris count  = {len(inner_tris)}")
    log("  [COORD] NOTE: saddle intersection inner surface may be partially missed")

    gmsh.finalize()

    return {
        "step_path": step_path,
        "msh_path": msh_path,
        "node_tags": node_tags,
        "node_xyz": node_xyz,
        "tet_conn": tet_conn,
        "base_nodes": base_nodes,
        "inner_nodes": inner_nodes,
        "inner_tris": inner_tris
    }


# ============================================================
# equivalent nodal loads
# ============================================================
def build_internal_pressure_nodal_loads(
    node_tags: np.ndarray,
    node_xyz: np.ndarray,
    inner_tris: np.ndarray,
    pressure_MPa: float
):
    p = float(pressure_MPa)
    if p <= 0.0:
        raise RuntimeError("internal_pressure_MPa must be > 0")

    id_to_idx = {int(nid): i for i, nid in enumerate(node_tags.tolist())}
    forces = {}
    total_scalar_area = 0.0
    total_force_vec = np.zeros(3, dtype=float)

    for tri in inner_tris:
        n1, n2, n3 = map(int, tri.tolist())

        p1 = node_xyz[id_to_idx[n1]]
        p2 = node_xyz[id_to_idx[n2]]
        p3 = node_xyz[id_to_idx[n3]]

        area_vec = 0.5 * np.cross(p2 - p1, p3 - p1)
        area = float(np.linalg.norm(area_vec))
        if area <= 1.0e-14:
            continue

        centroid = (p1 + p2 + p3) / 3.0
        r_main = math.hypot(float(centroid[0]), float(centroid[1]))

        if abs(float(centroid[0])) < 0.7 * max(1.0, r_main):
            radial = np.array([centroid[0], centroid[1], 0.0], dtype=float)
            rnorm = float(np.linalg.norm(radial))
            if rnorm <= 1.0e-14:
                continue
            edir = radial / rnorm
        else:
            nrm = float(np.linalg.norm(area_vec))
            if nrm <= 1.0e-14:
                continue
            edir = area_vec / nrm

        tri_force = p * area * edir

        total_scalar_area += area
        total_force_vec += tri_force

        share = tri_force / 3.0
        for nid in [n1, n2, n3]:
            if nid not in forces:
                forces[nid] = np.zeros(3, dtype=float)
            forces[nid] += share

    return forces, total_scalar_area, total_force_vec


def build_self_weight_nodal_loads(
    node_tags: np.ndarray,
    node_xyz: np.ndarray,
    tet_conn: np.ndarray,
    density_kg_m3: float,
    gravity_m_s2: float
):
    rho = float(density_kg_m3)
    g = float(gravity_m_s2)
    gamma_N_per_mm3 = rho * g / 1.0e9

    id_to_idx = {int(nid): i for i, nid in enumerate(node_tags.tolist())}
    forces = {}
    total_volume = 0.0
    total_force_vec = np.zeros(3, dtype=float)

    body_force = np.array([0.0, 0.0, -gamma_N_per_mm3], dtype=float)

    for conn in tet_conn:
        n1, n2, n3, n4 = map(int, conn.tolist())

        p1 = node_xyz[id_to_idx[n1]]
        p2 = node_xyz[id_to_idx[n2]]
        p3 = node_xyz[id_to_idx[n3]]
        p4 = node_xyz[id_to_idx[n4]]

        mat = np.column_stack((p2 - p1, p3 - p1, p4 - p1))
        vol = abs(np.linalg.det(mat)) / 6.0
        if vol <= 1.0e-14:
            continue

        elem_force = body_force * vol
        total_volume += vol
        total_force_vec += elem_force

        nodal_share = elem_force / 4.0
        for nid in [n1, n2, n3, n4]:
            if nid not in forces:
                forces[nid] = np.zeros(3, dtype=float)
            forces[nid] += nodal_share

    return forces, total_volume, total_force_vec


# ============================================================
# MAPDL solve
# ============================================================
def solve_one_eval(mapdl, eval_idx: int, eval_dir: str, gm: dict, cfg: dict, design: dict):
    mat = cfg["material_MPa"]
    cons = cfg["constraint"]
    outputs = cfg["outputs"]
    load_cfg = cfg["load"]

    E = float(mat["E"])
    nu = float(mat["nu"])
    density_kg_m3 = float(mat["density_kg_m3"])

    pressure_MPa = float(load_cfg["internal_pressure_MPa"])
    use_self_weight = bool(load_cfg.get("use_self_weight", False))
    gravity_m_s2 = float(load_cfg.get("gravity_m_s2", 9.81))

    stress_limit = float(cons["stress_limit_MPa"])
    self_weight_limit_N = float(cons["self_weight_limit_N"])

    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    inner_tris = gm["inner_tris"].astype(int)

    pressure_forces, loaded_area, pressure_force_vec = build_internal_pressure_nodal_loads(
        node_tags=node_tags,
        node_xyz=node_xyz,
        inner_tris=inner_tris,
        pressure_MPa=pressure_MPa
    )

    if use_self_weight:
        gravity_forces, mesh_volume, gravity_force_vec = build_self_weight_nodal_loads(
            node_tags=node_tags,
            node_xyz=node_xyz,
            tet_conn=tet_conn,
            density_kg_m3=density_kg_m3,
            gravity_m_s2=gravity_m_s2
        )
    else:
        gravity_forces = {}
        mesh_volume = 0.0
        gravity_force_vec = np.zeros(3, dtype=float)

    total_forces = add_force_dicts(pressure_forces, gravity_forces)
    total_force_vec = pressure_force_vec + gravity_force_vec
    self_weight_N = float(abs(gravity_force_vec[2]))

    workdir = mapdl.directory
    build_inp_name, build_inp_path = write_mapdl_build_inp(
        workdir,
        eval_idx,
        node_tags,
        node_xyz,
        tet_conn
    )

    cdb_base_name = f"cross_eval_{eval_idx:03d}"
    cdb_work_path_lower = os.path.join(workdir, f"{cdb_base_name}.cdb")
    cdb_eval_path = os.path.join(eval_dir, "model.cdb")
    png_path = os.path.join(eval_dir, "stress.png")

    log(f"    [MAPDL] workdir = {workdir}")
    log(f"    [MAPDL] build inp = {build_inp_path}")

    mapdl.clear()

    mapdl.prep7()
    mapdl.et(1, 285)
    mapdl.type(1)
    mapdl.mat(1)
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, nu)
    mapdl.finish()

    set_mapdl_cwd(mapdl, workdir)

    if not os.path.exists(build_inp_path):
        raise RuntimeError(f"Build inp file does not exist: {build_inp_path}")

    t0 = time.time()
    mapdl.list_files = lambda *args, **kwargs: []
    mapdl.input(build_inp_path)
    log(f"    [MAPDL] input build file done in {time.time() - t0:.2f}s")

    mapdl.prep7()
    mapdl.allsel()
    mapdl.run("*GET,NNODE_BUILD,NODE,0,COUNT")
    mapdl.run("*GET,NELEM_BUILD,ELEM,0,COUNT")
    nnode_build = int(mapdl.parameters["NNODE_BUILD"])
    nelem_build = int(mapdl.parameters["NELEM_BUILD"])
    log(f"    [MAPDL] after build: nodes={nnode_build}, elems={nelem_build}")

    if nnode_build <= 0 or nelem_build <= 0:
        raise RuntimeError("Build failed: empty DB")

    # --------------------------------------------------------
    # BASE by NSEL
    # --------------------------------------------------------
    ztol_base = max(1.0e-4, 1.0e-3 * float(cfg["mesh_mm"]["lc"]))

    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Z", -ztol_base, ztol_base)
    mapdl.run("*GET,NCOUNT_BASE,NODE,0,COUNT")
    ncount_base = int(mapdl.parameters["NCOUNT_BASE"])
    log(f"    [COORD-MAPDL] BASE selected by Z=0 tol={ztol_base:.6e}, count={ncount_base}")
    if ncount_base <= 0:
        raise RuntimeError("BASE selection by coordinate rule failed")
    mapdl.cm("BASE", "NODE")
    mapdl.allsel()

    # --------------------------------------------------------
    # debug: TRUE MAPDL NSEL-only INNER component
    # corrected version: main/nozzle selected separately, then union
    # --------------------------------------------------------
    L_main = float(design["main_length"])
    R_main_in = 0.5 * float(design["main_inner_d"])
    L_noz = float(design["nozzle_length"])
    R_noz_in = 0.5 * float(design["nozzle_inner_d"])
    mid_z = 0.5 * L_main
    tol_dbg = max(1.0e-3, 0.05 * float(cfg["mesh_mm"]["lc"]))

    # main-cylinder-like selection
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Z", 0.0 + tol_dbg, L_main - tol_dbg)
    mapdl.nsel("R", "LOC", "X", -R_main_in - tol_dbg, R_main_in + tol_dbg)
    mapdl.nsel("R", "LOC", "Y", -R_main_in - tol_dbg, R_main_in + tol_dbg)
    mapdl.run("*GET,NCOUNT_INNER_MAIN_NSEL,NODE,0,COUNT")
    ncount_inner_main = int(mapdl.parameters["NCOUNT_INNER_MAIN_NSEL"])
    log(f"    [COORD-MAPDL] INNER_MAIN_NSEL debug count={ncount_inner_main}")
    if ncount_inner_main > 0:
        mapdl.cm("INNER_MAIN_NSEL", "NODE")

    # nozzle-like selection
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Z", mid_z - R_noz_in - tol_dbg, mid_z + R_noz_in + tol_dbg)
    mapdl.nsel("R", "LOC", "X", -(L_noz - tol_dbg), (L_noz - tol_dbg))
    mapdl.nsel("R", "LOC", "Y", -R_noz_in - tol_dbg, R_noz_in + tol_dbg)
    mapdl.run("*GET,NCOUNT_INNER_NOZ_NSEL,NODE,0,COUNT")
    ncount_inner_noz = int(mapdl.parameters["NCOUNT_INNER_NOZ_NSEL"])
    log(f"    [COORD-MAPDL] INNER_NOZ_NSEL debug count={ncount_inner_noz}")
    if ncount_inner_noz > 0:
        mapdl.cm("INNER_NOZ_NSEL", "NODE")

    # union
    mapdl.allsel()
    if ncount_inner_main > 0:
        mapdl.cmsel("S", "INNER_MAIN_NSEL")
        if ncount_inner_noz > 0:
            mapdl.cmsel("A", "INNER_NOZ_NSEL")
        mapdl.cm("INNER_NSEL", "NODE")
    elif ncount_inner_noz > 0:
        mapdl.cmsel("S", "INNER_NOZ_NSEL")
        mapdl.cm("INNER_NSEL", "NODE")

    mapdl.run("*GET,NCOUNT_INNER_NSEL,NODE,0,COUNT")
    ncount_inner_nsel = int(mapdl.parameters["NCOUNT_INNER_NSEL"])
    log(f"    [COORD-MAPDL] INNER_NSEL debug final count={ncount_inner_nsel}")
    mapdl.allsel()

    mapdl.finish()

    if outputs.get("save_cdb", True):
        if os.path.exists(cdb_work_path_lower):
            os.remove(cdb_work_path_lower)

        set_mapdl_cwd(mapdl, workdir)
        mapdl.prep7()
        mapdl.allsel()
        mapdl.cdwrite("DB", cdb_base_name, "cdb")
        mapdl.finish()

        if not os.path.exists(cdb_work_path_lower):
            raise RuntimeError(f"CDB was not created: {cdb_work_path_lower}")

        safe_copy(cdb_work_path_lower, cdb_eval_path)
        log("    [MAPDL] CDB saved (current DB used for solve)")

    mapdl.slashsolu()
    mapdl.antype("STATIC")

    mapdl.cmsel("S", "BASE")
    mapdl.run("*GET,NFIX,NODE,0,COUNT")
    nfix = int(mapdl.parameters["NFIX"])
    if nfix <= 0:
        raise RuntimeError("BASE component empty")
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    batch = []
    count_force_nodes = 0
    for nid, fvec in total_forces.items():
        fx, fy, fz = float(fvec[0]), float(fvec[1]), float(fvec[2])

        if abs(fx) > 1.0e-14:
            batch.append(f"F,{int(nid)},FX,{fx}")
        if abs(fy) > 1.0e-14:
            batch.append(f"F,{int(nid)},FY,{fy}")
        if abs(fz) > 1.0e-14:
            batch.append(f"F,{int(nid)},FZ,{fz}")

        count_force_nodes += 1

        if len(batch) >= 2000:
            mapdl.input_strings(batch)
            batch = []

    if batch:
        mapdl.input_strings(batch)

    log(f"      pressure={pressure_MPa:.6f} MPa, loaded area={loaded_area:.6f} mm^2")
    log(f"      force nodes={count_force_nodes}, total resultant force={total_force_vec}")

    t0 = time.time()
    mapdl.solve()
    log(f"    [MAPDL] solve done in {time.time() - t0:.2f}s")
    mapdl.finish()

    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()

    seqv = np.asarray(mapdl.post_processing.nodal_eqv_stress(), dtype=float).reshape(-1)
    all_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)

    if seqv.size == 0:
        raise RuntimeError("POST: nodal equivalent stress array is empty")

    valid = np.isfinite(seqv)
    seqv_valid = seqv[valid]
    if seqv_valid.size == 0:
        raise RuntimeError("POST: no finite stress")

    if all_nodes.size == seqv.size:
        nodes_valid = all_nodes[valid]
        idx = int(np.argmax(seqv_valid))
        node_at_smax = int(nodes_valid[idx])
    else:
        node_at_smax = -1

    smax = float(np.max(seqv_valid))
    smin = float(np.min(seqv_valid))

    stress_ok = bool(smax <= stress_limit)
    weight_ok = bool(self_weight_N <= self_weight_limit_N)
    feasible = bool(stress_ok and weight_ok)

    png_ok = False
    if outputs.get("save_png", True):
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
        "Smax_MPa": smax,
        "Smin_MPa": smin,
        "node_at_Smax": node_at_smax,
        "stress_ok": stress_ok,
        "weight_ok": weight_ok,
        "feasible": feasible,
        "loaded_area_mm2": loaded_area,
        "pressure_MPa": pressure_MPa,
        "use_self_weight": use_self_weight,
        "gravity_m_s2": gravity_m_s2,
        "density_kg_m3": density_kg_m3,
        "mesh_volume_mm3": mesh_volume,
        "self_weight_N": self_weight_N,
        "pressure_resultant_force_vector_N": pressure_force_vec.tolist(),
        "gravity_resultant_force_vector_N": gravity_force_vec.tolist(),
        "total_resultant_force_vector_N": total_force_vec.tolist(),
        "cdb_path": cdb_eval_path if outputs.get("save_cdb", True) else "",
        "png_path": png_path if png_ok else ""
    }


# ============================================================
# evaluation
# ============================================================
def objective_score(rec: dict, cfg: dict, base_design: dict):
    cons = cfg["constraint"]
    opt = cfg["optimizer"]

    stress_limit = float(cons["stress_limit_MPa"])
    weight_limit = float(cons["self_weight_limit_N"])

    feasible_weight_coeff = float(opt["feasible_weight_coeff"])
    feasible_stress_coeff = float(opt["feasible_stress_coeff"])
    change_penalty_coeff = float(opt["change_penalty_coeff"])
    stress_violation_coeff = float(opt["stress_violation_coeff"])
    weight_violation_coeff = float(opt["weight_violation_coeff"])

    s = float(rec["Smax_MPa"])
    w = float(rec["self_weight_N"])
    change = float(rec["design_change_l1"])

    stress_violation = max(0.0, s - stress_limit)
    weight_violation = max(0.0, w - weight_limit)

    if stress_violation == 0.0 and weight_violation == 0.0:
        return feasible_weight_coeff * w + feasible_stress_coeff * s + change_penalty_coeff * change

    return (
        stress_violation_coeff * (stress_violation ** 2)
        + weight_violation_coeff * (weight_violation ** 2)
        + feasible_weight_coeff * w
        + feasible_stress_coeff * s
        + change_penalty_coeff * change
    )


_FAIL = {"n": 0}  # 연속으로 해석에 실패한 횟수


def evaluate_design(mapdl, out_root: str, cfg: dict, eval_idx: int, design: dict, base_design: dict):
    eval_dir = os.path.join(out_root, f"eval_{eval_idx:03d}")
    ensure_dir(eval_dir)

    log("\n============================================================")
    log(
        f"[EVAL {eval_idx:03d}] "
        f"main_wall_thk={design['main_wall_thk']:.3f} | "
        f"nozzle_wall_thk={design['nozzle_wall_thk']:.3f} | "
        f"pad_thk={design['pad_thk']:.3f}"
    )
    log("============================================================")

    vol_mm3 = cross_vessel_volume_mm3(design)
    gm = None
    failed_msg = None
    try:
        gm = gmsh_build_extract_cross_vessel_coord(eval_dir, cfg, design)
        res = solve_one_eval(mapdl, eval_idx, eval_dir, gm, cfg, design)
        _FAIL["n"] = 0
    except Exception as e:
        failed_msg = f"{type(e).__name__}: {str(e)[:300]}"
        _FAIL["n"] += 1
        log(f"  [FAIL] 이 설계는 해석에 실패했어요. 불합격으로 처리하고 계속해요: {failed_msg}")

        # ANSYS와 Gmsh를 다음 해석에 쓸 수 있게 정리해요
        try:
            mapdl.finish()
            mapdl.clear()
        except Exception:
            pass
        try:
            gmsh.finalize()
        except Exception:
            pass

        if _FAIL["n"] >= 3:
            raise RuntimeError("연속 3번 해석에 실패해서 중단해요 (ANSYS 상태를 확인하세요)")

        if gm is None:
            gm = {"step_path": "", "msh_path": ""}
        density = float(cfg.get("material_MPa", {}).get("density_kg_m3", 7850.0))
        gravity = float(cfg.get("load", {}).get("gravity_m_s2", 9.81))
        weight_n = vol_mm3 * 1.0e-9 * density * gravity
        weight_limit = float(cfg["constraint"]["self_weight_limit_N"])
        res = {
            "Smax_MPa": 1.0e4,  # 풀리지 않은 설계에 주는 벌점 (실제 계산값이 아니에요)
            "Smin_MPa": 0.0,
            "self_weight_N": weight_n,
            "stress_ok": False,
            "weight_ok": weight_n <= weight_limit,
            "feasible": False,
            "node_at_Smax": -1,
            "loaded_area_mm2": 0.0,
            "mesh_volume_mm3": 0.0,
            "pressure_resultant_force_vector_N": [0.0, 0.0, 0.0],
            "gravity_resultant_force_vector_N": [0.0, 0.0, 0.0],
            "total_resultant_force_vector_N": [0.0, 0.0, 0.0],
            "cdb_path": "",
            "png_path": "",
        }

    rec = {
        "eval": eval_idx,
        "design": dict(design),
        "volume_mm3": vol_mm3,
        "Smax_MPa": res["Smax_MPa"],
        "Smin_MPa": res["Smin_MPa"],
        "self_weight_N": res["self_weight_N"],
        "stress_ok": res["stress_ok"],
        "weight_ok": res["weight_ok"],
        "feasible": res["feasible"],
        "node_at_Smax": res["node_at_Smax"],
        "loaded_area_mm2": res["loaded_area_mm2"],
        "mesh_volume_mm3": res["mesh_volume_mm3"],
        "pressure_resultant_force_vector_N": res["pressure_resultant_force_vector_N"],
        "gravity_resultant_force_vector_N": res["gravity_resultant_force_vector_N"],
        "total_resultant_force_vector_N": res["total_resultant_force_vector_N"],
        "design_change_l1": design_change_l1(design, base_design),
        "files": {
            "step": gm["step_path"],
            "msh": gm["msh_path"],
            "cdb": res["cdb_path"],
            "png": res["png_path"]
        }
    }
    if failed_msg is not None:
        rec["solve_failed"] = True
        rec["error"] = failed_msg

    rec["score"] = objective_score(rec, cfg, base_design)
    write_json(os.path.join(eval_dir, "result.json"), rec)

    log(
        f"[RESULT] Smax={rec['Smax_MPa']:.6f} MPa | "
        f"self-weight={rec['self_weight_N']:.6f} N | "
        f"feasible={rec['feasible']} | "
        f"score={rec['score']:.6f}"
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


def optimize_case2(mapdl, out_root, cfg, state):
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

    out_root = os.path.join(base_dir, "outputs_CROSS_PRESSURE_VESSEL_COORD")
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

    log("[MAPDL] single launch start")
    mapdl = launch_mapdl(
        run_location=run_location,
        jobname=jobname_prefix,
        override=override
    )
    log(f"[MAPDL] launched: workdir = {mapdl.directory}")

    best = None
    best_design = None
    base_design = None

    try:
        best, best_design, base_design = optimize_case2(mapdl, out_root, cfg, state)
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
        log(f"Best main_wall_thk   = {best['design']['main_wall_thk']:.6f} mm")
        log(f"Best nozzle_wall_thk = {best['design']['nozzle_wall_thk']:.6f} mm")
        log(f"Best pad_thk         = {best['design']['pad_thk']:.6f} mm")
        log(f"Best Smax            = {best['Smax_MPa']:.6f} MPa")
        log(f"Best self-weight     = {best['self_weight_N']:.6f} N")
        log(f"Feasible             = {best['feasible']}")
    log(f"Evaluations          = {state['eval_count']}")
    log(f"Outputs              = {final_dir}")
    log("==============================")


if __name__ == "__main__":
    main()