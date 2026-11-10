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
    out = mapdl.run(f"/CWD,'{apdl_path}'")
    log(f"    [MAPDL] /CWD -> {apdl_path}")
    return out


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


def make_component_from_nodes(mapdl, comp_name: str, node_ids: np.ndarray):
    node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    if node_ids.size == 0:
        raise RuntimeError(f"{comp_name} node list is empty")

    node_ids = np.unique(node_ids)
    node_ids.sort()

    mapdl.allsel()
    mapdl.run("*GET,NNODE_NOW,NODE,0,COUNT")
    mapdl.run("*GET,NELEM_NOW,ELEM,0,COUNT")

    nnode_now = int(mapdl.parameters["NNODE_NOW"])
    nelem_now = int(mapdl.parameters["NELEM_NOW"])
    log(f"    [MAPDL] {comp_name}: current DB nodes={nnode_now}, elems={nelem_now}")

    if nnode_now <= 0:
        raise RuntimeError(f"{comp_name} component creation failed: MAPDL DB node count is zero")

    mapdl_nodes = np.asarray(mapdl.mesh.nnum, dtype=int).reshape(-1)
    if mapdl_nodes.size == 0:
        raise RuntimeError(f"{comp_name} component creation failed: mapdl.mesh.nnum is empty")

    common_ids = np.intersect1d(node_ids, mapdl_nodes)
    log(f"    [MAPDL] {comp_name}: requested={len(node_ids)}, matched={len(common_ids)}")

    if common_ids.size == 0:
        log(f"    [MAPDL] {comp_name}: requested min/max = {int(node_ids.min())} / {int(node_ids.max())}")
        log(f"    [MAPDL] {comp_name}: mapdl    min/max = {int(mapdl_nodes.min())} / {int(mapdl_nodes.max())}")
        raise RuntimeError(f"{comp_name} component creation failed: no matching node ids between Gmsh and MAPDL")

    ranges = compress_ranges(common_ids)
    if len(ranges) == 0:
        raise RuntimeError(f"{comp_name} component creation failed: matched node range is empty")

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


def make_component_by_z(mapdl, comp_name: str, zmin: float, ztol_small: float = 1.0e-4, ztol_large: float = 5.0e-2):
    mapdl.allsel()
    log(f"    [MAPDL] fallback {comp_name} by Z = [{zmin - ztol_small:.6f}, {zmin + ztol_small:.6f}]")
    mapdl.nsel("S", "LOC", "Z", zmin - ztol_small, zmin + ztol_small)
    mapdl.run(f"*GET,NCOUNT_{comp_name},NODE,0,COUNT")
    ncount = int(mapdl.parameters[f"NCOUNT_{comp_name}"])

    if ncount <= 0:
        log(f"    [MAPDL] fallback retry {comp_name} by Z = [{zmin - ztol_large:.6f}, {zmin + ztol_large:.6f}]")
        mapdl.nsel("S", "LOC", "Z", zmin - ztol_large, zmin + ztol_large)
        mapdl.run(f"*GET,NCOUNT_{comp_name},NODE,0,COUNT")
        ncount = int(mapdl.parameters[f"NCOUNT_{comp_name}"])

    if ncount <= 0:
        mapdl.allsel()
        raise RuntimeError(f"{comp_name} creation by z-fallback failed: selected node count is zero")

    mapdl.cm(comp_name, "NODE")
    mapdl.allsel()
    log(f"    [MAPDL] fallback {comp_name} done: ncount={ncount}")


# ============================================================
# geometry helpers
# ============================================================
def flange_pipe_volume_mm3(cfg: dict, wall_thk: float):
    geo = cfg["geometry_mm"]

    L = float(geo["length_total"])
    Di = float(geo["pipe_inner_d"])
    Do = Di + 2.0 * wall_thk

    Dfo = float(geo["flange_outer_d"])
    Dfi = float(geo["flange_inner_d"])
    tf = float(geo["flange_thk"])

    db = float(geo["bolt_hole_d"])
    nb = int(geo["n_bolt_holes"])

    pipe_len_mid = L - 2.0 * tf
    if pipe_len_mid <= 0.0:
        return float("inf")

    v_pipe_shell = math.pi * (Do**2 - Di**2) * 0.25 * pipe_len_mid
    v_flange_ring_one = math.pi * (Dfo**2 - Dfi**2) * 0.25 * tf - nb * math.pi * (db**2) * 0.25 * tf
    v_total = v_pipe_shell + 2.0 * v_flange_ring_one
    return float(v_total)


def bolt_hole_centers(bcd: float, n_holes: int):
    r = 0.5 * bcd
    centers = []
    for i in range(n_holes):
        ang = 2.0 * math.pi * i / float(n_holes)
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        centers.append((x, y))
    return centers


# ============================================================
# Gmsh build + extract
# ============================================================
def gmsh_build_extract_flanged_pipe(eval_dir: str, cfg: dict, wall_thk: float):
    geo = cfg["geometry_mm"]
    mesh_cfg = cfg["mesh_mm"]
    tags = cfg["physical_tags"]
    outputs = cfg["outputs"]

    L = float(geo["length_total"])
    Di = float(geo["pipe_inner_d"])
    tf = float(geo["flange_thk"])
    Dfo = float(geo["flange_outer_d"])
    db = float(geo["bolt_hole_d"])
    bcd = float(geo["bolt_circle_d"])
    n_holes = int(geo["n_bolt_holes"])

    lc = float(mesh_cfg["lc"])

    BASE = int(tags["BASE"])
    INNER = int(tags["INNER"])
    VOL = int(tags["VOL"])

    wall_thk = float(wall_thk)
    if wall_thk <= 0.0:
        raise RuntimeError("pipe_wall_thk must be > 0")

    Do = Di + 2.0 * wall_thk
    Ri = 0.5 * Di
    Ro = 0.5 * Do
    Rfo = 0.5 * Dfo

    mid_len = L - 2.0 * tf
    if mid_len <= 0.0:
        raise RuntimeError("length_total must be larger than 2 * flange_thk")

    step_path = os.path.join(eval_dir, "model.step")
    msh_path = os.path.join(eval_dir, "mesh.msh")

    log("  [GMSH] initialize")
    gmsh.initialize()
    gmsh.model.add("FLANGED_PIPE")
    occ = gmsh.model.occ

    bottom_flange = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, tf, Rfo)
    pipe_mid = occ.addCylinder(0.0, 0.0, tf, 0.0, 0.0, mid_len, Ro)
    top_flange = occ.addCylinder(0.0, 0.0, L - tf, 0.0, 0.0, tf, Rfo)

    outer_fused, _ = occ.fuse([(3, bottom_flange)], [(3, pipe_mid), (3, top_flange)])
    occ.synchronize()

    if len(outer_fused) != 1 or outer_fused[0][0] != 3:
        gmsh.finalize()
        raise RuntimeError("outer fuse did not return one 3D body")

    outer_tag = outer_fused[0][1]

    bore = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, L, Ri)

    holes = []
    for x, y in bolt_hole_centers(bcd, n_holes):
        h1 = occ.addCylinder(x, y, 0.0, 0.0, 0.0, tf, 0.5 * db)
        h2 = occ.addCylinder(x, y, L - tf, 0.0, 0.0, tf, 0.5 * db)
        holes.append((3, h1))
        holes.append((3, h2))

    cut_tools = [(3, bore)] + holes
    cut_out, _ = occ.cut([(3, outer_tag)], cut_tools)
    occ.synchronize()

    if len(cut_out) != 1 or cut_out[0][0] != 3:
        gmsh.finalize()
        raise RuntimeError("final cut did not return one 3D body")

    final_vol_tag = cut_out[0][1]

    log("  [GMSH] detect BASE/INNER surfaces")
    base_surfs = []
    inner_surfs = []

    tol = 1.0e-4
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin - 0.0) < tol and abs(zmax - 0.0) < tol:
            base_surfs.append(tag)
            continue

        rx = max(abs(xmin), abs(xmax))
        ry = max(abs(ymin), abs(ymax))
        rbox = max(rx, ry)
        zspan = zmax - zmin

        if abs(rbox - Ri) < 1.0 and zspan > 0.5 * tf:
            inner_surfs.append(tag)

    if not base_surfs:
        gmsh.finalize()
        raise RuntimeError("BASE surface not found")

    if not inner_surfs:
        gmsh.finalize()
        raise RuntimeError("INNER pressure surfaces not found")

    gmsh.model.addPhysicalGroup(2, base_surfs, BASE)
    gmsh.model.setPhysicalName(2, BASE, "BASE")

    gmsh.model.addPhysicalGroup(2, inner_surfs, INNER)
    gmsh.model.setPhysicalName(2, INNER, "INNER")

    gmsh.model.addPhysicalGroup(3, [final_vol_tag], VOL)
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

    log("  [GMSH] getVolumeElements start")
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
    log(f"  [GMSH] getVolumeElements done: nelem={len(tet_conn)}")

    def get_phys_surface_nodes_and_tris(phys_tag: int):
        surf_entities = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
        node_set = set()
        tri_list = []

        for ent in surf_entities:
            types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, ent)

            for etype, enodes in zip(types, elem_node_tags):
                enodes = np.asarray(enodes, dtype=np.int64)

                if enodes.size % 3 == 0:
                    tris = enodes.reshape(-1, 3)
                    tri_list.append(tris)
                    for tri in tris:
                        node_set.update(tri.tolist())

        if len(tri_list) == 0:
            tri_all = np.zeros((0, 3), dtype=np.int64)
        else:
            tri_all = np.vstack(tri_list)

        nodes = np.array(sorted(node_set), dtype=np.int64)
        return nodes, tri_all

    log("  [GMSH] get BASE nodes start")
    base_nodes, _ = get_phys_surface_nodes_and_tris(BASE)
    log(f"  [GMSH] get BASE nodes done: nbase={len(base_nodes)}")

    log("  [GMSH] get INNER nodes/tris start")
    inner_nodes, inner_tris = get_phys_surface_nodes_and_tris(INNER)
    log(f"  [GMSH] get INNER nodes/tris done: ninner={len(inner_nodes)}, ntri={len(inner_tris)}")

    gmsh.finalize()
    log("  [GMSH] finalize done")

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
# equivalent nodal load from internal pressure
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
        radial = np.array([centroid[0], centroid[1], 0.0], dtype=float)
        rnorm = float(np.linalg.norm(radial))
        if rnorm <= 1.0e-14:
            continue

        e_r = radial / rnorm
        tri_force = p * area * e_r

        total_scalar_area += area
        total_force_vec += tri_force

        share = tri_force / 3.0
        for nid in [n1, n2, n3]:
            if nid not in forces:
                forces[nid] = np.zeros(3, dtype=float)
            forces[nid] += share

    return forces, total_scalar_area, total_force_vec


# ============================================================
# MAPDL solve
# ============================================================
def solve_one_eval(mapdl, eval_idx: int, eval_dir: str, gm: dict, cfg: dict):
    mat = cfg["material_MPa"]
    cons = cfg["constraint"]
    outputs = cfg["outputs"]
    load_cfg = cfg["load"]

    E = float(mat["E"])
    nu = float(mat["nu"])
    pressure_MPa = float(load_cfg["internal_pressure_MPa"])
    stress_limit = float(cons["stress_limit_MPa"])

    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    base_nodes = gm["base_nodes"].astype(int)
    inner_tris = gm["inner_tris"].astype(int)

    id_to_idx = {int(nid): i for i, nid in enumerate(node_tags.tolist())}
    base_zvals = []
    for nid in base_nodes:
        nid = int(nid)
        if nid in id_to_idx:
            base_zvals.append(float(node_xyz[id_to_idx[nid], 2]))

    if len(base_zvals) == 0:
        raise RuntimeError("Could not compute base z values from Gmsh base_nodes")

    base_zmin = float(np.min(base_zvals))

    nodal_forces, loaded_area, total_force_vec = build_internal_pressure_nodal_loads(
        node_tags=node_tags,
        node_xyz=node_xyz,
        inner_tris=inner_tris,
        pressure_MPa=pressure_MPa
    )

    workdir = mapdl.directory
    build_inp_name, build_inp_path = write_mapdl_build_inp(
        workdir,
        eval_idx,
        node_tags,
        node_xyz,
        tet_conn
    )

    cdb_base_name = f"flange_eval_{eval_idx:03d}"
    cdb_work_base = os.path.join(workdir, cdb_base_name)
    cdb_work_path_lower = os.path.join(workdir, f"{cdb_base_name}.cdb")
    cdb_work_path_upper = os.path.join(workdir, f"{cdb_base_name}.CDB")
    cdb_eval_path = os.path.join(eval_dir, "model.cdb")
    png_path = os.path.join(eval_dir, "stress.png")

    log(f"    [MAPDL] workdir = {workdir}")
    log(f"    [MAPDL] build inp = {build_inp_path}")

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

    set_mapdl_cwd(mapdl, workdir)

    if not os.path.exists(build_inp_path):
        raise RuntimeError(f"Build inp file does not exist: {build_inp_path}")

    log(f"    [MAPDL] input build file start: {build_inp_name}")
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

    if nnode_build <= 0:
        raise RuntimeError("After build file input, MAPDL node count is zero")
    if nelem_build <= 0:
        raise RuntimeError("After build file input, MAPDL element count is zero")

    log("    [MAPDL] make BASE component before CDWRITE")
    try:
        make_component_from_nodes(mapdl, "BASE", base_nodes)
        log("    [MAPDL] BASE exact-id method done before CDWRITE")
    except Exception as e:
        log(f"    [MAPDL] BASE exact-id method failed before CDWRITE: {e}")
        make_component_by_z(mapdl, "BASE", base_zmin)

    mapdl.finish()

    if outputs.get("save_cdb", True):
        log("    [MAPDL] CDWRITE start")

        if os.path.exists(cdb_work_path_lower):
            os.remove(cdb_work_path_lower)
        if os.path.exists(cdb_work_path_upper):
            os.remove(cdb_work_path_upper)

        set_mapdl_cwd(mapdl, workdir)
        mapdl.prep7()
        mapdl.allsel()
        mapdl.cdwrite("DB", cdb_work_base, "cdb")
        mapdl.finish()

        cdb_found = ""
        if os.path.exists(cdb_work_path_lower):
            cdb_found = cdb_work_path_lower
        elif os.path.exists(cdb_work_path_upper):
            cdb_found = cdb_work_path_upper

        if cdb_found == "":
            files_now = sorted(os.listdir(workdir))
            raise RuntimeError(
                "CDWRITE failed: CDB file not created.\n"
                f"expected: {cdb_work_path_lower}\n"
                f"or: {cdb_work_path_upper}\n"
                f"MAPDL workdir: {workdir}\n"
                f"files in workdir: {files_now}"
            )

        safe_copy(cdb_found, cdb_eval_path)
        log(f"    [MAPDL] CDWRITE done: {cdb_found}")

        log("    [MAPDL] CLEAR + CDREAD start")
        mapdl.clear()
        set_mapdl_cwd(mapdl, workdir)

        if not os.path.exists(cdb_found):
            raise RuntimeError(f"CDB file disappeared before CDREAD: {cdb_found}")

        mapdl.prep7()
        mapdl.cdread("DB", cdb_work_base, "cdb")
        mapdl.mp("EX", 1, E)
        mapdl.mp("PRXY", 1, nu)

        mapdl.run("*GET,NNODECHK,NODE,0,COUNT")
        mapdl.run("*GET,NELEMCHK,ELEM,0,COUNT")
        nnode_chk = int(mapdl.parameters["NNODECHK"])
        nelem_chk = int(mapdl.parameters["NELEMCHK"])
        log(f"    [MAPDL] after CDREAD: nodes={nnode_chk}, elems={nelem_chk}")

        if nnode_chk <= 0:
            files_now = sorted(os.listdir(workdir))
            raise RuntimeError(
                "After CDREAD, MAPDL node count is zero\n"
                f"workdir={workdir}\n"
                f"cdb_found={cdb_found}\n"
                f"files in workdir={files_now}"
            )

        if nelem_chk <= 0:
            raise RuntimeError("After CDREAD, MAPDL element count is zero")

        log("    [MAPDL] remake BASE after CDREAD")
        try:
            make_component_from_nodes(mapdl, "BASE", base_nodes)
            log("    [MAPDL] BASE exact-id method done after CDREAD")
        except Exception as e:
            log(f"    [MAPDL] BASE exact-id method failed after CDREAD: {e}")
            make_component_by_z(mapdl, "BASE", base_zmin)

        mapdl.cmsel("S", "BASE")
        mapdl.run("*GET,NBASECHK,NODE,0,COUNT")
        nbase_chk = int(mapdl.parameters["NBASECHK"])
        mapdl.allsel()

        if nbase_chk <= 0:
            raise RuntimeError("BASE exists but selected node count is zero after CDREAD")

        log(f"    [MAPDL] BASE nodes after CDREAD = {nbase_chk}")
        mapdl.finish()
        log("    [MAPDL] CLEAR + CDREAD done")

    else:
        log("    [MAPDL] CDB workflow skipped (save_cdb = false)")

    log("    [MAPDL] SOLU start")
    mapdl.slashsolu()
    mapdl.antype("STATIC")

    mapdl.cmsel("S", "BASE")
    mapdl.run("*GET,NFIX,NODE,0,COUNT")
    nfix = int(mapdl.parameters["NFIX"])
    if nfix <= 0:
        raise RuntimeError("BASE component exists but selected node count is zero")

    log(f"      fixed base nodes = {nfix}")
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    batch = []
    count_force_nodes = 0

    for nid, fvec in nodal_forces.items():
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
    log(f"      force nodes={count_force_nodes}, resultant force vector={total_force_vec}")

    t0 = time.time()
    mapdl.solve()
    log(f"    [MAPDL] solve done in {time.time() - t0:.2f}s")
    mapdl.finish()

    log("    [MAPDL] POST1 start")
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
        raise RuntimeError("POST: nodal equivalent stress contains no finite values")

    if all_nodes.size == seqv.size:
        nodes_valid = all_nodes[valid]
        idx = int(np.argmax(seqv_valid))
        node_at_smax = int(nodes_valid[idx])
    else:
        node_at_smax = -1

    smax = float(np.max(seqv_valid))
    smin = float(np.min(seqv_valid))
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
        except Exception as e:
            log(f"    [MAPDL] save PNG failed: {e}")
            png_ok = False

    mapdl.finish()
    log("    [MAPDL] POST1 done")

    return {
        "Smax_MPa": smax,
        "Smin_MPa": smin,
        "node_at_Smax": node_at_smax,
        "feasible": feasible,
        "loaded_area_mm2": loaded_area,
        "pressure_MPa": pressure_MPa,
        "resultant_force_vector_N": total_force_vec.tolist(),
        "cdb_path": cdb_eval_path if outputs.get("save_cdb", True) else "",
        "png_path": png_path if png_ok else ""
    }


# ============================================================
# evaluate one design
# ============================================================
def evaluate_design(mapdl, out_root: str, cfg: dict, eval_idx: int, wall_thk: float):
    eval_dir = os.path.join(out_root, f"eval_{eval_idx:03d}")
    ensure_dir(eval_dir)

    geo = cfg["geometry_mm"]
    stress_limit = float(cfg["constraint"]["stress_limit_MPa"])

    pipe_inner_d = float(geo["pipe_inner_d"])
    pipe_outer_d = pipe_inner_d + 2.0 * wall_thk

    log("\n============================================================")
    log(f"[EVAL {eval_idx:03d}] wall_thk = {wall_thk:.6f} mm | outer_d = {pipe_outer_d:.6f} mm")
    log("============================================================")

    log("[1/3] Gmsh build + extract start")
    gm = gmsh_build_extract_flanged_pipe(eval_dir, cfg, wall_thk)
    log("[1/3] Gmsh build + extract done")

    log("[2/3] MAPDL build->CDB->CDREAD->solve->post start")
    res = solve_one_eval(mapdl, eval_idx, eval_dir, gm, cfg)
    log("[2/3] MAPDL build->CDB->CDREAD->solve->post done")

    vol_mm3 = flange_pipe_volume_mm3(cfg, wall_thk)

    rec = {
        "eval": eval_idx,
        "pipe_wall_thk_mm": wall_thk,
        "pipe_outer_d_mm": pipe_outer_d,
        "pipe_inner_d_mm": pipe_inner_d,
        "volume_mm3": vol_mm3,
        "Smax_MPa": res["Smax_MPa"],
        "Smin_MPa": res["Smin_MPa"],
        "stress_limit_MPa": stress_limit,
        "node_at_Smax": res["node_at_Smax"],
        "feasible": res["feasible"],
        "pressure_MPa": res["pressure_MPa"],
        "loaded_area_mm2": res["loaded_area_mm2"],
        "resultant_force_vector_N": res["resultant_force_vector_N"],
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
        f"thk={rec['pipe_wall_thk_mm']:.6f} mm"
    )

    return rec


# ============================================================
# cache + bisection
# ============================================================
def get_cached_or_eval(mapdl, out_root, cfg, wall_thk, state):
    key = round(float(wall_thk), 6)

    if key in state["cache"]:
        log(f"[CACHE] hit for wall_thk={wall_thk:.6f}")
        return state["cache"][key]

    if state["eval_count"] >= state["max_eval"]:
        raise RuntimeError(f"max_eval reached: {state['max_eval']}")

    rec = evaluate_design(
        mapdl=mapdl,
        out_root=out_root,
        cfg=cfg,
        eval_idx=state["eval_count"],
        wall_thk=float(wall_thk)
    )

    state["cache"][key] = rec
    state["history"].append(rec)
    state["eval_count"] += 1

    return rec


def bisection_min_wall_thk(mapdl, out_root, cfg, state):
    thk_min = float(cfg["design_var"]["pipe_wall_thk_mm"]["min"])
    thk_max = float(cfg["design_var"]["pipe_wall_thk_mm"]["max"])
    tol = float(cfg["optimizer"]["thk_bisect_tol_mm"])

    log("\n[SEARCH] pipe_wall_thk minimum feasible search")

    rec_hi = get_cached_or_eval(mapdl, out_root, cfg, thk_max, state)
    if not rec_hi["feasible"]:
        log(f"[SEARCH] thk_max={thk_max:.6f} mm에서도 불가능")
        return None

    rec_lo = get_cached_or_eval(mapdl, out_root, cfg, thk_min, state)
    if rec_lo["feasible"]:
        log(f"[SEARCH] thk_min={thk_min:.6f} mm already feasible")
        return rec_lo

    lo = thk_min
    hi = thk_max
    best = rec_hi

    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        rec_mid = get_cached_or_eval(mapdl, out_root, cfg, mid, state)

        if rec_mid["feasible"]:
            best = rec_mid
            hi = mid
        else:
            lo = mid

        log(f"[BISECTION] lo={lo:.6f}, hi={hi:.6f}")

    return best


# ============================================================
# main
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = read_json(os.path.join(base_dir, "problem.json"))

    out_root = os.path.join(base_dir, "outputs_FLANGED_PIPE_CASE1_NOCLI")
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

    try:
        best = bisection_min_wall_thk(mapdl, out_root, cfg, state)

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
        log(f"Best pipe_wall_thk = {best['pipe_wall_thk_mm']:.6f} mm")
        log(f"Best pipe_outer_d  = {best['pipe_outer_d_mm']:.6f} mm")
        log(f"Best Smax          = {best['Smax_MPa']:.6e} MPa")
    log(f"Evaluations        = {state['eval_count']}")
    log(f"Outputs            = {final_dir}")
    log("==============================")


if __name__ == "__main__":
    main()