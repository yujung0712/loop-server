import os
import json
import time
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


# ============================================================
# Gmsh build + extract
# ============================================================
def gmsh_build_extract(iter_dir: str, cfg: dict, cyl_d: float):
    geo = cfg["geometry_mm"]
    mesh_cfg = cfg["mesh_mm"]
    tags = cfg["physical_tags"]
    outputs = cfg["outputs"]

    L_total = float(geo["L_total"])
    L_sq = float(geo["L_sq"])
    sq_w = float(geo["sq_w"])
    sq_h = float(geo["sq_h"])
    lc = float(mesh_cfg["lc"])

    BASE = int(tags["BASE"])
    TOP = int(tags["TOP"])
    VOL = int(tags["VOL"])

    step_path = os.path.join(iter_dir, "model.step")
    msh_path = os.path.join(iter_dir, "mesh.msh")

    log("  [GMSH] initialize")
    gmsh.initialize()
    gmsh.model.add("TSB")
    occ = gmsh.model.occ

    log("  [GMSH] build geometry")
    box = occ.addBox(-sq_w / 2.0, -sq_h / 2.0, 0.0, sq_w, sq_h, L_sq)
    cyl = occ.addCylinder(0.0, 0.0, L_sq, 0.0, 0.0, L_total - L_sq, cyl_d / 2.0)

    occ.fuse([(3, box)], [(3, cyl)])
    occ.synchronize()

    log("  [GMSH] detect base/top surfaces")
    base_surfs = []
    top_surfs = []

    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin - 0.0) < 1e-6 and abs(zmax - 0.0) < 1e-6:
            base_surfs.append(tag)

        if abs(zmin - L_total) < 1e-6 and abs(zmax - L_total) < 1e-6:
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
# write local APDL mesh input file
# ============================================================
def write_mapdl_build_inp(work_dir: str, iter_idx: int, node_tags: np.ndarray, node_xyz: np.ndarray, tet_conn: np.ndarray):
    inp_path = os.path.join(work_dir, f"build_iter_{iter_idx:03d}.inp")

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
# MAPDL one-session build -> cdb -> cdread -> solve -> post
# ============================================================
def solve_one_iter(mapdl, iter_idx: int, iter_dir: str, gm: dict, cfg: dict):
    mat = cfg["material_MPa"]
    load = cfg["load"]
    cons = cfg["constraint"]
    outputs = cfg["outputs"]

    E = float(mat["E"])
    nu = float(mat["nu"])
    F_total = float(load["total_N"])
    dof = load["direction"].upper()
    sign = float(load.get("sign", 1.0))
    comp = cons["monitor_component"].upper()
    U_limit = float(cons["U_limit_mm"])

    node_tags = gm["node_tags"].astype(int)
    node_xyz = gm["node_xyz"]
    tet_conn = gm["tet_conn"].astype(int)
    base_nodes = gm["base_nodes"].astype(int)
    top_nodes = gm["top_nodes"].astype(int)

    mapdl_work_dir = mapdl.directory
    cdb_base_name = f"model_iter_{iter_idx:03d}"
    cdb_work_base = os.path.join(mapdl_work_dir, cdb_base_name)
    cdb_work_path_lower = os.path.join(mapdl_work_dir, f"{cdb_base_name}.cdb")
    cdb_work_path_upper = os.path.join(mapdl_work_dir, f"{cdb_base_name}.CDB")
    cdb_iter_path = os.path.join(iter_dir, "model.cdb")
    png_path = os.path.join(iter_dir, "deformed.png")

    build_inp_path = write_mapdl_build_inp(
        mapdl_work_dir,
        iter_idx,
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
            safe_copy(cdb_work_path_lower, cdb_iter_path)
        elif os.path.exists(cdb_work_path_upper):
            safe_copy(cdb_work_path_upper, cdb_iter_path)

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

    mapdl.cmsel("S", "TOP")
    tip_nodes = np.array(mapdl.mesh.nnum, dtype=int)

    if len(tip_nodes) == 0:
        raise RuntimeError("POST: TOP component selection returned zero nodes")

    disp_vals = []
    for n in tip_nodes:
        val = mapdl.get_value("NODE", int(n), "U", comp)
        disp_vals.append(float(val))

    disp_vals = np.array(disp_vals, dtype=float)

    umax = float(np.max(disp_vals))
    umin = float(np.min(disp_vals))
    uabs = max(abs(umax), abs(umin))
    feasible = bool(uabs <= U_limit)

    mapdl.allsel()

    png_ok = False
    if outputs.get("save_png", True):
        try:
            log("    [MAPDL] save PNG start")
            mapdl.post_processing.plot_nodal_displacement(
                component=comp,
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
        "Uabs_mm": uabs,
        "Umax_mm": umax,
        "Umin_mm": umin,
        "feasible": feasible,
        "n_top": n_top,
        "f_each": f_each,
        "cdb_path": cdb_iter_path if outputs.get("save_cdb", True) else "",
        "png_path": png_path if png_ok else ""
    }


# ============================================================
# main
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = read_json(os.path.join(base_dir, "problem.json"))

    out_root = os.path.join(base_dir, "outputs_TSB_FINAL_NOCLI")
    ensure_dir(out_root)

    dv = cfg["design_var"]
    opt = cfg["optimizer"]
    mapdl_cfg = cfg["mapdl"]

    lo = float(dv["min"])
    hi = float(dv["max"])
    max_iter = int(opt["max_iter"])
    tol_mm = float(opt["tol_mm"])

    run_location = mapdl_cfg["run_location"]
    jobname_prefix = mapdl_cfg["jobname_prefix"]
    override = bool(mapdl_cfg["override"])

    ensure_dir(run_location)

    log("[MAPDL] single launch start")
    mapdl = launch_mapdl(
        run_location=run_location,
        jobname=jobname_prefix,
        override=override
    )
    log(f"[MAPDL] launched: workdir = {mapdl.directory}")

    history = []
    best = None

    try:
        for it in range(max_iter):
            d_try = 0.5 * (lo + hi)
            iter_dir = os.path.join(out_root, f"iter_{it:03d}")
            ensure_dir(iter_dir)

            log("\n============================================================")
            log(f"[ITER {it:03d}] cyl_d = {d_try:.6f} mm")
            log("============================================================")

            log("[1/3] Gmsh build + extract start")
            gm = gmsh_build_extract(iter_dir, cfg, d_try)
            log("[1/3] Gmsh build + extract done")

            log("[2/3] MAPDL build->CDB->CDREAD->solve->post start")
            res = solve_one_iter(mapdl, it, iter_dir, gm, cfg)
            log("[2/3] MAPDL build->CDB->CDREAD->solve->post done")

            rec = {
                "iter": it,
                "cyl_d_mm": d_try,
                "Uabs_mm": res["Uabs_mm"],
                "Umax_mm": res["Umax_mm"],
                "Umin_mm": res["Umin_mm"],
                "U_limit_mm": cfg["constraint"]["U_limit_mm"],
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

            write_json(os.path.join(iter_dir, "result.json"), rec)
            history.append(rec)

            log(
                f"[RESULT] Uabs={res['Uabs_mm']:.6e} mm | "
                f"limit={cfg['constraint']['U_limit_mm']:.6e} mm | "
                f"feasible={res['feasible']}"
            )

            if res["feasible"]:
                best = rec
                hi = d_try
            else:
                lo = d_try

            if abs(hi - lo) <= tol_mm:
                log(f"[STOP] converged: hi-lo={hi - lo:.6e} mm <= tol={tol_mm:.6e} mm")
                break

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

    if best is None and history:
        best = history[-1]

    if best is not None:
        best_iter_dir = os.path.join(out_root, f"iter_{best['iter']:03d}")

        for fname in ["model.step", "mesh.msh", "model.cdb", "deformed.png", "result.json"]:
            safe_copy(os.path.join(best_iter_dir, fname), os.path.join(final_dir, fname))

    write_json(
        os.path.join(final_dir, "summary.json"),
        {
            "best": best,
            "history": history
        }
    )

    log("\n==============================")
    log("[DONE]")
    if best:
        log(f"Best cyl_d = {best['cyl_d_mm']:.6f} mm")
        log(f"Best Uabs  = {best['Uabs_mm']:.6e} mm")
    log(f"Outputs: {final_dir}")
    log("==============================")


if __name__ == "__main__":
    main()