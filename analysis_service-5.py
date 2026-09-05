"""
Lumexa Analysis Service — split out from the main backend specifically to
isolate the heaviest computation (Gmsh volume meshing + CalculiX solid-tet
FEM solving) into its own process/container.

WHY THIS EXISTS: the combined backend (cadquery/OCP + gmsh + calculix +
vtk/numba, all in one process) was hitting Render's free-tier 512MB RAM
ceiling and getting OOM-killed mid-request — confirmed via a Termux curl
test that showed a connection dying with 0 bytes received, immediately
followed by Render auto-restarting the container in the logs. Splitting the
single heaviest step into its own service means its memory use doesn't
compound on top of whatever the generation service already has resident
from CadQuery/OCP script execution.

This service does NOT need cadquery/OCP at all — only trimesh (for reading
the uploaded mesh), numpy/scipy, gmsh, and the CalculiX binary. That's a much
lighter dependency footprint than the combined service carried.

Everything in this file (MATERIALS dict, FEM functions) is copied verbatim
from main.py — same logic, same tested behavior, just running in an
isolated process now.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import trimesh
import numpy as np
import tempfile, os, math, json, subprocess, threading

try:
    import gmsh
    GMSH = True
except ImportError:
    GMSH = False

CALCULIX_ERROR = None
try:
    r = subprocess.run(["ccx", "-v"], capture_output=True, timeout=5)
    CALCULIX = True
except Exception as _e:
    CALCULIX = False
    CALCULIX_ERROR = f"{type(_e).__name__}: {_e}"


def _json_safe(obj):
    """Same numpy-type sanitizer as main.py — see that file's docstring for
    the full explanation of why this is needed (numpy.bool_/integer/floating
    are not natively JSON serializable)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_json_safe(content))


app = FastAPI(title="Lumexa Analysis Service", version="1.0.0",
              default_response_class=SafeJSONResponse)
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


MATERIALS = {
    "aluminum_6061":{"name":"Aluminum 6061-T6","density":2.70,
        "yield_strength_mpa":276,"ultimate_strength_mpa":310,
        "youngs_modulus_gpa":68.9,"poissons_ratio":0.33,
        "thermal_expansion_per_c":23.6e-6,"thermal_conductivity":167,
        "max_service_temp_c":150,"fatigue_limit_mpa":96,
        "fracture_toughness_mpa_sqrtm":29.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":1.2e-4,
        "paris_C":1.5e-10,"paris_m":3.58,"shear_modulus_gpa":26.0,
        "hardness_brinell":95,"endurance_ratio":0.4,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":3.5,"machinability":0.85},
    "aluminum_7075":{"name":"Aluminum 7075-T6","density":2.81,
        "yield_strength_mpa":503,"ultimate_strength_mpa":572,
        "youngs_modulus_gpa":71.7,"poissons_ratio":0.33,
        "thermal_expansion_per_c":23.4e-6,"thermal_conductivity":130,
        "max_service_temp_c":120,"fatigue_limit_mpa":159,
        "fracture_toughness_mpa_sqrtm":24.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":1.0e-4,
        "paris_C":1.2e-10,"paris_m":3.5,"shear_modulus_gpa":26.9,
        "hardness_brinell":150,"endurance_ratio":0.4,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":5.5,"machinability":0.70},
    "alsi10mg_slm":{"name":"AlSi10Mg SLM (3D Printed)","density":2.68,
        "yield_strength_mpa":230,"ultimate_strength_mpa":345,
        "youngs_modulus_gpa":70.0,"poissons_ratio":0.33,
        "thermal_expansion_per_c":21.0e-6,"thermal_conductivity":130,
        "max_service_temp_c":120,"fatigue_limit_mpa":70,
        "fracture_toughness_mpa_sqrtm":20.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":2.0e-4,
        "paris_C":2.0e-10,"paris_m":3.8,"shear_modulus_gpa":26.3,
        "hardness_brinell":80,"endurance_ratio":0.35,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.095,
        "min_wall_mm":0.8,"min_fillet_mm":0.4,
        "cost_per_kg_usd":45.0,"machinability":0.60},
    "titanium_6al4v":{"name":"Titanium Ti-6Al-4V","density":4.43,
        "yield_strength_mpa":880,"ultimate_strength_mpa":950,
        "youngs_modulus_gpa":114.0,"poissons_ratio":0.34,
        "thermal_expansion_per_c":8.6e-6,"thermal_conductivity":7.2,
        "max_service_temp_c":315,"fatigue_limit_mpa":510,
        "fracture_toughness_mpa_sqrtm":75.0,"creep_exponent_n":4.0,
        "creep_activation_energy":250000,"creep_A_constant":5.0e-6,
        "paris_C":5.0e-11,"paris_m":3.2,"shear_modulus_gpa":44.0,
        "hardness_brinell":334,"endurance_ratio":0.55,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.075,
        "min_wall_mm":0.8,"min_fillet_mm":0.3,
        "cost_per_kg_usd":85.0,"machinability":0.30},
    "steel_4340":{"name":"Steel AISI 4340","density":7.85,
        "yield_strength_mpa":470,"ultimate_strength_mpa":745,
        "youngs_modulus_gpa":205.0,"poissons_ratio":0.29,
        "thermal_expansion_per_c":12.3e-6,"thermal_conductivity":44.5,
        "max_service_temp_c":370,"fatigue_limit_mpa":380,
        "fracture_toughness_mpa_sqrtm":50.0,"creep_exponent_n":5.5,
        "creep_activation_energy":280000,"creep_A_constant":6.0e-7,
        "paris_C":6.0e-12,"paris_m":3.0,"shear_modulus_gpa":80.0,
        "hardness_brinell":217,"endurance_ratio":0.5,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.5,"min_fillet_mm":1.0,
        "cost_per_kg_usd":2.5,"machinability":0.55},
    "inconel_718":{"name":"Inconel 718","density":8.19,
        "yield_strength_mpa":1034,"ultimate_strength_mpa":1241,
        "youngs_modulus_gpa":200.0,"poissons_ratio":0.29,
        "thermal_expansion_per_c":13.0e-6,"thermal_conductivity":11.4,
        "max_service_temp_c":650,"fatigue_limit_mpa":550,
        "fracture_toughness_mpa_sqrtm":100.0,"creep_exponent_n":4.5,
        "creep_activation_energy":300000,"creep_A_constant":3.0e-7,
        "paris_C":3.0e-12,"paris_m":3.0,"shear_modulus_gpa":77.0,
        "hardness_brinell":310,"endurance_ratio":0.45,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.080,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":65.0,"machinability":0.20},
    "carbon_fiber_ud":{"name":"Carbon Fiber CFRP (UD)","density":1.60,
        "yield_strength_mpa":600,"ultimate_strength_mpa":1500,
        "youngs_modulus_gpa":135.0,"poissons_ratio":0.28,
        "thermal_expansion_per_c":2.1e-6,"thermal_conductivity":5.0,
        "max_service_temp_c":180,"fatigue_limit_mpa":450,
        "fracture_toughness_mpa_sqrtm":35.0,"creep_exponent_n":3.0,
        "creep_activation_energy":200000,"creep_A_constant":1.0e-8,
        "paris_C":1.0e-11,"paris_m":3.0,"shear_modulus_gpa":5.0,
        "hardness_brinell":0,"endurance_ratio":0.6,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.070,
        "min_wall_mm":0.5,"min_fillet_mm":0.3,
        # Composite-specific
        "E1_gpa":135.0,"E2_gpa":10.0,"G12_gpa":5.0,"nu12":0.28,
        "Xt_mpa":1500,"Xc_mpa":1200,"Yt_mpa":50,"Yc_mpa":250,"S12_mpa":70,
        "cost_per_kg_usd":80.0,"machinability":0.15},
    "pla_plastic":{"name":"PLA Plastic (FDM)","density":1.24,
        "yield_strength_mpa":50,"ultimate_strength_mpa":65,
        "youngs_modulus_gpa":3.5,"poissons_ratio":0.36,
        "thermal_expansion_per_c":68e-6,"thermal_conductivity":0.13,
        "max_service_temp_c":60,"fatigue_limit_mpa":20,
        "fracture_toughness_mpa_sqrtm":3.5,"creep_exponent_n":3.0,
        "creep_activation_energy":80000,"creep_A_constant":1.0e-3,
        "paris_C":1.0e-8,"paris_m":4.0,"shear_modulus_gpa":1.3,
        "hardness_brinell":0,"endurance_ratio":0.35,
        "Sut_at_1000":0.80,"fatigue_slope_b":-0.110,
        "min_wall_mm":1.2,"min_fillet_mm":0.8,
        "cost_per_kg_usd":25.0,"machinability":0.90},
    "petg_plastic":{"name":"PETG Plastic (FDM)","density":1.27,
        "yield_strength_mpa":53,"ultimate_strength_mpa":50,
        "youngs_modulus_gpa":2.1,"poissons_ratio":0.38,
        "thermal_expansion_per_c":60e-6,"thermal_conductivity":0.20,
        "max_service_temp_c":80,"fatigue_limit_mpa":18,
        "fracture_toughness_mpa_sqrtm":4.0,"creep_exponent_n":3.0,
        "creep_activation_energy":80000,"creep_A_constant":1.2e-3,
        "paris_C":1.2e-8,"paris_m":4.0,"shear_modulus_gpa":0.76,
        "hardness_brinell":0,"endurance_ratio":0.32,
        "Sut_at_1000":0.78,"fatigue_slope_b":-0.115,
        "min_wall_mm":1.2,"min_fillet_mm":0.8,
        "cost_per_kg_usd":28.0,"machinability":0.88},
    "stainless_316l":{"name":"Stainless Steel 316L","density":7.98,
        "yield_strength_mpa":170,"ultimate_strength_mpa":485,
        "youngs_modulus_gpa":193.0,"poissons_ratio":0.28,
        "thermal_expansion_per_c":16.0e-6,"thermal_conductivity":16.3,
        "max_service_temp_c":870,"fatigue_limit_mpa":240,
        "fracture_toughness_mpa_sqrtm":200.0,"creep_exponent_n":5.0,
        "creep_activation_energy":270000,"creep_A_constant":4.0e-7,
        "paris_C":4.0e-12,"paris_m":3.1,"shear_modulus_gpa":74.0,
        "hardness_brinell":217,"endurance_ratio":0.5,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.5,"min_fillet_mm":1.0,
        "cost_per_kg_usd":8.0,"machinability":0.45},
    "magnesium_az31":{"name":"Magnesium AZ31B","density":1.77,
        "yield_strength_mpa":200,"ultimate_strength_mpa":260,
        "youngs_modulus_gpa":45.0,"poissons_ratio":0.35,
        "thermal_expansion_per_c":26.0e-6,"thermal_conductivity":96,
        "max_service_temp_c":120,"fatigue_limit_mpa":90,
        "fracture_toughness_mpa_sqrtm":18.0,"creep_exponent_n":4.5,
        "creep_activation_energy":135000,"creep_A_constant":3.0e-4,
        "paris_C":2.0e-10,"paris_m":3.5,"shear_modulus_gpa":17.0,
        "hardness_brinell":73,"endurance_ratio":0.35,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.095,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":4.0,"machinability":0.80},
}

_GMSH_LOCK = threading.Lock()

def _parse_ccx_dat(dat_path):
    """
    Shared .dat parser for both the solid and shell CalculiX paths. Anchors on
    *NODE PRINT / *EL PRINT section headers rather than guessing a line's meaning
    from its column count, which can silently misparse unrelated numeric output.
    Returns (max_von_mises_mpa, max_displacement_mm).
    """
    max_vm = 0.0
    max_disp = 0.0
    section = None  # "disp" | "stress" | None

    if not os.path.exists(dat_path):
        return max_vm, max_disp

    with open(dat_path) as f:
        content = f.read()

    for line in content.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if not stripped:
            continue
        if "DISP" in upper and ("NODE" in upper or upper.startswith("D")):
            section = "disp"; continue
        if upper.startswith("STRESSES") or "S.MISES" in upper or (
                section != "disp" and set("S11 S22 S33 S12 S13 S23".split()) & set(upper.split())):
            section = "stress"; continue
        if upper.startswith(("SUMMARY", "MAXIMUM", "MINIMUM", "*")):
            continue

        parts = stripped.split()
        if section == "disp" and len(parts) >= 4:
            try:
                ux, uy, uz = float(parts[1]), float(parts[2]), float(parts[3])
                d = math.sqrt(ux**2 + uy**2 + uz**2)
                if d > max_disp: max_disp = d
            except (ValueError, IndexError):
                pass
        elif section == "stress" and len(parts) >= 7:
            try:
                s11=float(parts[1]); s22=float(parts[2])
                s33=float(parts[3]); s12=float(parts[4])
                s13=float(parts[5]); s23=float(parts[6])
                vm = math.sqrt(0.5*(
                    (s11-s22)**2+(s22-s33)**2+(s33-s11)**2+
                    6*(s12**2+s13**2+s23**2)
                ))
                if vm > max_vm: max_vm = vm
            except (ValueError, IndexError):
                pass

    return max_vm, max_disp


def _set_gmsh_option_safe(name, value):
    """Try an option name across Gmsh API vintages; ignore if this version doesn't have it."""
    try:
        gmsh.option.setNumber(name, value)
    except Exception:
        pass


def _tetrahedralize_with_gmsh(mesh, mesh_size_factor=0.08, max_tets=80000):
    """
    Volume-mesh a watertight trimesh surface into linear (4-node) tetrahedra
    using Gmsh's STL-reconstruction workflow: merge STL -> classify surfaces ->
    reconstruct a real geometric model from the facets -> add a volume -> generate
    a 3D mesh. This is the standard documented approach (Gmsh tutorial t13) for
    turning an arbitrary triangulated surface into a solid mesh without a CAD
    kernel behind it.

    Returns (node_coords, tets):
      node_coords: list of (x,y,z), index i corresponds to CalculiX node id i+1
      tets: list of (n1,n2,n3,n4) 1-based CalculiX node ids, one tuple per element

    Raises RuntimeError on any failure (non-manifold input, degenerate geometry,
    meshing failure, mesh too large) so the caller falls back to the shell path.

    IMPORTANT: this has not been executed in the development sandbox this was
    written in — gmsh isn't installed there and it has no network to install it.
    It follows Gmsh's documented STL-reconstruction API closely, but treat it as
    untested until it's actually run once on a deployment where both gmsh and
    ccx are available (this codebase already gates on that via the GMSH/CALCULIX
    flags), and confirm the shell fallback still engages cleanly if it raises.
    """
    stl_path = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
    try:
        mesh.export(stl_path)

        with _GMSH_LOCK:
            gmsh.initialize()
            try:
                _set_gmsh_option_safe("General.Terminal", 0)
                _set_gmsh_option_safe("General.Verbosity", 0)
                gmsh.model.add("lumexa_part")
                gmsh.merge(stl_path)

                angle_deg, curve_angle_deg = 40.0, 180.0
                gmsh.model.mesh.classifySurfaces(
                    angle_deg * math.pi / 180.0,
                    True,   # includeBoundary
                    False,  # forceParametrizablePatches
                    curve_angle_deg * math.pi / 180.0,
                )
                gmsh.model.mesh.createGeometry()

                surfaces = gmsh.model.getEntities(2)
                if not surfaces:
                    raise RuntimeError("Gmsh found no reconstructable surfaces in this mesh "
                                        "(often means the input isn't watertight/manifold).")
                loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
                gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()

                # Target element size relative to part size so small and large parts
                # both get a sane element count instead of one fixed absolute size.
                diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
                target = max(diag * mesh_size_factor, 0.1)
                for name in ("Mesh.MeshSizeMin", "Mesh.CharacteristicLengthMin"):
                    _set_gmsh_option_safe(name, target * 0.3)
                for name in ("Mesh.MeshSizeMax", "Mesh.CharacteristicLengthMax"):
                    _set_gmsh_option_safe(name, target)

                gmsh.model.mesh.generate(3)

                node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
                if len(node_tags) == 0:
                    raise RuntimeError("Gmsh produced zero nodes for this part.")

                tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
                coords = np.asarray(node_coords_flat, dtype=float).reshape(-1, 3)

                elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
                tets = []
                for etype, enodes in zip(elem_types, elem_node_tags):
                    if etype != 4:  # 4 = linear 4-node tetrahedron in Gmsh's element numbering
                        continue
                    flat = np.asarray(enodes, dtype=int)
                    for k in range(0, len(flat), 4):
                        n1, n2, n3, n4 = flat[k:k+4]
                        tets.append((
                            tag_to_idx[int(n1)] + 1, tag_to_idx[int(n2)] + 1,
                            tag_to_idx[int(n3)] + 1, tag_to_idx[int(n4)] + 1,
                        ))

                if not tets:
                    raise RuntimeError("Gmsh generated zero tetrahedral (C3D4) elements.")
                if len(tets) > max_tets:
                    raise RuntimeError(f"Tet mesh too large ({len(tets)} elements > {max_tets} cap) "
                                        f"for a timely solve; falling back to shell FEM.")

                return [tuple(c) for c in coords], tets
            finally:
                gmsh.finalize()
    finally:
        if os.path.exists(stl_path):
            try: os.unlink(stl_path)
            except: pass


def _run_calculix_solid_tet(mesh, mat_key, force_n=1000, force_dir="z"):
    """
    Real solid FEM: Gmsh-tetrahedralizes the part into C3D4 linear tetrahedra,
    then runs CalculiX on the actual solid volume — not a shell approximation.
    Returns (result, diag): result is None (never raises) if tetrahedralization
    or the solve fails, so the caller can fall back to the shell path without
    the request failing; diag always explains what actually happened at
    whichever stage it stopped, instead of a bare None that looked identical
    for "not available", "crashed", and "solved but zero stress".
    """
    if not GMSH:
        return None, "gmsh not available"

    try:
        node_coords, tets = _tetrahedralize_with_gmsh(mesh)
    except Exception as e:
        return None, f"gmsh tetrahedralization failed: {type(e).__name__}: {e}"

    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    E = mat["youngs_modulus_gpa"] * 1000  # MPa
    nu = mat["poissons_ratio"]
    rho = mat["density"] * 1e-9  # tonne/mm³

    try:
        coords_arr = np.asarray(node_coords)
        mins = coords_arr.min(axis=0)
        maxs = coords_arr.max(axis=0)

        inp = ["*HEADING", "Lumexa solid tetrahedral FEM (Gmsh + CalculiX)"]

        inp.append("*NODE,NSET=NALL")
        for i, (x, y, z) in enumerate(node_coords):
            inp.append(f"{i+1},{x:.6f},{y:.6f},{z:.6f}")

        inp.append("*ELEMENT,TYPE=C3D4,ELSET=EALL")
        for i, (n1, n2, n3, n4) in enumerate(tets):
            inp.append(f"{i+1},{n1},{n2},{n3},{n4}")

        inp.append("*MATERIAL,NAME=MAT")
        inp.append("*ELASTIC")
        inp.append(f"{E},{nu}")
        inp.append("*DENSITY")
        inp.append(f"{rho}")
        inp.append("*SOLID SECTION,ELSET=EALL,MATERIAL=MAT")

        axis_map = {"z": 2, "x": 0, "y": 1}
        ax = axis_map.get(force_dir, 2)
        span = maxs[ax] - mins[ax]

        fixed_nodes = [i+1 for i, c in enumerate(node_coords) if c[ax] < mins[ax] + span*0.05]
        if not fixed_nodes:
            fixed_nodes = [1, 2, 3]

        loaded_nodes = [i+1 for i, c in enumerate(node_coords) if c[ax] > maxs[ax] - span*0.05]
        if not loaded_nodes:
            loaded_nodes = [len(node_coords)]

        max_bc_nodes = max(50, min(3000, len(node_coords) // 10))
        if len(fixed_nodes) > max_bc_nodes:
            idx = np.linspace(0, len(fixed_nodes)-1, max_bc_nodes).astype(int)
            fixed_nodes = [fixed_nodes[i] for i in idx]
        if len(loaded_nodes) > max_bc_nodes:
            idx = np.linspace(0, len(loaded_nodes)-1, max_bc_nodes).astype(int)
            loaded_nodes = [loaded_nodes[i] for i in idx]

        # C3D4 solid nodes have 3 translational DOFs only (no rotation) — unlike
        # the S3 shell path, fixing DOFs 1-6 here would reference DOFs the
        # element doesn't have. *BOUNDARY is valid here, outside any *STEP,
        # as a permanent constraint — but *CLOAD is a history (step-specific)
        # keyword and MUST be inside *STEP...*END STEP, or ccx fatally
        # rejects the whole deck before solving anything (exit code 201).
        inp.append("*BOUNDARY")
        for n in fixed_nodes:
            inp.append(f"{n},1,3,0")

        force_per_node = force_n / max(len(loaded_nodes), 1)
        dof_map = {"x":1, "y":2, "z":3}
        dof = dof_map.get(force_dir, 3)

        inp.append("*STEP")
        inp.append("*STATIC")
        inp.append("*CLOAD")
        for n in loaded_nodes:
            inp.append(f"{n},{dof},{force_per_node:.4f}")
        inp.append("*NODE PRINT,NSET=NALL")
        inp.append("U")
        inp.append("*EL PRINT,ELSET=EALL")
        inp.append("S")
        inp.append("*END STEP")

        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False, mode="w") as f:
            f.write("\n".join(inp))
            inp_path = f.name
        job_name = inp_path.replace(".inp", "")

        try:
            proc = subprocess.run(["ccx", "-i", job_name], capture_output=True, text=True,
                                   timeout=300, cwd=os.path.dirname(inp_path))
        except subprocess.TimeoutExpired:
            return None, f"ccx solve timed out after 300s ({len(tets)} elements)"

        dat_path = job_name + ".dat"
        dat_existed = os.path.exists(dat_path)
        max_vm, max_disp = _parse_ccx_dat(dat_path)

        for ext in [".inp",".dat",".frd",".cvg",".sta",".12d"]:
            p = job_name + ext
            if os.path.exists(p):
                try: os.unlink(p)
                except: pass

        if max_vm > 0:
            Sy = mat["yield_strength_mpa"]
            return {
                "method": "calculix_solid_tet_fem",
                "solver_note": "Real solid FEM: part volume-meshed into C3D4 linear "
                                "tetrahedra via Gmsh, solved with CalculiX. This is a "
                                "true solid analysis, not a shell/thin-wall approximation.",
                "von_mises_mpa": round(max_vm, 3),
                "safety_factor": round(Sy/max(max_vm,0.001), 3),
                "status": "PASS" if Sy/max(max_vm,0.001) >= 2.0 else "FAIL",
                "max_displacement_mm": round(max_disp, 6),
                "elements": len(tets),
                "nodes": len(node_coords),
                "constrained_nodes": len(fixed_nodes),
                "loaded_nodes": len(loaded_nodes),
                "calculix_available": True,
            }, None
        # ccx ran without raising, but produced no usable stress result — the
        # exact case that used to be indistinguishable from "not available".
        diag = (f"ccx exit code {proc.returncode}, .dat file "
                f"{'existed but had no parseable stress section' if dat_existed else 'was never created'}"
                f"; stdout: {(proc.stdout or '(empty)')[-400:]}"
                f"; stderr: {(proc.stderr or '(empty)')[-400:]}")
        return None, diag
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _run_calculix_shell(mesh, mat_key, force_n=1000, force_dir="z"):
    """
    Real FEM using CalculiX (ccx), built on S3 (triangular) shell elements
    generated directly from the input surface mesh with an estimated uniform
    thickness. Used when Gmsh isn't available or solid tetrahedralization fails
    on a given part (e.g. non-watertight input) — a real solver run, just not a
    solid-tetrahedral one. Best suited to thin-walled parts. Returns (result,
    diag) — see _run_calculix_solid_tet's docstring for why diag matters.
    """
    if not CALCULIX:
        return None, "ccx not available"

    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    E = mat["youngs_modulus_gpa"] * 1000  # MPa
    nu = mat["poissons_ratio"]
    rho = mat["density"] * 1e-9  # tonne/mm³

    try:
        verts = mesh.vertices
        faces = mesh.faces
        n_verts = len(verts)
        bounds = mesh.bounds

        inp = ["*HEADING", "Lumexa shell FEM (fallback path)"]

        inp.append("*NODE,NSET=NALL")
        for i, v in enumerate(verts):
            inp.append(f"{i+1},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}")

        inp.append("*ELEMENT,TYPE=S3,ELSET=EALL")
        for i, f in enumerate(faces):
            inp.append(f"{i+1},{f[0]+1},{f[1]+1},{f[2]+1}")

        inp.append("*MATERIAL,NAME=MAT")
        inp.append("*ELASTIC")
        inp.append(f"{E},{nu}")
        inp.append("*DENSITY")
        inp.append(f"{rho}")

        avg_thickness = float(min(mesh.bounding_box.extents)) * 0.1
        avg_thickness = max(avg_thickness, 1.0)
        inp.append("*SHELL SECTION,ELSET=EALL,MATERIAL=MAT")
        inp.append(f"{avg_thickness:.3f}")

        axis_map = {"z": 2, "x": 0, "y": 1}
        ax = axis_map.get(force_dir, 2)
        fixed_nodes = []
        for i, v in enumerate(verts):
            if v[ax] < bounds[0][ax] + (bounds[1][ax]-bounds[0][ax])*0.05:
                fixed_nodes.append(i+1)
        if not fixed_nodes:
            fixed_nodes = [1, 2, 3]

        max_bc_nodes = max(50, min(2000, n_verts // 10))
        if len(fixed_nodes) > max_bc_nodes:
            idx = np.linspace(0, len(fixed_nodes) - 1, max_bc_nodes).astype(int)
            fixed_nodes = [fixed_nodes[i] for i in idx]

        # *BOUNDARY is valid outside any *STEP as a permanent constraint — but
        # *CLOAD is step-specific and MUST be inside *STEP...*END STEP, or ccx
        # fatally rejects the whole deck before solving anything (exit 201).
        inp.append("*BOUNDARY")
        for n in fixed_nodes:
            inp.append(f"{n},1,6,0")

        loaded_nodes = []
        for i, v in enumerate(verts):
            if v[ax] > bounds[1][ax] - (bounds[1][ax]-bounds[0][ax])*0.05:
                loaded_nodes.append(i+1)
        if not loaded_nodes:
            loaded_nodes = [len(verts)]

        if len(loaded_nodes) > max_bc_nodes:
            idx = np.linspace(0, len(loaded_nodes) - 1, max_bc_nodes).astype(int)
            loaded_nodes = [loaded_nodes[i] for i in idx]

        force_per_node = force_n / max(len(loaded_nodes), 1)
        dof_map = {"x":1, "y":2, "z":3}
        dof = dof_map.get(force_dir, 3)

        inp.append("*STEP")
        inp.append("*STATIC")
        inp.append("*CLOAD")
        for n in loaded_nodes:
            inp.append(f"{n},{dof},{force_per_node:.4f}")
        inp.append("*NODE PRINT,NSET=NALL")
        inp.append("U")
        inp.append("*EL PRINT,ELSET=EALL")
        inp.append("S")
        inp.append("*END STEP")

        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False, mode="w") as f:
            f.write("\n".join(inp))
            inp_path = f.name
        job_name = inp_path.replace(".inp", "")

        try:
            proc = subprocess.run(["ccx", "-i", job_name], capture_output=True, text=True,
                                   timeout=300, cwd=os.path.dirname(inp_path))
        except subprocess.TimeoutExpired:
            return None, f"ccx solve timed out after 300s ({len(faces)} elements)"

        dat_path = job_name + ".dat"
        dat_existed = os.path.exists(dat_path)
        max_vm, max_disp = _parse_ccx_dat(dat_path)

        for ext in [".inp",".dat",".frd",".cvg",".sta",".12d"]:
            p = job_name + ext
            if os.path.exists(p):
                try: os.unlink(p)
                except: pass

        if max_vm > 0:
            Sy = mat["yield_strength_mpa"]
            return {
                "method": "calculix_shell_fem",
                "solver_note": "S3 shell-element FEM on the input surface mesh with an "
                                "estimated uniform thickness — a real solver run, not a "
                                "solid-tetrahedral analysis. Best suited to thin-walled parts.",
                "von_mises_mpa": round(max_vm, 3),
                "safety_factor": round(Sy/max(max_vm,0.001), 3),
                "status": "PASS" if Sy/max(max_vm,0.001) >= 2.0 else "FAIL",
                "max_displacement_mm": round(max_disp, 6),
                "elements": len(faces),
                "nodes": n_verts,
                "constrained_nodes": len(fixed_nodes),
                "loaded_nodes": len(loaded_nodes),
                "calculix_available": True,
            }, None
        diag = (f"ccx exit code {proc.returncode}, .dat file "
                f"{'existed but had no parseable stress section' if dat_existed else 'was never created'}"
                f"; stdout: {(proc.stdout or '(empty)')[-400:]}"
                f"; stderr: {(proc.stderr or '(empty)')[-400:]}")
        return None, diag
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_calculix_fem(mesh, mat_key, force_n=1000, force_dir="z"):
    """
    Real FEM entry point. Tries, in order:
      1. Solid tetrahedral FEM (C3D4 elements, Gmsh-volume-meshed) — the
         accurate path for solid/chunky parts.
      2. Shell FEM (S3 elements on the surface mesh) — used when Gmsh is
         unavailable or tetrahedralization fails (e.g. non-watertight input),
         and reasonable in its own right for genuinely thin-walled parts.
      3. None — signals the caller to use the analytical multi-section fallback.
    Both real paths are honestly labeled in their return dict's "method" and
    "solver_note" fields; nothing here reports itself as more rigorous than
    what it actually did. Returns (result, diag): diag collects what actually
    happened on every path attempted, so a None result is always explained.
    """
    if not CALCULIX:
        return None, f"ccx not available" + (f" ({CALCULIX_ERROR})" if CALCULIX_ERROR else "")

    diags = {}
    if GMSH:
        result, diag = _run_calculix_solid_tet(mesh, mat_key, force_n, force_dir)
        if result is not None:
            return result, None
        diags["solid_tet"] = diag
    else:
        diags["solid_tet"] = "gmsh not available"

    result, diag = _run_calculix_shell(mesh, mat_key, force_n, force_dir)
    if result is not None:
        return result, None
    diags["shell"] = diag
    return None, diags

# ═══════════════════════════════════════════════════════════════════


@app.get("/")
def home():
    return {
        "status": "Lumexa Analysis Service v1.0",
        "capabilities": {"calculix_available": CALCULIX, "gmsh_available": GMSH,
                          "solid_tet_fem_available": bool(CALCULIX and GMSH)},
        "note": "Internal service — called by the main Lumexa backend, not meant "
                "for direct end-user use. Runs Gmsh + CalculiX solid-tet FEM in "
                "isolation so its memory use doesn't compound with the generation "
                "service's CadQuery/OCP footprint.",
    }


def _repair_mesh_for_meshing(mesh):
    """
    Clean up common tessellation artifacts — near-duplicate vertices, duplicate
    or degenerate (near-zero-area) faces, inconsistent normals — before handing
    the mesh to Gmsh. These are frequent byproducts of CadQuery/OCC boolean
    union at a seam (e.g. make_bent_bracket's leg1.union(leg2)) and are the
    most common real cause of Gmsh's "overlapping facets" rejection seen live
    on folded-bracket geometry. Pure mesh cleanup — doesn't change the actual
    part shape, only its triangulated representation. Never raises: repair
    failing just means Gmsh sees the original mesh and may reject it as before,
    not that the whole request fails.
    """
    try:
        before = len(mesh.faces)
        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        after = len(mesh.faces)
        return mesh, {"faces_before": before, "faces_after": after}
    except Exception as e:
        return mesh, {"repair_failed": f"{type(e).__name__}: {e}"}

@app.post("/run-fem")
async def run_fem(
    mesh_file: UploadFile = File(...),
    material: str = Form("aluminum_6061"),
    force_n: float = Form(1000.0),
    force_dir: str = Form("z"),
):
    """
    Run real FEM (solid tetrahedral via Gmsh+CalculiX, or shell-element
    fallback) on an uploaded mesh. Returns the exact same result shape
    run_calculix_fem always returned in the combined backend — this is a
    drop-in remote replacement, not a redesigned API.
    """
    contents = await mesh_file.read()
    fn = mesh_file.filename or "part.stl"
    suffix = os.path.splitext(fn)[1] or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        t.write(contents)
        tmp_path = t.name

    try:
        mesh = trimesh.load(tmp_path)
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        mesh, repair_info = _repair_mesh_for_meshing(mesh)

        result, diag = run_calculix_fem(mesh, material, force_n, force_dir)
        if result is None:
            return {"fem_result": None,
                    "note": "Neither solid-tet nor shell FEM produced a result "
                            "(CalculiX unavailable, or the solve failed) — the "
                            "caller should fall back to its own analytical path.",
                    "diagnostic": diag,
                    "mesh_repair": repair_info}
        return {"fem_result": result}
    except Exception as e:
        raise HTTPException(500, f"FEM analysis failed: {type(e).__name__}: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass
