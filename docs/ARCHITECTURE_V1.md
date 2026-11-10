# simws architecture v1: contract for the build team

Status: written by the lead on 10 Nov 2026. This file is the single source of truth for the v1 build. If code and
this file disagree, the file wins until the lead amends it (amendments are appended at the end, dated).

## 1. Goal

simws connects open-source simulation engines under one workspace. Before v1, every device was written inside an
engine module (for example `simws/tcad/hbt.py` is DEVSIM code), so a new device needed new code per engine, and
most results bypassed the hub. v1 adds:

1. an **engine-neutral description** of a problem (`simws.spec`, already written, do not change it);
2. **one translator and one capability sheet per engine** (`simws.engines`);
3. a **planner** that says which engines can run a problem and why the others cannot;
4. **one hub adapter per engine** (`device.<engine>`) that runs any supported problem;
5. **trust labels** on every result, **measured data stored like engine results**, one **compare** step for
   engine-vs-engine and engine-vs-measurement, a flat **catalogue**, and a small **study** layer.

Nothing existing is removed in v1. Legacy builders and adapters stay and are the regression reference.

## 2. Ground rules for every team member

- Repo: `/home/claude/simws` (git, branch `master`). **Do not commit.** The lead commits.
- Edit only the files listed as yours. Never edit `simws/spec/*`, `docs/ARCHITECTURE_V1.md`, existing `simws/tcad/*`,
  `simws/hub/*` (except files listed as yours), `simws/validation/*`, or another member's files. If you need a
  change outside your files, write it under "Requests to the lead" in your final report.
- Python 3.13, pydantic v2, numpy, pint (`simws.units`: one registry, never create another).
- DEVSIM works here only with this environment variable (set it in every shell that imports devsim):
  `export DEVSIM_MATH_LIBS="/usr/lib/x86_64-linux-gnu/liblapack.so.3:/usr/lib/x86_64-linux-gnu/libblas.so.3"`
- Run tests with: `cd /home/claude/simws && PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider -o addopts= <paths>`
  (the `-o addopts=` is required: `pytest.ini` otherwise ignores unit tests). The machine has 2 cores shared by
  the whole team: run only the tests you need, never the whole `tests/unit` suite.
- No `pip install`. Charon, gmsh, shapely, Quantum ESPRESSO, GPAW, Geant4 and Solcore are NOT available here.
- Project rules that apply to you: never invent data or reference values (every expected number in a test must be
  derived in the test from a formula, or come from the legacy code path run in the same test); tolerances are
  written before running and never loosened to make something pass; a failure is reported, not hidden.
- Engine imports (`devsim`) stay inside functions, never at module top level, so every module imports without
  the engine.
- Style: the existing code style (type hints, short docstrings that state units, line length <= 120).
- Final report: files written, test command and result (counts), anything you could not do, requests to the lead.

## 3. The description (`simws/spec`, owner: lead, frozen)

Read `simws/spec/schema.py` and `simws/spec/library.py` in full before writing anything. Summary:

- `Problem(device: DeviceSpec, models: ModelSpec, analysis: Equilibrium | DCSweep, numerics: Numerics)`;
  `problem.problem_id()` is a content hash; `problem.with_changes({"models.temperature_K": 77.0})` returns a
  validated copy.
- Units in the description: micrometres, cm^-3, V, K, cm^2/V/s, s, eV.
- Interval rule: a point c is inside `(lo, hi)` when `lo <= c < hi`; with `include_upper=True`, `lo <= c <= hi`;
  `None` is unbounded. Doping terms add. `DeviceSpec.doping_at(x, y)` and `mole_fraction_at(x, y)` are the
  reference meaning: a translator is correct when its engine sees those values at every mesh node.
- Library: `pn_diode_1d(...)` mirrors `simws.validation.engine_pairs._devsim_diode`; `npn_hbt_2d(...)` mirrors
  `simws.tcad.hbt.HBTCase` with default (non-cryogenic, `eb_refine=False`) settings and `simws.tcad.hbt.gummel`.

## 4. Engines (`simws/engines`)

### 4.1 `simws/engines/base.py` (owner: builder B)

```python
class EngineNotAvailable(RuntimeError): ...        # the engine or its executable is not installed here
class Unsupported(ValueError):                     # .reasons: List[str]
    def __init__(self, engine: str, reasons: List[str]): ...

class EngineResult(BaseModel):                     # extra="forbid"
    engine: str
    problem_id: str
    analysis: Literal["equilibrium", "dc_sweep"]
    sweep_contact: Optional[str] = None            # None for equilibrium
    sweep_V: List[float]                           # [0.0] for equilibrium
    currents: Dict[str, List[float]]               # contact name -> current INTO the device, one per sweep value
    current_unit: Literal["A/cm**2", "A/cm"]       # 1D: per cm^2 of cross-section; 2D: per cm of depth
    flags: List[str] = []
    diagnostics: Dict[str, Any] = {}               # JSON-serialisable only

class Capability(BaseModel):                       # extra="forbid"; every list holds the ALLOWED values
    engine: str
    env: str                                       # conda env name on the laptop
    description: str
    in_process: bool                               # True: run() works inside Python
    dimensions: List[int]
    max_regions: int
    materials: List[str]
    grading_elements: List[str]
    contact_kinds: List[str]
    statistics: List[str]
    mobility: List[str]
    velocity_saturation: List[bool]
    srh: List[str]
    auger: List[bool]
    band_gap_narrowing: List[str]
    incomplete_ionization: List[str]
    impact_ionization: List[str]
    analyses: List[str]
    tested_temperature_K: Tuple[float, float]      # range checked against something; outside = runs but flagged
    notes: str = ""
    def unsupported(self, problem: Problem) -> List[str]: ...
    def outside_tested_range(self, problem: Problem) -> List[str]: ...
```

`Capability.unsupported` returns one human-readable reason per violated field, each starting with the field name,
e.g. `"mobility: 'klaassen' not in ['constant']"`, `"dimensions: 2 not in [1]"`, `"max_regions: 2 > 1"`,
`"materials: 'sic_4h' not in ['silicon']"`, `"contact_kinds: 'schottky' not in ['ohmic']"`,
`"grading_elements: 'Ge' not in []"`, `"analyses: 'dc_sweep' not in [...]"`. Empty list = supported.
`outside_tested_range` returns `["temperature_K: 77 outside tested [250, 400]"]`-style strings (empty when inside,
bounds included).

### 4.2 `simws/engines/__init__.py` (owner: builder B)

```python
ENGINES: Dict[str, str] = {"devsim": "simws.engines.devsim_engine", "charon": "simws.engines.charon_engine"}
def names() -> List[str]                        # sorted
def module(name: str)                           # imports the engine module; KeyError with the known names if unknown
def capability(name: str) -> Capability         # module(name).CAPABILITY
def unsupported(name: str, problem) -> List[str]   # module(name).unsupported(problem)
```

Every engine module exposes exactly:

```python
CAPABILITY: Capability
def unsupported(problem) -> List[str]           # CAPABILITY.unsupported(problem) + engine-specific reasons
def available() -> bool                         # cheap: can the engine run on this machine? never raises
def run(problem, workdir: Optional[Path] = None) -> EngineResult
    # raises Unsupported (reasons), EngineNotAvailable, simws.hub.adapter.NotConverged
```

Out-of-process engines (Charon) additionally expose `prepare(problem, workdir) -> List[Path]` (input files written)
and `collect(problem, workdir) -> EngineResult`; their `run` is prepare + execute + collect.

### 4.3 `simws/engines/planner.py` (owner: builder B)

```python
class Plan(BaseModel):
    problem_id: str
    primary: Optional[str]                 # first supporting engine in `prefer` order, then registry (sorted) order
    checkers: List[str]                    # the other supporting engines, sorted
    refused: Dict[str, List[str]]          # engine -> reasons
    warnings: Dict[str, List[str]]         # supporting engine -> outside_tested_range strings (only non-empty)
    def summary(self) -> str               # short multi-line text for a person

def plan(problem, prefer: Sequence[str] = ("devsim",), engines: Optional[Sequence[str]] = None) -> Plan
```

A plan is about capability, not about what is installed on this machine.

### 4.4 `simws/engines/devsim_engine.py` (owner: builder A)

Translates a Problem to DEVSIM using the EXISTING physics (`simws.tcad.physics.Physics`, `setup_region`,
`contact_current`, `simws.tcad.solve.ramp_bias`). Do not re-implement any physics. Read `simws/tcad/hbt.py`,
`simws/tcad/mixed_mode.py::_mesh_2d`, `simws/validation/engine_pairs.py::_devsim_diode` and
`simws/adapters/tcad.py::HbtGummel._sweep` first: the translator must reproduce them.

CAPABILITY (state exactly this): engine "devsim", env "tcad", in_process True, dimensions [1, 2], max_regions 1,
materials ["silicon"], grading_elements ["Ge"], contact_kinds ["ohmic"], statistics ["boltzmann", "fermi_dirac"],
mobility ["constant", "masetti", "klaassen"], velocity_saturation [False, True], srh ["off", "constant",
"scharfetter"], auger [False, True], band_gap_narrowing ["off", "slotboom"], incomplete_ionization ["off",
"altermatt", "constant_energy"], impact_ionization ["off", "van_overstraeten"], analyses ["equilibrium",
"dc_sweep"], tested_temperature_K (250.0, 400.0).
Engine-specific `unsupported` reasons: 2D contacts on `x_min`/`x_max` (the legacy mesh has contact pads on top and
bottom only): `"contacts: 2D DEVSIM contacts must be on y_min or y_max ('<name>' is on <side>)"`; a region box that
does not start at 0 on every axis: `"regions: the region box must start at 0"`.

Translation rules (all coordinates micrometres x 1e-4 = cm; device name "d", mesh name "m", region name = the
spec's region name):

- Call `reset_devsim()` first (import `simws.tcad` before `devsim`; it installs the safe reset).
- Doping node expressions: for each term the factor per bounded axis is `step(c - lo)` for a lower bound,
  `step(hi - c)` for an included upper bound and `(1 - step(c - hi))` for an excluded upper bound (DEVSIM's
  `step(0)` is 1; verified). donors = sum of donor terms, acceptors = sum of acceptor terms (use "0" when there
  are none); net = `(donors) - (acceptors)`, total = `(donors) + (acceptors)`.
- Ge grading (one GradeTerm): `step(c - c0)*step(c1 - c)*(a + (b - a)*(c - c0)/(c1 - c0))` -> `Physics.ge`,
  `Physics.ge_mode` = term mode. More than one GradeTerm: Unsupported.
- 1D mesh: `add_1d_mesh_line` per mesh line (`pos`, `ps = spacing * scale`), tags on the first and last line for the
  `x_min` / `x_max` contacts (`add_1d_contact(..., material="metal")`), `add_1d_region(material="Si", ...)`.
- 2D mesh: the same construction as `_mesh_2d` (thin pad regions above and below carry the contacts) with the
  spacings multiplied by `scale`; contacts as in `simws.tcad.hbt.build` (`add_2d_contact(..., bloat=1e-10)`), a
  `span` of None meaning the full width.
- ModelSpec -> Physics:
  `T=temperature_K`; `fermi_dirac = statistics == "fermi_dirac"`; `mobility = "klaassen"` when klaassen else
  "masetti"; constant mobility -> `mu_n`, `mu_p`; `field_mobility = velocity_saturation`; `srh = srh != "off"`;
  constant lifetimes -> `tau_n`, `tau_p`; `auger`; `bgn = band_gap_narrowing == "slotboom"`;
  `impact = impact_ionization == "van_overstraeten"`; `incomplete_ionization = incomplete_ionization != "off"`,
  `ii_model = "altermatt"` or `"charon"` (constant_energy) with `ii_E_const`, `ii_N_crit`; `donor_species`,
  `acceptor_species`; `extended_precision`.
- `setup_region(device, region, load(material), net, total, phys, {contact: False, ...})` builds and solves
  equilibrium. Keep its returned dict in `diagnostics["used"]` (JSON-safe values only).
- Solver options from Numerics: `absolute_error`, `relative_error`, `maximum_iterations`. If `setup_region`
  returns `equilibrium_rel_tol`, use `max(numerics.relative_error, that)` and add the flag
  `"newton_rel_tol:<value>"` (legacy cryogenic behaviour).
- Klaassen mobility needs the outer iteration of the legacy code: pass
  `callback=lambda _v: relax_mobility("d", region, opts)` with `retry_callback=True` to every `ramp_bias`.
- Equilibrium analysis: after the build, record `contact_current` for every contact (`sweep_V = [0.0]`).
- DCSweep: ramp each `fixed_V` contact once (step 0.25 V); then for every value v, in order: each tracking
  contact to `v + offset_V` (its `step_V`), then the swept contact to v (`step_V`); then record `contact_current`
  for EVERY contact. `current_unit` is "A/cm**2" in 1D and "A/cm" in 2D. DEVSIM's contact current is positive into
  the device: no sign change.
- A Newton failure (`RuntimeError` whose text contains "converge" or "stalled", as in `adapters/tcad.py`) becomes
  `simws.hub.adapter.NotConverged` with the bias reached in the message.
- Also expose `build(problem) -> dict` (everything up to and including equilibrium; returns the `used` dict) so
  later analyses (AC, strikes) can start from it.

Acceptance (the checker tests these; tolerances fixed here):
- R1: `run(pn_diode_1d(V=[0.1, ..., 0.6]))` vs `_devsim_diode([0.1, ..., 0.6])`: anode current density within
  1e-6 relative at every bias.
- R2: `run(npn_hbt_2d(VBE=[0.5, 0.6, 0.7]))` vs `simws.tcad.hbt.gummel(HBTCase(), VBE=[0.5, 0.6, 0.7])`: |IC| and
  |IB| within 1e-6 relative (legacy currents are A per um of emitter length = A/cm x 1e-4, absolute values).
- R3: node values of NetDoping and of the Ge fraction in DEVSIM equal `doping_at` / `mole_fraction_at` evaluated at
  the node coordinates, to 1e-12 relative (HBT) .
- R4: equilibrium currents of both library devices below 1e-8 x the forward current at 0.6 V.

### 4.5 `simws/engines/charon_engine.py` (owner: builder B)

Charon cannot run here. Deliver the translator and its capability sheet so the laptop can run it.
Reuse `simws/validation/charon_xcheck.py` by import (`_material`, `_deck`, `MESHES`, the reader used by its
`compare` step); do not copy its code and do not change that file. Read it in full first.

CAPABILITY (only what the 0.9a cross-check actually exercised): engine "charon", env "tcad", in_process False,
dimensions [1], max_regions 1, materials ["silicon"], grading_elements [], contact_kinds ["ohmic"], statistics
["fermi_dirac"], mobility ["constant"], velocity_saturation [False], srh ["off", "constant"], auger [False],
band_gap_narrowing ["off"], incomplete_ionization ["constant_energy"], impact_ionization ["off"], analyses
["dc_sweep"], tested_temperature_K (77.0, 300.0).
Engine-specific `unsupported` reasons, checked from the description alone: the doping must be exactly one acceptor
term on `x < xj` and one donor term on `x >= xj` (an abrupt pn step, p side at `x_min`):
`"doping: Charon translator supports an abrupt two-term pn step only"`; `tau_n_s != tau_p_s`:
`"srh: Charon deck takes one lifetime for both carriers"`; sweep contact must be the `x_min` contact and no
tracking / fixed contacts: `"analysis: ..."`; device length and junction position must match a mesh in
`charon_xcheck.MESHES` (1.0 um long, junction at 0.5 um): `"mesh: no Charon mesh for this geometry"`.
`available()` is True only when `simws.config.path("toolchain.charon_exe")` exists.
`prepare` writes one deck per bias plus the run list, exactly as `charon_xcheck`'s `write` step does for one
temperature, into `workdir`; `run` raises `EngineNotAvailable` when `available()` is False (after `prepare`, so the
decks can be inspected); `collect` reads the Charon outputs with the existing reader and returns anode current
density in "A/cm**2", positive into the device at the anode.

## 5. Hub adapters (`simws/adapters/device.py`, owner: builder A)

One adapter class per engine, sharing a base class:

- names `device.devsim` (env "tcad", engine "devsim") and `device.charon` (env "tcad", engine "charon"); register
  both in `simws/adapters/__init__.py` (builder A is the only one who edits that file; add two lines).
- Manifest: `version="1.0"`, `engine_version="env-locked"`, `inputs=["problem"]`, `outputs=[]` (output names
  depend on the contact names), `native_conventions={"terminal_current": "into_device+"}`.
- `resolve(params)`: `params["problem"]` (a dict or a Problem) is validated and replaced by
  `Problem.model_validate(...).model_dump(mode="json")`, so every default is in the job key.
- `prepare`: `Unsupported` reasons -> `InvalidInput("unsupported by <engine>: " + "; ".join(reasons))`.
- `run`: `engines.module(engine).run(problem, ctx.staging)`; `EngineNotAvailable` -> let it raise as an
  `AdapterError` subclass or wrap so that the status is `engine_error` and the message says the engine is not
  installed.
- `collect`: one quantity per contact, name `dc.I.<contact>`, unit "A/m**2" (1D: A/cm^2 x 1e4) or "A/m" (2D: A/cm x
  100), `conventions={"terminal_current": "into_device+"}`; DC sweep: one axis
  `{"name": "V_<sweep contact>", "unit": "V", "values": sweep_V}`; equilibrium: scalars, no axis.
  `ctx.flags` gets the engine flags plus `"extrapolated:" + s` for each `outside_tested_range` string.
  `ctx.diagnostics`: `problem_id`, `engine`, `device` (name), `temperature_K`, and the engine diagnostics.
- `self_check`: Kirchhoff, the sum over contacts of the current at each sweep value is below
  `1e-4 x max|I| + floor` at that value, where floor is 1e-8 A/m (2D) or 1e-4 A/m**2 (1D); for equilibrium every
  |I| is below the floor.

Acceptance: `Hub(tmp, mode="inprocess").submit("device.devsim", params={"problem": pn_diode_1d(V=[0.3, 0.5]).model_dump(mode="json")})`
returns status "ok", quantities `dc.I.anode` and `dc.I.cathode` in A/m**2 that sum to ~0, and a second identical
submit is a cache hit; an unsupported problem returns status "invalid_input" with the reasons in the message.

## 6. Trust, measured data, compare, catalogue, study (owner: builder C)

Read `simws/hub/schema.py`, `jobs.py` (especially `Hub.sketch`, which shows how to commit a result that no adapter
produced), `store.py`, `graph.py` and `gatekeeper.py` first.

### 6.1 `simws/hub/trust.py`

```python
LEVELS = ("sketch", "unverified", "verified", "validated", "measured")
class Trust(BaseModel): level: str; in_range: bool; reasons: List[str]
def assess(res: ResultEnvelope, comparisons: Sequence[dict] = ()) -> Trust
def assess_stored(hub, result_id: str) -> Trust      # assess(hub.result(rid), compare.records_for(hub, rid))
```

Rules, first match wins:
1. `res.status != "ok"` -> level "unverified", reason `"status:<status>"`.
2. "measured" in `res.flags` -> "measured".
3. "sketch" in `res.flags` -> "sketch".
4. any comparison with `kind == "measurement"` and `passed` true -> "validated" (reason names the other result).
5. any comparison with `kind == "engine_pair"` and `passed` true, OR any value in `res.verification` that is a
   dict with `kind == "code"` and `passed` true -> "verified".
6. otherwise "unverified". A `res.verification` entry of `kind == "solution"` (mesh error quantified) does not
   raise the level; add the reason `"numerical error quantified"`.
A failed or inconclusive comparison never raises the level; it adds a reason (`"failed measurement comparison
with <id>"`). `in_range` is False when any flag of the result or of any of its quantities starts with
"extrapolated" or "cryo_unvalidated"; each such flag is added to the reasons.
`comparisons` are dicts as produced by `Comparison.model_dump()` (6.3); only those that involve this result count
(`a == res.id or b == res.id`); a measurement comparison validates the non-measured side only.

### 6.2 `simws/measured.py`

```python
def ingest(hub, name: str, axis: Optional[Tuple[str, Sequence[float], str]],
           columns: Dict[str, Tuple[Sequence[float], str]], meta: Dict[str, Any],
           sigma: Optional[Dict[str, Sequence[float]]] = None, label: str = "validation") -> ResultEnvelope
def load_csv(path, x_col: int = 0, y_col: int = 1, skip_header: int = 0, delimiter: str = ",") -> Tuple[List[float], List[float]]
```

`ingest` stores measured data as an ordinary result: adapter name `measured.<name>`, engine "measurement",
engine_version "n/a", env_lock "measured", status "ok", `flags` ["measured"] (plus "digitized" when
`meta.get("digitized")`), each Quantity named by its column with the axis (`axis = (name, values, unit)`; None for
scalars), `flags=["measured"]`, `label=label` ("calibration" or "validation"), `source=meta["source"]`,
`uncertainty` sigma when given (absolute, same unit), `provenance={"measured": meta}`. `meta["source"]` (a citation
string) is required: without it raise `ValueError` (project rule: no data without a source). The job key's params
hold `meta` and the SHA-256 of the canonical data, so the same data ingested twice is the same job (cache hit:
return the stored envelope). Commit through `gatekeeper.apply` + `store.finalize` + `graph.commit_result` under
`store.lock()`, as `Hub.sketch` does, and add a graph node of type "Dataset" named `name` linked to the result
with a "produced" edge. Lengths of axis and columns must agree, else `ValueError`.

### 6.3 `simws/compare.py`

```python
class Comparison(BaseModel):
    kind: Literal["engine_pair", "measurement"]
    a: str = ""; b: str = ""                     # result ids ("" when comparing bare quantities)
    quantity_a: str; quantity_b: str
    metric: Literal["relative", "ratio"]
    tolerance: float
    worst: Optional[float]                       # None when no point could be compared
    worst_at: Optional[float]                    # axis value of the worst point (None for scalars)
    n_points: int
    window: Optional[Tuple[float, float]] = None
    passed: bool
    inconclusive: bool
    note: str = ""

def compare(qa: Quantity, qb: Quantity, tolerance: float, metric: str = "relative",
            window: Optional[Tuple[float, float]] = None, min_points: int = 1,
            interpolation: str = "linear") -> Comparison
def compare_results(hub, rid_a: str, name_a: str, rid_b: str, name_b: Optional[str] = None, **kw) -> Comparison
def record(hub, cmp: Comparison) -> None
def records_for(hub, result_id: str) -> List[dict]
```

`compare`: b is the reference. b is converted to a's unit (different dimension -> `ValueError`). Both scalars: one
point. Otherwise both must have exactly one axis of the same dimension (b's axis converted to a's axis unit); the
points are a's axis values that lie inside b's axis range (bounds included, never extrapolate) and inside `window`
(a range of axis values in a's axis unit) when given; b is interpolated onto them: "linear", or "log" = linear in
ln|y| (requires every b value non-zero and of one sign, else `ValueError`). metric "relative": `|a/b - 1|`;
"ratio": `max(a/b, b/a)` (a point where a and b differ in sign or either is zero counts as infinity). `worst` is
the maximum over the points; `passed = n_points >= min_points and worst <= tolerance`;
`inconclusive = n_points < min_points` (then `passed` is False). `kind` is "measurement" when "measured" is in
`qa.flags` or `qb.flags`, else "engine_pair".
`record` stores the comparison in the graph: an edge `a -> b` of type "validated_against" (measurement) or
"checked_by" (engine_pair) whose data is `{"comparisons": [...]}`; a new comparison for the same pair of results
and the same (quantity_a, quantity_b) replaces the old one, others are kept; also append a graph event
"compared". `record` needs both ids non-empty. `records_for` returns every stored comparison dict in which the
result is `a` or `b`.

### 6.4 `simws/catalogue.py`

```python
def rows(hub, adapter: Optional[str] = None, status: Optional[str] = "ok", quantity: Optional[str] = None,
         flag: Optional[str] = None, fidelity: Optional[str] = None,
         temperature: Optional[Tuple[float, float]] = None, trust: Optional[str] = None) -> List[dict]
def table(rows: List[dict]) -> str
```

A flat, searchable list of committed results (this replaces graph traversal for everyday use). One row per
result: `result_id, job_id, adapter, engine, status, fidelity, finished, flags, quantities (names),
temperature_K, trust, in_range`. `adapter` is an fnmatch pattern ("device.*"); `quantity` and `flag` are
substring matches (any quantity name / any flag); `temperature` is an inclusive range; `trust` a level.
`temperature_K` comes from the job key: `params["problem"]["models"]["temperature_K"]`, else `params["T"]` (a
canonical quantity dict with "value" in K, or a number), else None (rows with None never match a temperature
filter). Rows are ordered by `finished`, newest first.

### 6.5 `simws/study.py`

```python
class Study:
    def __init__(self, hub, problem: Problem, engines: Optional[Sequence[str]] = None, fidelity: str = "screening")
    def sweep(self, over: Dict[str, Sequence]) -> "StudyResult"       # full factorial over dotted paths
class StudyResult:
    points: List[dict]      # {"changes": {...}, "engine": str, "status": str, "result_id": str | None,
                            #  "reasons": [...], "trust": str | None, "in_range": bool | None}
    def quantity(self, name: str, engine: Optional[str] = None) -> List[Tuple[dict, Quantity]]
    def table(self, name: str) -> str
    def compare_engines(self, name: str, a: str, b: str, tolerance: float, record: bool = True, **kw) -> List[Comparison]
```

`engines=None` means `[plan(problem).primary]`. For every combination of the `over` values (in the order of the
dict, last key varying fastest) and every engine: build `problem.with_changes(changes)`; if the engine does not
support it the point has status "refused" and the reasons, and nothing is submitted; otherwise
`hub.submit(f"device.{engine}", params={"problem": p.model_dump(mode="json")}, fidelity=fidelity)`. An empty
`over` runs the problem once per engine. `trust` / `in_range` come from `trust.assess_stored`.
`compare_engines` compares, for each combination where both engines have status "ok", quantity `name` of a against
b (reference) with `compare_results`, and records it when `record` is true.

## 7. Tests (owner: checker)

All new tests go in `tests/v1/`, one file per module: `test_spec.py`, `test_engines_base_planner.py`,
`test_devsim_engine.py`, `test_charon_engine.py`, `test_device_adapter.py`, `test_trust.py`, `test_measured.py`,
`test_compare.py`, `test_catalogue.py`, `test_study.py`. The checker writes them from this file and the frozen
`simws/spec` only, without reading the builders' implementations, and marks DEVSIM runs that take longer than
about 30 s with `@pytest.mark.slow`. Use `Hub(tmp_path, mode="inprocess")` for hub tests.

## 8. Amendments

### A1 (10 Nov 2026, lead, after the builders' and the checker's reports)

1. **R3 wording.** The description is in micrometres and DEVSIM stores nodes in cm, so converting a node back to
   micrometres can land one floating-point step across a doping bound. R3 is evaluated with each node kept on the
   side of every bound that the engine sees (the checker's method). The 1e-12 tolerance on the values is unchanged.
2. **R4 for the HBT is a recorded FAIL.** Measured: the largest zero-bias terminal current is 2.89e-12 A/cm, which
   is 1.35e-8 of the 0.6 V emitter current, against the 1e-8 fixed before the run. The legacy build gives the
   identical current (bit for bit), so this is not a translation error: the tolerance was the lead's and was tighter
   than the equilibrium solve (relative error 1e-8) can deliver. The tolerance is not changed and the test stays in
   the suite as an expected failure. **R4b (post hoc):** zero-bias terminal currents below the project's existing
   floor, `HbtGummel.I_FLOOR_A_PER_M` = 1e-8 A/m, used since 0.9a. R4 for the diode passed as written.
3. **`equilibrium_rel_tol` (4.4).** `setup_region` always returns it (1e-8 in the normal case). It is applied, and
   flagged `newton_rel_tol:<value>`, only when it exceeds 1e-8 (the cold-device fallback); otherwise the Numerics
   value is used unchanged. The original wording would have loosened the diode from 1e-12 to 1e-8.
4. **Invalid descriptions (5).** A description that fails validation raises `InvalidInput` from `Hub.submit` itself
   (it is checked in `resolve`, before a job exists). A valid description that an engine cannot run returns a
   committed result with status `invalid_input` and the reasons.
5. **Extra refusal reasons, accepted.** DEVSIM: more than one grading term; mesh lines that do not span the region.
   Charon: sweep values must be distinct multiples of 1 mV (deck names carry millivolts).
6. **Charon `collect` (4.5).** It returns the cathode current as minus the anode current, flagged
   `<cathode>_current_from_kirchhoff`, and takes the sign of the anode current from the bias, flagged
   `current_sign_from_bias`. For Charon results the adapter's Kirchhoff check is therefore not an independent check.
   For temperatures below 250 K `prepare` writes its own continuation chain (300 K down through 200, 150, 110, 95,
   85, 77 K). None of the Charon execute / collect path has run: it must be tested on the laptop.
7. **Citations.** `device.charon` carries no citation key until a checked Charon reference is added to
   `simws.hub.citations`.
8. **Declared output (5).** The project rule "an adapter must declare outputs" (`tests/connections/test_manifests.py`)
   stands. Each `device.<engine>` adapter declares one output that every problem has, `dc.kirchhoff_residual`
   (dimensionless: the largest |sum of terminal currents| / (largest |terminal current| + noise floor) over the
   sweep), next to the per-contact `dc.I.<contact>` quantities, whose names depend on the device.
9. **Fidelity.** The `device.<engine>` adapters are screening fidelity in v1: they report no mesh error bars, so the
   gatekeeper refuses them at paper fidelity (it requires an uncertainty on every quantity). Paper-fidelity results
   still come from the legacy adapters until mesh refinement is added to the device adapters.
10. **Trust rule 4 tightened (6.1).** A failed measurement comparison (not merely inconclusive) blocks "validated",
    however many other measurement comparisons passed; the level falls back to rule 5 or 6 and the reasons say
    how many failed and passed. Written with the SiC held-out protocols in mind: matching the fit temperatures and
    missing the held-out ones is not validation.

### A2 (10 Nov 2026, lead, after an outside review of v1)

Four defects in sections 5 and 6 that let a trust label say more than the evidence. Where A2 and the earlier text
disagree, A2 wins.

1. **Interpolation default (6.3).** `compare(..., interpolation="auto")` is the new default. "auto" means "log" when
   every reference (b) value is non-zero and of one sign, and "linear" otherwise. "linear" and "log" can still be
   forced ("log" still raises `ValueError` on a zero or a sign change). `Comparison` gains the field
   `interpolation: Literal["linear", "log", "none"]`: the method actually used, "none" for two scalars.
   Reason: currents are exponential in bias, and a straight line between two points of an exponential is far above
   it. Reference case (must be a test): `a = I0*(exp(V/Vt) - 1)` on V = 0.10, 0.15, ..., 0.60 V and the same formula
   as b on V = 0.10, 0.20, ..., 0.60 V, with I0 = 1e-14 A and Vt = 0.02585 V: metric "relative" gives a worst value
   of 0.7175 with "linear" (at the 0.15 V midpoint; for a pure exponential without the "- 1" it would be
   |1/cosh(0.05/Vt) - 1| = 0.7168 at every midpoint) and below 0.01 with "log" and with "auto". The expected value
   in the test is computed from the formula with straight-line interpolation written out in the test.
2. **Calibration is not validation (6.1, 6.3).** `Comparison` gains
   `measurement_label: Optional[Literal["calibration", "validation"]] = None`. For `kind == "measurement"` it is the
   `label` of the measured quantity (b's if b is measured, else a's); a measured quantity whose label is "none"
   gives None; for `kind == "engine_pair"` it is None.
   Trust rule 4 becomes: "validated" needs at least one passed measurement comparison with
   `measurement_label == "validation"`, and no failed (not merely inconclusive) measurement comparison of ANY label
   (A1.10 still holds). A passed measurement comparison whose label is "calibration" or None never raises the
   level; it adds the reason `"matches calibration data <other id> (fitted to it: not validation)"` for
   "calibration" and `"matches unlabelled measured data <other id> (not counted as validation)"` for None.
3. **A second engine must be a different engine (6.1, 6.3).** `Comparison` gains `engine_a: str = ""`,
   `engine_b: str = ""` and `independent: bool = False`. `compare()` on bare quantities leaves them at the defaults.
   `compare_results` fills the engines from the stored job keys (`hub.store.job(rid).key.engine`) and sets
   `independent`: for `kind == "engine_pair"` it is True only when both engine names are non-empty and different
   AND neither result carries the flag "sketch"; for `kind == "measurement"` it is True.
   Trust rule 5 becomes: an engine-pair comparison raises the level to "verified" only when it passed AND
   `independent` is true. A passed engine-pair comparison that is not independent never raises the level; it adds
   the reason `"same-engine or sketch comparison with <other id> (consistency only: not verification)"`.
   Comparison dicts stored before A2 (no `independent` key) count as not independent.
   `Study.compare_engines(name, a, a, ...)` therefore still runs and records, and verifies nothing.
4. **Kirchhoff residual that is true by construction (5).** When the engine result carries a flag ending in
   `_current_from_kirchhoff` (Charon, A1.6), the `dc.kirchhoff_residual` quantity gets the quantity flag
   `"not_independent"`. Its value stays as computed (0.0): the gatekeeper refuses non-finite values, so it cannot be
   NaN. For engines that measure every terminal current the quantity has no such flag.
5. **Clarifications after the checker's report on A2.** "Every reference value" in A2.1 means all of b's values, not
   only those in the overlap or window. With two scalars the recorded method is "none" whatever was asked for. A
   stored measurement comparison without a `measurement_label` key (written before A2) counts as unlabelled: it
   does not validate. `independent` is set only by `compare_results`; `compare()` on bare quantities always leaves
   it False, measured or not.
