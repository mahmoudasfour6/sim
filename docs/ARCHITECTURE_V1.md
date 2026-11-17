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

### A3 (10 Nov 2026, lead, before the first engine pair is run through the hub)

1. **Magnitude floor in `compare` (6.3).** `compare(..., floor: Optional[float] = None)`, also accepted by
   `compare_results` and `Study.compare_engines` (they pass it through). `floor` is a magnitude in a's unit and must
   be > 0 (`ValueError` otherwise). A point is compared only when the reference value there, |b| after conversion to
   a's unit and after interpolation onto a's axis, is strictly greater than `floor`; the other points are left out
   exactly like points outside the window: they do not count in `n_points`, they cannot be the worst point, and
   they do not make the comparison fail. Two scalars with |b| <= floor give `n_points == 0`, `worst is None`,
   inconclusive. `Comparison` gains `floor: Optional[float] = None` (the value used) and `n_below_floor: int = 0`
   (how many points inside the range and window were left out by the floor). With `floor=None` nothing changes.
   Reason: this is the rule of the 0.9a Charon cross-check (currents are compared only where |J| > 1e-6 A/cm^2):
   below a solver's noise floor a relative difference means nothing.
2. **Pilot gate, fixed before the run.** `tools/v1_pilot_hub.py` runs the Charon-compatible pn diode of
   `tools/v1_charon_check.py` on `device.devsim` and `device.charon` through the hub and compares `dc.I.anode`
   (DEVSIM against Charon as reference) with metric "relative", tolerance 0.02 and floor 1e-2 A/m**2
   (= 1e-6 A/cm^2): the 0.9a gate, unchanged. Declared before the hub run; the same pair outside the hub gave a
   worst difference of 6.8e-4 on 10 Nov, so this run is a test of the hub path, not a new physics result.

### A4 (15 Nov 2026, lead): mesh error bars for the device adapters

Lifts the limit recorded in A1.9. The rule is the one the legacy HBT adapter has used since 0.9a
(`simws.adapters.tcad.HbtGummel._mesh_uncertainty`); nothing in the legacy code changes, and it is the reference.

1. **When.** A `device.<engine>` job runs a mesh study when `job.fidelity == "paper"` or the job parameter
   `mesh_study` is true (`hub.submit("device.devsim", params={"problem": ..., "mesh_study": True})`). Otherwise
   nothing changes (one run, no uncertainty, A1.9 still describes the result).
2. **Which engines.** `Capability` gains `mesh_study: bool = False`: True for DEVSIM, False for Charon (its meshes
   are fixed files). A mesh study on an engine without it is refused before anything runs: status `invalid_input`,
   message containing `"mesh_study: <engine> cannot refine its mesh"`. So `device.charon` cannot be paper fidelity.
3. **Meshes.** Three runs of the same problem with `device.mesh.scale` = s0, s0/r, s0/r^2, r = sqrt(2), where s0 is
   the scale in the description (coarse to fine; the same ratio and number of levels as the legacy adapter). The
   reported values, the sweep axis and the engine flags are those of the finest run. A run that does not converge
   on any level makes the job `not_converged`.
4. **Error bar per point.** For every contact current and every sweep point, with |I| on the three meshes
   (coarse to fine) as f3, f2, f1:
   - `rec = simws.hub.verification.solution_from_values([0, 1, 2], [f3, f2, f1], ratio=r)`;
   - the point is "good" when `rec["sigma_num"] == 0.0`, or when `rec["sigma_num"]` is not None and the observed
     order p satisfies 0.5 <= p <= 4.0 and the asymptotic ratio satisfies 0.5 < ratio < 2.0; then
     sigma = `rec["sigma_num"]`;
   - otherwise sigma = 1.25 * |f1 - f2| / (r^0.5 - 1) / `simws.hub.uncertainty.GCI_COVERAGE_K` (the conservative
     bound: an order no better than 0.5 assumed);
   - exception, new in A4: when |f1| is below the adapter's noise floor (1e-8 A/m in 2D, 1e-4 A/m^2 in 1D), the
     three values are solver noise: sigma = that floor, the point counts as neither good nor conservative, and the
     record marks it `"below_floor": true`.
   Each `dc.I.<contact>` quantity carries `uncertainty.kind == "sigma"`, sigma in the quantity's unit and of the
   value's shape (a float for equilibrium), and `uncertainty.components == {"numerical": sigma}`.
   `dc.kirchhoff_residual` is computed on each of the three meshes; its value is the finest mesh's and its sigma
   is (max - min over the three meshes) / (2 * sqrt(3)) (a uniform interval), also as component "numerical".
5. **Record.** `result.verification["mesh"]` =
   `{"kind": "solution", "ratio": r, "levels": 3, "mesh_scales": [s0, s0/r, s0/r^2], "quantities": {name: [per
   point ...]}, "passed": bool}` where each per-point entry has `axis_value` (the sweep value; None for
   equilibrium), `values_coarse_to_fine` (|I| in the quantity's unit), `order`, `gci_fine`, `asymptotic_ratio`,
   `conservative_bound` (bool), `below_floor` (bool) and `rel_sigma` (sigma / |f1|, None when f1 is 0).
   `passed` is True when no point used the conservative bound. When it is False the result gets the flag
   `"mesh_not_asymptotic: conservative mesh bound used for some points"`. Every mesh-study result gets the flag
   `"budget:numerical_only"`: the error bar covers the mesh error and nothing else (no parameter or model-form
   part), and no model-form number is borrowed for these quantities.
6. **Manifest.** `device.devsim`: version "1.1" and
   `verification={"mesh": "paper fidelity: GCI per sweep point over 3 meshes (spacings x1, 1/sqrt2, 1/2)"}`, so the
   gatekeeper requires the record at paper fidelity. `device.charon`: version "1.1", no verification entry.
   (The version is part of the job key: results stored under 1.0 stay in the store, and a new submit runs again.)
7. **Acceptance R5 (tolerances fixed here, before any run).** Problem `npn_hbt_2d(VBE=[0.6, 0.7])`.
   New path: `device.devsim` with `mesh_study: True`, screening fidelity. Legacy path:
   `devsim.hbt_gummel` with `params={"VBE": [0.6, 0.7], "mesh_study": True, "case": {"eb_refine": False}}`
   (the explicit `eb_refine: False` keeps the legacy adapter on the same three meshes). Then
   |dc.I.collector|, |dc.I.base|, |dc.I.emitter| equal hbt.IC, hbt.IB, |hbt.IE| within 1e-6 relative at both
   biases; the sigmas agree within 1e-3 relative; `verification["mesh"]["passed"]` is the same on both paths; and
   the `conservative_bound` value of every point is the same (legacy record: `quantities["hbt.IC"]` etc.).
   **R6:** the same problem at `fidelity="paper"` on `device.devsim` returns status "ok" (the gatekeeper accepts
   it), and at paper fidelity on `device.charon` a Charon-compatible diode returns `invalid_input` with the A4.2
   message.
8. **Clarifications after the checker's report on A4.** (a) The study runs its three meshes with the exact scales
   s0, s0/r, s0/r^2. A separate hub job at a recorded scale goes through the 12-significant-digit rounding of job
   keys, so for s0/r (an irrational number) it can differ from the study's middle run by about 1e-9 relative; the
   coarse and fine scales are exact. This is why the Kirchhoff-residual sigma, a difference of cancellations, is
   only reproducible to about 1e-3 from outside. (b) A below-floor point records `order`, `gci_fine` and
   `asymptotic_ratio` as None. (c) `verification["mesh"]["quantities"]` holds the `dc.I.<contact>` entries only.
   (d) A non-finite `order` (finest two values identical) is stored as None by the result envelope. (e) The legacy
   record's `passed` also counts `hbt.beta`, which the new path does not have; R5 compares `passed` as written and
   the per-point `conservative_bound` values are the stronger check. (f) R5 result on 15 Nov (build machine):
   values identical, sigmas within 1.8e-14 relative, `passed` False on both paths: without the legacy adapter's
   `eb_refine` mesh the HBT is not in the asymptotic range at mesh scale 1, so most points carry the conservative
   bound. The library device has no `eb_refine` variant yet.

### A5 (16 Nov 2026, lead): refined HBT mesh and cold-temperature HBT on the new path

0. **Owner's instruction, 16 Nov:** build the refined HBT mesh, the cold-temperature path, AC analysis, and new
   engines. Project rule 6 freezes new engines while plumbing is above 30 % of logged hours (last figure 61 %); the
   lead pointed this out and the owner chose to go ahead with new engines. Recorded here as the dated amendment
   that project rule 5 asks for. The engines themselves are specified in a later amendment.
1. **Library.** `npn_hbt_2d(..., eb_refine=False, cryo=False)`. `eb_refine=True` uses the mesh lines of the legacy
   `HBTCase(eb_refine=True)` (spacing graded to 0.05 nm at the emitter-base junction and 0.2 nm at the
   base-collector junction). `cryo=True` selects the legacy `HBTCase(cryo=True)` physics: Fermi-Dirac statistics,
   Klaassen mobility, Altermatt incomplete ionization. The defaults leave every earlier description (and its
   problem id) unchanged.
2. **Acceptance R7 (refined mesh; tolerances fixed here).**
   R7a: for mesh scale 1 the DEVSIM node coordinates (x and y arrays of the region) built from
   `npn_hbt_2d(eb_refine=True)` are exactly equal (`numpy.array_equal`) to those of the legacy
   `simws.tcad.hbt.build(HBTCase(eb_refine=True))`.
   R7b (slow): `engine.run(npn_hbt_2d(eb_refine=True, VBE=[0.6]))` against
   `simws.tcad.hbt.gummel(HBTCase(eb_refine=True), VBE=[0.6])`: |IC| and |IB| within 1e-6 relative.
   R7c (very slow, for the overnight queue, not part of an installer): R5 repeated with `eb_refine=True` on both
   paths (legacy `case: {"eb_refine": True}`), same criteria as R5.
3. **Acceptance R8 (cold temperature; tolerances fixed here; slow).** For T = 250 K with VBE = [0.65, 0.70], T = 150 K
   with VBE = [0.85, 0.90] and T = 77 K with VBE = [1.00, 1.05]: `device.devsim` on
   `npn_hbt_2d(T=T, cryo=True, VBE=...)` against the legacy adapter
   `devsim.hbt_gummel` with `params={"VBE": [...], "T": T, "cryo": True}`, both through the hub at screening
   fidelity. At every bias the legacy result keeps (it drops points below its 1e-8 A/m floor; at least one point
   must remain per temperature), |dc.I.collector|, |dc.I.base| and |dc.I.emitter| equal hbt.IC, hbt.IB and |hbt.IE|
   within 1e-6 relative. Flags: below 250 K the new result carries a flag starting with
   `"extrapolated:temperature_K"`; and whenever the legacy result carries a flag starting with `"newton_rel_tol:"`,
   so does the new one.
   What this shows: the new path reproduces the legacy cold-device results. It does not validate them: below
   250 K they are checked against closed forms and Charon only, never against a measured device.
4. **After the checker's report (16 Nov).** R7a, R7b and R8 passed as written: the refined mesh is identical node
   for node (24,521 nodes), and the currents agree with the legacy path to 1.1e-16 relative or better at 250, 150
   and 77 K. The legacy adapter emitted no `newton_rel_tol` flag at these points, so that flag rule was not
   exercised. R7c was not run. Added on the checker's remark: below 250 K the DEVSIM engine result also carries
   the legacy warning flag starting with `"cryo_unvalidated:"` (the trust rules already read it as out of range).

### A6 (16 Nov 2026, lead): small-signal (AC) analysis on the new path

The legacy AC code (`simws.tcad.small_signal`, adapter `devsim.hbt_ac`) is written for the HBT. A6 adds a general
two-port small-signal analysis to the description; the legacy code is unchanged and is the reference.

1. **Description (`simws.spec`, additive; `SPEC_VERSION` and every existing problem id stay as they are).**

   ```python
   class Bias(_M):
       contact: str
       value_V: float
       step_V: float = Field(default=0.05, gt=0)

   class SmallSignal(_M):
       kind: Literal["small_signal"] = "small_signal"
       bias: List[Bias]                    # the DC operating point, ramped in this order; other contacts stay at 0 V
       ports: Tuple[str, str]              # (port 1, port 2) contacts; every other contact is AC ground
       frequencies_Hz: List[float]         # at least one, all > 0, strictly increasing
   ```

   Validation: every contact named in `bias` and `ports` exists; a contact appears at most once in `bias`; the two
   ports differ. `Analysis = Equilibrium | DCSweep | SmallSignal`.
2. **Engine result.** `EngineResult.analysis` may be "small_signal"; then `sweep_contact` is None, `sweep_V` is
   `[0.0]` and `currents` holds the DC operating-point current into each contact (one value each). New fields,
   empty for the other analyses: `frequencies_Hz: List[float]`, `y_real` and `y_imag`:
   `Dict[str, List[float]]` with keys "11", "12", "21", "22" (one value per frequency), and
   `y_unit: Optional[Literal["S/cm**2", "S/cm"]]` (1D: per cm^2 of cross-section; 2D: per cm of depth).
   Sign: y_ij = (small-signal current INTO the device at port i) / (small-signal voltage at port j), all other
   contacts AC-shorted.
3. **Two-port figures, engine-neutral (`simws/analysis/two_port.py`, pure numpy, no engine import).**
   With complex y11, y12, y21, y22 at a frequency f:
   `h21 = y21 / y11`;
   `mason_u = |y21 - y12|^2 / (4 (Re y11 Re y22 - Re y12 Re y21))`, and `inf` when that denominator is <= 0;
   `f_t_single_pole(f, |h21|, beta0) = beta0 * f / sqrt((beta0 / |h21|)^2 - 1)`, None unless `beta0 > |h21|`;
   `f_max(f, U) = f * sqrt(U)`, None unless U is finite and > 0.
   These are the formulas of the legacy `small_signal.figures`.
4. **DEVSIM translation.** Capability `analyses` gains "small_signal". Every contact is built as a circuit contact
   with its own voltage source (as `simws.tcad.hbt.build(circuit=True)` does); the bias entries are ramped in order
   with `simws.tcad.solve.ramp_source` and the problem's Numerics; with Klaassen mobility the legacy outer
   iteration runs after every ramp and the result carries the flag `"ac:frozen_mobility_linearisation"` (the AC
   solve linearises at frozen mobility, as in the legacy code). Y parameters as in the legacy
   `small_signal.y_params`: drive one port source with unit AC amplitude, `solve(type="ac", frequency=f)`, read
   the source currents.
5. **Adapter quantities (`device.<engine>`, version "1.2").** `dc.I.<contact>` scalars (operating point) and
   `dc.kirchhoff_residual` as for equilibrium; with the axis `{"name": "f", "unit": "Hz"}`:
   `ac.y11_re`, `ac.y11_im`, ... `ac.y22_im` in "S/m" (2D: S/cm x 100) or "S/m**2" (1D: S/cm^2 x 1e4), and
   `ac.h21_abs` (dimensionless). Scalars, present only when defined: `ac.fT` in Hz, the single-pole value with
   beta0 = |h21| at the LOWEST frequency and |h21| at the HIGHEST frequency (needs at least two frequencies;
   absent with the flag `"ac:no_rolloff"` when beta0 <= |h21| there); `ac.fmax` in Hz, `f_max` at the highest
   frequency (absent with the flag `"ac:U_unbounded"` when U is not finite and positive there).
6. **Limits, stated.** Small-signal results are screening fidelity: a mesh study (and so paper fidelity) on a
   small-signal problem is refused with `invalid_input` and a message containing
   `"mesh_study: not available for small_signal"`. No noise analysis. Charon does not support it.
7. **Acceptance R9 (tolerances fixed here; slow).** Legacy: `devsim.hbt_ac` through the hub with
   `params={"VBE": [0.75], "VCE": 1.0}`; from its `hbt.tau_charge` take fT_c = 1 / (2 pi tau) and the two
   frequencies the legacy adapter uses, f_lo = 1e-4 fT_c and f_hi = 0.1 fT_c. New path: `device.devsim` on the
   library HBT with `SmallSignal(bias=[Bias("collector", 1.0, 0.1), Bias("base", 0.75, 0.05)],
   ports=("base", "collector"), frequencies_Hz=[f_lo, f_hi])` and
   `Numerics(relative_error=1e-10, absolute_error=1e10, max_iterations=60)` (the legacy AC bias settings).
   Then `ac.fT` and `ac.fmax` equal `hbt.fT` and `hbt.fmax` within 1e-6 relative, and `dc.I.collector`,
   `dc.I.base` equal `hbt.IC`, `hbt.IB` within 1e-6 relative.
8. **Roll-off guard (lead, 16 Nov, found on the first trial run before any acceptance run).** A zero-biased diode
   returned |h21| = 1.0000000 at both frequencies and a meaningless `ac.fT` from a 1e-7 difference. `ac.fT` is
   reported only when |h21| falls by more than 1e-6 relative between the lowest and the highest frequency;
   otherwise it is absent with the flag `"ac:no_rolloff"`. R9 is unaffected (the legacy procedure evaluates
   |h21| a decade below f_T, where the fall is large).
9. **After the checker's report (16 Nov).** R9 passed: `ac.fT` 4.98316e9 Hz and `ac.fmax` 1.69436e10 Hz at
   VBE 0.75 V, VCE 1.0 V agree with the legacy adapter to 1.3e-11 and 1.6e-11 relative, and the operating-point
   currents exactly. **One recorded FAIL**, in the checker's own sanity test of a zero-biased diode as a one-port:
   current conservation (y11 = -y12 = -y21 = y22) to 1e-6 relative, fixed before the run, is met by the imaginary
   parts (1e-13) and not by the real parts: Re y11 = 4.74e-7 against -Re y12 = 4.90e-7 S/cm^2 at 1 kHz, a
   mismatch of 7.8e-5 of |y|. The checker reproduced the same error with the legacy AC code alone, so it comes
   from DEVSIM's AC solve, not from the translation. The tolerance is not changed and the check stays in the
   suite as an expected failure. Post hoc: the imaginary parts conserve to 1e-9. **What it means for use:** an AC
   conductance smaller than about 1e-4 of |Y| (here about 1e-8 S/cm^2) is solver error, not physics; capacitances
   and the RF figures (f_T, f_max, evaluated where the real parts are large) are unaffected. The same test
   confirmed the capacitance: 3.2335e-8 F/cm^2 against 3.1187e-8 from the depletion formula (+3.7 %, of which
   about +3.5 % is the known correction for the carrier tails).
   Clarifications: an empty `bias` list is allowed (zero bias); with one frequency there is no `ac.fT` and no
   `"ac:no_rolloff"` flag; the self-check of a small-signal result is the Kirchhoff sum of the operating-point
   currents; the mesh-study refusal happens before the engine runs; `Bias` takes keyword arguments.

### A7 (16 Nov 2026, lead): ion implantation as a second problem family; first engine RustBCA

Everything before A7 describes one family of problems: a device with contacts, solved for currents. Ion
implantation is a different question (a beam and a layered target in, a depth profile out), so it gets its own
description and result type next to the device ones. The hub, the store, trust, compare, the catalogue and the
study layer are shared. Owner's instruction for new engines: A5.0.

0. **What the lead had already seen when this was written (so the reader can judge the tolerances below).**
   (a) Scouting, 16 Nov: RustBCA 3.0.0, 5000 P ions at 100 keV into Si with recoil tracking, Kr-C potential and
   placeholder binding energies gave Rp = 138.3 nm and dRp = 50.9 nm. (b) An interface trial, 16 Nov: 300 As ions
   at 80 keV, only to learn the input format and the timing (0.03 s without recoils; the same seed gives identical
   output). (c) The same trial with recoil tracking switched on was killed by the operating system for memory.
   No case of R16 below had been run. Because of (a), P at 100 keV is reported but is not a gate.
1. **Description (`simws/spec/implant.py`, exported from `simws.spec`; nothing in the device description
   changes, and no existing problem id changes).** Units in the field names. All classes forbid unknown fields.

   ```python
   class Atom(_M):
       symbol: str                        # element symbol, e.g. "P"; not empty
       Z: int                             # 1 .. 118
       mass_amu: float                    # > 0 (for an implanted ion: the isotope mass)
       cutoff_eV: float                   # > 0: a moving atom slower than this counts as stopped
       surface_binding_eV: float = 0.0    # >= 0
       bulk_binding_eV: float = 0.0       # >= 0

   class Layer(_M):
       name: str
       thickness_um: float                # > 0
       density_cm3: Dict[str, float]      # symbol -> atoms per cm^3; at least one entry, every value > 0

   class Target(_M):
       atoms: List[Atom]                  # at least one; symbols unique
       layers: List[Layer]                # at least one; the beam side first; names unique
       structure: Literal["amorphous"] = "amorphous"
       # every symbol used in a layer must be in `atoms`; property thickness_um = sum of the layers

   class Beam(_M):
       ion: Atom
       energy_keV: float                  # > 0
       tilt_deg: float = 0.0              # 0 <= tilt < 90, measured from the surface normal
       dose_cm2: Optional[float] = None   # > 0 when given

   class ImplantModels(_M):
       potential: Literal["zbl", "kr_c", "moliere"] = "zbl"
       electronic_stopping: Literal["lindhard_scharff"] = "lindhard_scharff"
       recoils: bool = False              # follow the target atoms that are knocked on (needed for damage)

   class Statistics(_M):
       n_ions: int = 20000                # 100 <= n_ions <= 2_000_000
       seed: int = 1                      # >= 0

   class ImplantProblem(_M):
       spec_version: str = "1.0"
       kind: Literal["implant"] = "implant"
       beam: Beam
       target: Target
       models: ImplantModels = ImplantModels()
       statistics: Statistics = Statistics()
       depth_bin_um: float = 0.005        # > 0; target thickness / depth_bin_um must not exceed 5000
       def problem_id(self) -> str: ...                 # same recipe as Problem.problem_id
       def with_changes(self, changes) -> "ImplantProblem": ...   # dotted paths, validated again
   ```

   Every violated rule above raises pydantic's `ValidationError` (a `ValueError`).
   `simws.spec.family(problem)` returns `"implant"` for an `ImplantProblem` and `"device"` for a `Problem`.
2. **Library (`simws.spec.library.implant_si`).**
   `implant_si(ion="P", energy_keV=100.0, tilt_deg=0.0, dose_cm2=None, thickness_um=2.0, n_ions=20000, seed=1)`
   returns an `ImplantProblem`: one layer named "si" of amorphous silicon. `ion` is "B", "P" or "As"; anything
   else raises `ValueError` naming the known ions. Parameters and where they come from (project rule 1):

   | value | source |
   |---|---|
   | isotope masses: 11B 11.009305, 31P 30.973762, 75As 74.921595 amu | AME2020 atomic mass evaluation (Wang et al., Chinese Physics C 45, 030003, 2021), rounded |
   | silicon mass 28.085 amu | IUPAC standard atomic weight (abridged, 2021) |
   | silicon density 4.994e22 atoms/cm^3 | 8 atoms per cubic cell of edge 0.5431020511 nm (CODATA 2018 lattice parameter of silicon) |
   | silicon surface binding 4.72 eV, cutoff 1.5 eV, bulk binding 0 eV | RustBCA 3.0.0 example table `scripts/materials.py`, which its authors label "example input values only" |
   | the ion's cutoff 1.5 eV and surface binding 0 eV | **not data, a modelling choice**: the ion counts as stopped when slower than the silicon cutoff, and meets no barrier at the surface. R14 below checks that the range does not depend on it. |

3. **Engine layer (`simws/engines/base.py`, additive).**

   ```python
   class ImplantResult(BaseModel):          # forbids unknown fields
       engine: str
       problem_id: str
       n_ions: int
       n_reflected: int                     # left through the front surface
       n_transmitted: int                   # left through the back of the last layer
       depth_um: List[float]                # one value per implanted ion, in launch order: depth below the front surface
       lateral_um: List[float]              # same ions: displacement from the entry point along the axis
                                            # perpendicular to the plane of incidence
       flags: List[str] = []
       diagnostics: Dict[str, Any] = {}

   class ImplantCapability(BaseModel):      # forbids unknown fields; every list holds the ALLOWED values
       engine: str
       env: str
       description: str
       in_process: bool
       structures: List[str]
       potentials: List[str]
       electronic_stopping: List[str]
       recoils: List[bool]
       max_layers: int
       max_keV_per_amu: float               # beam energy / ion mass above this is refused
       tested_keV_per_amu: Tuple[float, float]
       tested_target_symbols: List[str]
       notes: str = ""
       def unsupported(self, problem) -> List[str]: ...
       def outside_tested_range(self, problem) -> List[str]: ...
   ```

   `unsupported` returns one reason per violated field, each starting with the field name and a colon
   (`"structures:"`, `"potentials:"`, `"electronic_stopping:"`, `"recoils:"`, `"max_layers:"`,
   `"max_keV_per_amu:"`); empty when supported. `outside_tested_range` returns a string starting with
   `"energy_keV_per_amu:"` when energy/mass is outside the tested interval (bounds included), and one starting
   with `"target:"` when the target holds a symbol that is not in `tested_target_symbols`; else `[]`.
   **Registry.** `ENGINES` gains `"rustbca": "simws.engines.rustbca_engine"`. An engine module may set
   `FAMILY` ("device" when absent). `engines.family(name)` returns it; `engines.adapter_name(name)` returns
   `"<family>.<name>"` (`"device.devsim"`, `"implant.rustbca"`). `engines.unsupported(name, problem)` returns
   exactly one reason, starting with `"problem kind:"`, when the engine's family differs from
   `simws.spec.family(problem)`, without calling the engine module's own check. So
   `plan(implant_problem)` has primary "rustbca" with devsim and charon refused, and `plan(device_problem)`
   lists rustbca under `refused`. `Study` submits to `engines.adapter_name(engine)`.
4. **RustBCA translation (`simws/engines/rustbca_engine.py`; `FAMILY = "implant"`).** RustBCA 3.0.0 (Drobny and
   Curreli, JOSS, doi 10.21105/joss.03298), built from source as the Python module `libRustBCA`, called in
   process through `rustbca_local_py(input, "1D")`. Module functions: `available()`, `unsupported(problem)`,
   `build_input(problem) -> dict` (pure Python: it does not import the engine) and
   `run(problem, workdir) -> ImplantResult` (raises `EngineNotAvailable` when `libRustBCA` cannot be imported,
   `Unsupported` when the capability refuses the problem).
   Capability: structures ["amorphous"]; potentials ["zbl", "kr_c", "moliere"]; electronic_stopping
   ["lindhard_scharff"]; recoils [False] (see 0c: following every recoil in memory is not safe on a laptop, so
   damage profiles wait for a streaming path); max_layers 20; max_keV_per_amu 25.0 (RustBCA's README gives
   25 keV per nucleon as the limit of its low-energy stopping models); tested_keV_per_amu (1.6, 9.1) and
   tested_target_symbols ["Si"] (the cases of R16).
   `build_input` returns the RustBCA input dictionary with these entries (others may be present):
   `options`: `track_recoils` False, `electronic_stopping_mode` "LOW_ENERGY_NONLOCAL", `interaction_potential`
   `[["ZBL"]]`, `[["KR_C"]]` or `[["MOLIERE"]]`, `scattering_integral` `[["MENDENHALL_WELLER"]]`,
   `mean_free_path_model` "LIQUID", `seed` = statistics.seed, `num_chunks` 1;
   `particle_parameters`: `length_unit` "MICRON", `energy_unit` "EV", `mass_unit` "AMU", `N` [n_ions], `Z`, `m`,
   `Ec`, `Es` one-element lists from the ion, `E` [energy_keV x 1000], `pos` [[0, 0, 0]],
   `dir` [[cos tilt, sin tilt, 0]];
   `geometry_input`: `length_unit` "MICRON", `layer_thicknesses` in micrometres, `densities` one row per layer
   and one column per entry of `target.atoms` (in that order) in atoms per cubic micrometre (cm^-3 x 1e-12, 0
   for an absent species), `electronic_stopping_correction_factors` all 1.0;
   `material_parameters`: `energy_unit` "EV", `mass_unit` "AMU", and `Z`, `m`, `Ec`, `Es`, `Eb` one entry per
   target atom in the order of `target.atoms`.
   x is depth, the plane of incidence is x-y, so `lateral_um` is RustBCA's z. An ion that left the target is
   counted as transmitted when its last x is beyond half the target thickness, otherwise as reflected.
   The number of threads comes from the environment variable `SIMWS_THREADS` (default: the smaller of 4 and the
   number of cores); the result must not depend on it. Every result carries a flag starting with
   `"implant:amorphous_target"` (no channeling: a crystalline wafer shows a deeper tail; no damage build-up, so
   the profile is linear in dose).
5. **Adapter `implant.rustbca`** (`simws/adapters/implant.py`: base `ImplantAdapter`, subclass
   `ImplantRustbca`; manifest version "1.0", env "tcad", inputs ["problem"], citation "drobny2021"). It follows
   the device adapter's rules: an invalid description is `invalid_input` raised from submit; a problem the
   engine refuses is `invalid_input` with `"unsupported by rustbca: <reasons>"`; a missing engine is
   `engine_error` saying it is not installed. With N = n_ions, k = number of implanted ions, x their depths:

   | quantity | unit | value | standard uncertainty |
   |---|---|---|---|
   | `implant.Rp` | m | mean of x | s / sqrt(k) |
   | `implant.dRp` | m | s = standard deviation of x (k-1 in the denominator) | (s/2) sqrt((m4/s^4 - (k-3)/(k-1)) / k), m4 = mean of (x - mean)^4 |
   | `implant.lateral_straggle` | m | standard deviation of `lateral_um` (k-1) | same formula on the lateral values |
   | `implant.fraction_implanted`, `implant.fraction_reflected`, `implant.fraction_transmitted` | dimensionless | count / N | sqrt(p(1-p)/N) with p = (count+1)/(N+2) |
   | `implant.profile` | 1/m | count_i / (N b): stopping probability per ion and per metre of depth | sqrt(max(count_i, 1)) / (N b) |
   | `implant.concentration` (only when `dose_cm2` is given) | 1/m**3 | dose x profile | dose x the profile's |

   The profile axis is `{"name": "depth", "unit": "m"}` with the bin centres; bin width b = `depth_bin_um`;
   the bins start at the surface and there are ceil(thickness / b) of them (so the sum of profile x b equals
   `implant.fraction_implanted`). All but `implant.concentration` are declared outputs. Every quantity carries
   the uncertainty component `"statistical"` equal to its sigma, and the result carries the flag
   `"budget:statistical_only"` (the error bars are counting statistics; they say nothing about the potential or
   the stopping model). Further rules:
   fewer than 100 implanted ions: status `outside_validity`, message containing `"stopped in the target"`;
   any transmitted ion: a flag starting with `"implant:transmitted:"`;
   each `outside_tested_range` string s: the flag `"extrapolated:" + s`;
   `verification["statistics"] = {"kind": "solution", "method": "split halves", "n_implanted": k,
   "rp_halves_um": [a, b], "z": z, "passed": abs(z) <= 4.0}`, where a and b are the mean depths of the first
   k//2 implanted ions and of the rest, and z is their difference over its standard error;
   self-check (status `check_failed`): the three counts do not add up to N; `abs(z) > 4`; or paper fidelity with
   k < 10000 (message containing `"paper fidelity needs at least 10000 implanted ions"`).
6. **Limits, stated.** Amorphous targets only: no channeling. Static target: no damage build-up, no sputtering
   of the surface during the implant, no annealing or diffusion afterwards, so the profile is the as-implanted
   one and is linear in dose. Range profiles only: no vacancy or damage profile yet. Lindhard-Scharff stopping
   only, so beams above 25 keV per nucleon are refused. **Trust:** an implant result stays "unverified" until a
   second, different implant engine is connected and compared; the table check R16 is acceptance evidence for
   the translation, it is not a verification of the physics and not a validation.
7. **Acceptance (tolerances fixed here, before any of these runs).** Library problems with their defaults
   (20 000 ions, seed 1, ZBL) unless stated. "Within 4 sigma" means: the difference is at most 4 times the root
   sum of squares of the two standard uncertainties the adapter reports.
   R10 (repeatability): `implant_si("P", 50, n_ions=2000)` run twice gives identical `depth_um`; with seed 2 it
   differs; with `SIMWS_THREADS=1` it is identical to the default thread count.
   R11 (counting): in every run implanted + reflected + transmitted = N. A 0.05 um target under P at 100 keV
   (n_ions=2000) has transmitted ions and the adapter flags it; a 0.005 um target gives `outside_validity`.
   R12 (honest error bars): `implant_si("P", 50)` with seed 1 and with seed 2: `implant.Rp`, `implant.dRp` and
   `implant.lateral_straggle` each agree within 4 sigma.
   R13 (layers): the same problem with the target written as two silicon layers, 0.03 um and 1.97 um, agrees
   with the one-layer result in `implant.Rp` and `implant.dRp` within 4 sigma.
   R14 (the ion's modelling choice): the same problem with the ion's `cutoff_eV` = 5.0 and
   `surface_binding_eV` = 3.0 agrees with the default in `implant.Rp` and `implant.dRp` within 4 sigma.
   R15 (trends; signs and one band): Rp(P, 100 keV) > Rp(P, 50 keV); at 50 keV Rp(B) > Rp(P) > Rp(As); and for
   P at 50 keV the ratio Rp(tilt 60 degrees) / Rp(tilt 0) lies in [0.45, 0.60] (in an infinite amorphous medium
   it is cos 60 = 0.5; the surface removes some shallow ions).
   R16 (published range tables): Gibbons, Johnson and Mylroie, *Projected Range Statistics*, 2nd ed. (1975), as
   reproduced in G. Chapman, Simon Fraser University ENSC 495 lecture 7
   (https://www.sfu.ca/~gchapman/e495/e495l7o-2.pdf, read 16 Nov 2026). Rp / dRp in um: B 50 keV 0.1608 / 0.0504;
   B 100 keV 0.2994 / 0.0710; P 50 keV 0.0607 / 0.0256. Gate: `implant.Rp` and `implant.dRp` each within 25 % of
   the table value. Reported without a gate: P 100 keV 0.1238 / 0.0456 (see 0a).
   Why 25 %: the tables are a 1975 transport theory (LSS), RustBCA is a Monte Carlo code with a different
   potential, and codes of this kind are documented to miss measured ranges by up to 25 % (Titze et al., J. Vac.
   Sci. Technol. A 39, 2021, doi 10.1116/6.0001406, antimony in silicon). A mistake in units, density or
   geometry gives a factor of two or more, which this gate catches; a 10 % physics difference it does not.
8. **Laptop install.** `tools/install_rustbca.sh` builds RustBCA at commit
   51bee889956147e3287491d0c68957967235a33c (version 3.0.0) in a throw-away build environment and installs the
   one resulting wheel into the `tcad` env without touching its other packages. It needs a Rust compiler
   (`cargo`); when there is none it says how to get one and stops. Tests that need the engine are skipped where
   `libRustBCA` is not installed.
9. **After the checker's report (16 Nov).** The checker wrote 161 tests from the text of A7 alone; 158 passed.
   R10, R12, R13, R14 and R15 passed (20 000 ions, seed 1, ZBL): two seeds agree in Rp, dRp and lateral straggle
   within 0.4 sigma; a different ion cutoff and surface binding moves Rp by 0.02 sigma; the 60 degree tilt ratio
   is 0.560. R16 for phosphorus passed: at 50 keV Rp 71.34 +- 0.21 nm against 60.7 (+17.5 %) and dRp 29.47 +- 0.13
   nm against 25.6 (+15.1 %); at 100 keV (reported, no gate) Rp 139.81 nm (+12.9 %) and dRp 50.69 nm (+11.2 %).
   **Recorded FAIL: R16 for boron.** At 50 keV Rp 216.1 nm against 160.8 (+34.4 %) and dRp 72.3 nm against 50.4
   (+43.4 %); at 100 keV Rp 407.1 nm against 299.4 (+36.0 %) and dRp 107.2 nm against 71.0 (+51.0 %). The gate
   is 25 % and is not changed; the two boron checks stay in the suite as expected failures.
   It is not a translation error: the checker called RustBCA directly with an input written from A7.4, not from
   `build_input`, and got the same ranges. *Post hoc diagnostics* (run after the result was seen, so they explain
   and do not excuse): the choice of potential moves the boron range by 3 % (Kr-C +32 %, Moliere +36 %);
   RustBCA's other low-energy stopping modes are further off (checker: local +64 %, equipartition +49 % at
   50 keV); multiplying the electronic stopping by 1.6 brings boron at 100 keV to +1.7 % in Rp and -0.1 % in
   dRp. So the cause is the Lindhard-Scharff electronic stopping, which is too weak for boron in silicon; for a
   light ion at these energies electronic stopping decides the range, for phosphorus it does not.
   **What it means for use:** RustBCA boron ranges in silicon are about 35 % too deep and must not be used as
   numbers. No correction factor is applied: a factor needs a source (measured or tabulated stopping), and
   fitting it to the same table that failed would be calibration presented as agreement.
   **Changes made because of the FAIL (after the result, so listed here):**
   (a) `ImplantCapability` gains `tested_ion_symbols: List[str] = []` (empty: any ion). `outside_tested_range`
   returns a string starting with `"ion:"` when the list is not empty and the beam ion is not in it.
   (b) The RustBCA capability now lists only what passed: `tested_keV_per_amu` (1.6, 3.3) and
   `tested_ion_symbols` ["P"]. Boron and arsenic results therefore carry `extrapolated:` flags and
   `in_range` False (arsenic is not known to be wrong; the source used for R16 has no arsenic table).
   (c) A RustBCA result for boron into a target that holds silicon carries a flag starting with
   `"implant:known_failure:"`.
   **Corrections and clarifications (no tolerance involved):**
   R11 as written by the lead was inconsistent: the 0.05 um target under P at 100 keV stops only 71 of 2000
   ions, so the result is `outside_validity`, and the adapter raised before it set the transmitted flag. A7.5
   says "any transmitted ion: a flag" without a condition, so the adapter now sets the flag first; it is present
   on the `outside_validity` result too.
   Split halves: z = (a - b) / sqrt(var_a/n_a + var_b/n_b), sample variances with n-1.
   `implant.concentration` = `dose_cm2` x 1e4 (ions per m^2) x profile.
   An ion exactly on a bin edge is counted in the deeper bin; the last bin takes anything at or beyond its end.
   The RustBCA capability's `env` is "tcad". "The target holds a symbol" means: a symbol used in a layer.
   A paper-fidelity run with at least 10 000 implanted ions is accepted like any other (status ok).
   R13 with the same seed is bit-identical to the one-layer run, so it tests the layer bookkeeping only.
   **Next for this engine, not done here:** an electronic-stopping correction per ion and target taken from a
   source (the SRIM stopping tables the workspace can already read), with the range tables held out as the test;
   a second implant engine, so that results can become "verified"; damage profiles through a streaming path.
