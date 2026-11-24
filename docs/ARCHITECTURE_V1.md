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

### A8 (17 Nov 2026, lead): single-event energy deposition as a third problem family; engine G4SEE

A particle hits a chip; how much energy does it leave in a small sensitive volume (SV), event by event? That
spectrum is the first step of every single-event-effect estimate. It gets its own description and result type
(as A7 did for implants); hub, store, trust, compare, catalogue and study are shared. Owner's instruction: A5.0.

0. **What the lead had already seen when this was written.** (a) G4SEE v1.0 (Lucsanyi et al., CERN; GPL-3) built
   here against Geant4 11.2.2 from conda-forge. (b) One trial run to learn the macro and output format, not an
   acceptance case: 2000 protons of 3 MeV into a 20 um silicon SV, 3 s. (c) The reference values of R22 to R25
   below, computed from the NIST tables before any acceptance run. No acceptance case below had been run.
1. **Description (`simws/spec/see.py`, exported from `simws.spec`).** Units in the field names. All classes forbid
   unknown fields; every violated rule raises pydantic's `ValidationError`.

   ```python
   G4_MATERIAL = r"^G4_[A-Za-z0-9_]+$"   # a Geant4 NIST material name, e.g. "G4_Si", "G4_SILICON_DIOXIDE"

   class SeeSource(_M):
       particle: Literal["proton", "alpha", "neutron", "e-", "gamma", "ion"]
       ion_Z: Optional[int] = None          # 1..118; required when particle == "ion", forbidden otherwise
       ion_A: Optional[int] = None          # >= ion_Z; required when particle == "ion", forbidden otherwise
       energy_MeV: float                    # > 0, kinetic energy of the whole particle
       tilt_deg: float = 0.0                # 0 <= tilt < 90, from the surface normal

   class BeolLayer(_M):
       name: str                            # not empty
       material: str                        # matches G4_MATERIAL
       thickness_um: float                  # > 0

   class SeeGeometry(_M):
       bulk_material: str = "G4_Si"         # matches G4_MATERIAL; the SV is made of the same material
       bulk_width_um: float                 # > 0, both lateral sides
       bulk_thickness_um: float             # > 0
       sv_depth_um: float = 0.0             # >= 0: top of the SV below the top of the bulk
       sv_width_um: Tuple[float, float]     # each > 0 and <= bulk_width_um
       sv_thickness_um: float               # > 0; sv_depth_um + sv_thickness_um <= bulk_thickness_um
       beol: List[BeolLayer] = []           # layers on top of the bulk, the beam side first; names unique

   class SeePhysics(_M):
       em: Literal["option4", "option3", "livermore"] = "option4"
       hadronic: Literal["none", "ftfp_bert_hp"] = "none"

   class EdepHistogram(_M):
       scale: Literal["lin", "log"] = "lin"
       bins: int = 2000                     # 10 <= bins <= 20000
       low_MeV: float = 0.0                 # >= 0; > 0 when scale == "log"
       high_MeV: float                      # > low_MeV

   class SeeStatistics(_M):
       n_events: int = 10000                # 100 <= n_events <= 100_000_000
       seed: int = 1                        # >= 0

   class SeeProblem(_M):
       spec_version: str = "1.0"
       kind: Literal["see"] = "see"
       source: SeeSource
       geometry: SeeGeometry
       physics: SeePhysics = SeePhysics()
       histogram: EdepHistogram
       statistics: SeeStatistics = SeeStatistics()
       def problem_id(self) -> str: ...                 # same recipe as Problem.problem_id
       def with_changes(self, changes) -> "SeeProblem": ...       # dotted paths, validated again
   ```

   `simws.spec.family(problem)` returns `"see"` for a `SeeProblem` (A7.1 otherwise unchanged).
2. **Library (`simws.spec.library.see_si_slab`).**
   `see_si_slab(particle="alpha", energy_MeV=5.0, sv_thickness_um=10.0, sv_depth_um=0.0, bulk_thickness_um=100.0,
   width_um=1000.0, tilt_deg=0.0, beol=None, ion_Z=None, ion_A=None, hadronic="none", n_events=10000, seed=1,
   bins=2000, high_MeV=None)` returns a `SeeProblem`: bulk "G4_Si" of `width_um` x `width_um` x
   `bulk_thickness_um`, SV of `width_um` x `width_um` x `sv_thickness_um` at `sv_depth_um`; `beol` is a list of
   `(name, material, thickness_um)` tuples, beam side first; a linear histogram from 0 to `high_MeV`, which
   defaults to 1.05 x `energy_MeV`. Only Geant4's own material data are used (G4_Si: 2.33 g/cm3), so the library
   introduces no numbers of its own.
3. **Engine layer (`simws/engines/base.py`, additive).**

   ```python
   class SeeResult(BaseModel):              # forbids unknown fields
       engine: str
       problem_id: str
       n_events: int
       edges_MeV: List[float]               # bins + 1 values, from the description (see 4), increasing
       counts: List[float]                  # bins values: events with edges[i] <= Edep < edges[i+1]
       underflow: float                     # events with Edep < low_MeV
       overflow: float                      # events with Edep >= high_MeV
       flags: List[str] = []
       diagnostics: Dict[str, Any] = {}

   class SeeCapability(BaseModel):          # forbids unknown fields; every list holds the ALLOWED values
       engine: str
       env: str
       description: str
       in_process: bool
       particles: List[str]
       em: List[str]
       hadronic: List[str]
       max_beol_layers: int
       tested_particles: List[str]
       tested_energy_MeV: Tuple[float, float]
       tested_materials: List[str]
       tested_em: List[str]
       tested_hadronic: List[str]
       notes: str = ""
       def unsupported(self, problem) -> List[str]: ...
       def outside_tested_range(self, problem) -> List[str]: ...
   ```

   `unsupported`: one reason per violated field, starting with `"particles:"`, `"em:"`, `"hadronic:"` or
   `"max_beol_layers:"`. `outside_tested_range` (bounds included): `"particle:"` when the particle is not in
   `tested_particles`; `"energy_MeV:"` outside `tested_energy_MeV`; `"materials:"` when the bulk or a BEOL
   material is not in `tested_materials`; `"em:"`, `"hadronic:"` when not in the tested lists; else `[]`.
   Registry: `ENGINES` gains `"g4see": "simws.engines.g4see_engine"` with `FAMILY = "see"`; the A7.3 rules on
   families, `adapter_name` ("see.g4see") and the `"problem kind:"` refusal apply to all three families.
4. **G4SEE translation (`simws/engines/g4see_engine.py`; `FAMILY = "see"`).** G4SEE v1.0 (git tag v1.0, commit
   8c6fc45; D. Lucsanyi, R. Garcia Alia, K. Bilko, M. Cecchetto, S. Fiore, E. Pirovano, "G4SEE: A Geant4-Based
   Single Event Effect Simulation Toolkit and Its Validation Through Monoenergetic Neutron Measurements", IEEE
   Trans. Nucl. Sci. 69(3), 273-281, 2022, doi 10.1109/TNS.2022.3149989), single-threaded build, against Geant4 11.2.2, run as an
   executable (out of process). Module functions: `available()`, `unsupported(problem)`,
   `build_macro(problem) -> str` (pure Python, needs no Geant4), `edges(problem) -> List[float]`,
   `parse_histogram(path, problem) -> (counts, underflow, overflow)`, `run(problem, workdir) -> SeeResult`.
   Where it is: the conda env prefix in `SIMWS_G4SEE_PREFIX`, else `<envs_root>/g4see` from the simws
   configuration; the executable is `<prefix>/bin/g4see`; `available()` is True when that file exists and is
   executable. Geant4 data paths come from `simws.radiation.geant4.dataset_env(prefix)`.
   Capability: particles ["proton", "alpha", "neutron", "e-", "gamma", "ion"]; em ["option4", "option3",
   "livermore"]; hadronic ["none", "ftfp_bert_hp"]; max_beol_layers 20; tested_particles ["alpha", "proton"];
   tested_energy_MeV (5.0, 10.0); tested_materials ["G4_Si"]; tested_em ["option4"]; tested_hadronic ["none"]
   (the cases of R22 to R25); env "g4see"; in_process False.
   The macro (`build_macro`) contains, in this order of sections:
   `/random/setSeeds <2 seed + 1> <2 seed + 2>`;
   `/SEE/geometry/Bulk <bulk_material> <bulk_width_um> um <bulk_thickness_um> um`;
   `/SEE/geometry/SV 0 0 -<sv_depth_um> um <wx> <wy> um <sv_thickness_um> um`;
   one `/SEE/geometry/BEOL/addLayer <material> <bulk_width_um> um <thickness_um> um <name>` per layer **in
   reverse order** (G4SEE stacks each new layer on top of the previous one, the description lists the beam side
   first);
   `/SEE/physics/addPhysics G4EmStandardPhysics_option3` or `G4EmLivermorePhysics` when em is not option4 (G4SEE
   adds option4 itself); for hadronic "ftfp_bert_hp" the four modules `G4HadronElasticPhysicsHP`,
   `G4HadronPhysicsFTFP_BERT_HP`, `G4IonElasticPhysics`, `G4IonPhysics`;
   `/run/initialize`;
   `/gps/particle <particle>` (and `/gps/ion <Z> <A>` for an ion), `/gps/ene/type Mono`,
   `/gps/ene/mono <energy_MeV> MeV`, `/gps/pos/type Point`,
   `/gps/direction <sin tilt> 0 <-cos tilt>`, `/gps/pos/centre <x0> 0 <z0> um` with z0 = (total BEOL thickness
   + 1) um and x0 = -tan(tilt) um, so the beam enters the top of the stack at x = 0;
   `/SEE/scoring/standard/Edep 0`, `/SEE/scoring/setHistogram <scale> <bins> <low_MeV> <high_MeV> MeV`,
   `/SEE/scoring/setPrecision 10` (G4SEE's default of 4 significant digits loses counts above 9999);
   `/run/beamOn <n_events>`.
   `edges(problem)`: lin: low + i (high - low) / bins; log: 10^(log10 low + i (log10 high - log10 low) / bins),
   i = 0 .. bins (G4SEE's own formulas). `parse_histogram` reads `Edep_0_hist_t0.out` and raises `ValueError`
   when the number of bins differs, or when a printed bin edge differs from `edges(problem)` by more than 1e-6
   relative (1e-12 MeV absolute for an edge of 0).
   `run` writes the macro to `<workdir>/g4see_<problem id, 12 characters>/run.mac`, creates `out/` there, runs
   `<exe> run.mac -o out` in that folder with the Geant4 data variables set, writes its output to `g4see.log`
   there, raises `RuntimeError` with the last 20 lines of that log when it exits with an error, and raises
   `EngineNotAvailable` / `Unsupported` as the other engines do. Flags on every result: one starting with
   `"see:em_only"` when hadronic is "none" (no nuclear reactions: neutrons deposit nothing, protons and ions lose
   their nuclear-reaction events).
5. **Adapter `see.g4see`** (`simws/adapters/see.py`: base `SeeAdapter`, subclass `SeeG4see`; manifest version
   "1.0", env "tcad", inputs ["problem"], citation "lucsanyi2022"). Invalid description, refused problem and missing
   engine behave as in A7.5. With N = n_events, c_i the counts, w_i = edges[i+1] - edges[i], m_i the bin centres
   (lin) or geometric centres sqrt(edges[i] edges[i+1]) (log), all energies converted to eV:

   | quantity | unit | value | standard uncertainty |
   |---|---|---|---|
   | `see.edep_spectrum` | 1/eV | c_i / (N w_i), axis `{"name": "edep", "unit": "eV"}` at m_i | sqrt(max(c_i, 1)) / (N w_i) |
   | `see.p_above` | dimensionless | (sum of c_j for j >= i, + overflow) / N, axis `{"name": "threshold", "unit": "eV"}` at edges[i], i = 0 .. bins-1 | sqrt(p(1-p)/N), p = (k+1)/(N+2), k the count |
   | `see.fraction_underflow`, `see.fraction_overflow` | dimensionless | count / N | as `see.p_above` |
   | `see.edep_mean` (only when underflow = overflow = 0) | eV | sum c_i m_i / N | sqrt(s^2/N + b^2), s = sqrt(sum c_i (m_i - mean)^2 / (N - 1)), b = (largest w_i with c_i > 0) / sqrt(12) |
   | `see.edep_std` (same condition) | eV | s | (s/2) sqrt((m4/s^4 - (N-3)/(N-1)) / N), m4 = sum c_i (m_i - mean)^4 / N |

   `see.edep_spectrum`, `see.p_above`, `see.fraction_underflow` and `see.fraction_overflow` are declared outputs.
   Uncertainty components: "statistical" (the counting part; for `see.edep_mean` sqrt(s^2/N)) on every quantity,
   and "binning" = b on `see.edep_mean`; the result carries the flag `"budget:statistical_and_binning"`. When
   the mean is not reported, the flag starting with `"see:mean_undefined:"` says how many events fell outside
   the histogram. Self-check (status `check_failed`): underflow + overflow + sum of counts differs from N by
   more than 1e-6 N; or paper fidelity with N < 10000. Each `outside_tested_range` string s gives the flag
   `"extrapolated:" + s`.
6. **Limits, stated.** One SV, box geometry, a point source with one energy and one direction; histogram
   statistics only (no event-by-event list, no charge or current); single-threaded. Trust stays "unverified":
   the only other Geant4 path in simws (`simws.radiation.geant4`, g4track) runs the same Geant4, so a comparison
   with it is not independent (A2). R22 to R25 compare with NIST tables built on ICRU Report 49; Geant4's
   low-energy stopping for alphas and protons in standard materials is expected to rest on the same ICRU 49
   data, so these checks test the translation (geometry, units, direction, histogram), not the physics.
7. **Acceptance (tolerances fixed here, before any of these runs).** Library problems, 10 000 events, seed 1,
   unless stated. "Within 4 sigma" as in A7.7.
   Reference method for R22 to R25: integrate dE/dx = -S(E) rho along the path through silicon (RK4, step
   0.01 um), S the NIST total (electronic + nuclear) stopping power interpolated linearly in log S against
   log E, rho = 2.33 g/cm3; the expected mean deposit is E(entry of the SV) - E(exit of the SV). Tables:
   ASTAR (alphas) and PSTAR (protons) for silicon, NIST Standard Reference Database 124
   (https://physics.nist.gov/PhysRefData/Star/Text/, retrieved 17 Nov 2026, default energy grid), stored with
   their source in `simws/data/nist_star/` and read by `simws.radiation.nist_star`. Values from that method:
   R22 1.6137 MeV, R23 0.8327 MeV, R24 0.8558 MeV; removing every other table point changes them by at most
   0.13 %.
   R20 (repeatability): `see_si_slab(n_events=1000)` twice gives identical counts; seed 2 gives different counts.
   R21 (bookkeeping): in every run of R20 to R26, underflow + overflow + sum of counts = N, and
   `see.p_above` at threshold 0 is 1 when underflow is 0.
   R22 (alpha, partial): alpha 5.0 MeV, SV 0 to 10 um: `see.edep_mean` within 3 % of 1.6137 MeV.
   R23 (proton, partial): proton 10.0 MeV, SV 0 to 100 um in a 300 um bulk: within 3 % of 0.8327 MeV.
   R24 (alpha through layers): alpha 5.0 MeV, BEOL two G4_Si layers of 1.0 um and 2.0 um, SV from 2 um to 7 um
   below the bulk top (path to the SV 5 um, through it 5 um): within 3 % of 0.8558 MeV.
   R25 (tilt): alpha 5.0 MeV at 60 degrees, SV 0 to 5 um (path in the SV 10 um): within 3 % of 1.6137 MeV.
   R26 (full absorption): alpha 5.0 MeV, SV 0 to 50 um (CSDA range 24.3 um): `see.edep_mean` within 1 % of
   5.0 MeV and `see.p_above` at the threshold edge closest to 4.75 MeV at least 0.99.
   R27 (honest error bars): R23 with seed 1 and seed 2: `see.edep_mean` within 4 sigma.
   R28 (hadronic smoke test, no number gated): neutron 14 MeV, hadronic "ftfp_bert_hp", 300 um bulk, SV 0 to
   100 um, 2000 events, histogram 0 to 20 MeV: status ok, no `see:em_only` flag, and at least one event above
   0.1 MeV.
   Why 3 %: ICRU 49 gives 2 to 5 % uncertainty for these stopping powers; Geant4 and the table should share it,
   so 3 % leaves room for the Monte Carlo mean (straggling, delta electrons leaving the SV) while a slip in
   thickness, density, direction or units (10 % and more) fails. Why 1 % for R26: energy is conserved; what may
   be missing is backscatter and escaping delta electrons, well below 1 % for alphas.
8. **Laptop install.** `tools/install_g4see.sh` creates a conda env `g4see` (conda-forge: geant4 11.2.2 and a
   C++ compiler; it downloads several GB of Geant4 data once), downloads G4SEE at tag v1.0, builds it with
   `geant4-config` (as `tools/geant4_track/build.sh` does; Geant4's CMake files ask for development headers the
   conda packages do not ship) and installs `g4see` and `mergeHistograms` into that env. It does not touch
   `process_rad` or `tcad`. Engine tests are skipped where G4SEE is not available.

9. **Correction during implementation (17 Nov, before any acceptance run).** A8.4 said
   `/SEE/scoring/setPrecision 10`. With 10 digits after the point a number is 16 characters wide and G4SEE
   writes it into a 15-character column, so edge and count run together ("4.6620000000e-016.0000000000e+00",
   seen on the trial case of 0b). The macro uses `/SEE/scoring/setPrecision 8` instead (9 significant digits,
   counts exact up to 999 999 999, always a space between the columns). `see_si_slab` rounds its default
   `high_MeV` to 12 significant digits, as the hub rounds job inputs, so the stored problem id equals the id of
   the problem the library returns.
10. **After the checker's report (17 Nov).** The checker wrote 283 tests from the text of A8 alone; 282 passed.
    **All acceptance checks passed** (10 000 events, seed 1; sigma includes the binning term):
    R22 1.61364 +- 0.00106 MeV against 1.6137 (-0.004 %); R23 0.83125 +- 0.00159 MeV against 0.8327 (-0.18 %);
    R24 0.85598 +- 0.00083 MeV against 0.8558 (+0.02 %); R25 1.61603 +- 0.00126 MeV against 1.6137 (+0.14 %);
    R26 4.99930 +- 0.00076 MeV against 5.0 (-0.01 %) and `see.p_above` at 4.75 MeV 1.00000; R27 seeds 1 and 2
    differ by 0.38 sigma; R28 ok with 6 events above 0.1 MeV; R20 and R21 hold. The agreement is far inside the
    3 % gates, as expected when Geant4 and the tables share ICRU 49 data (A8.6): it confirms the set-up, it says
    nothing about the physics.
    **Defect fixed:** a result with more counted events than N (an inconsistent result) made the `see.p_above`
    uncertainty the square root of a negative number; the NaN stopped the store before the self-check could mark
    it `check_failed`. p is now clipped to [0, 1]; the self-check then fails the result as A8.5 says.
    **Clarifications:** `parse_histogram` takes the histogram file's path. `budget:statistical_and_binning` is
    set on every result. `see.edep_std` has uncertainty 0 when every event is in one bin. One `"materials:"`
    string lists all materials. The 0.13 % in A8.7 holds for removing the odd table rows; removing the even rows
    changes R24 by -0.15 %.
    **R28 is a smoke test whose outcome depends on the seed:** about 2 neutron interactions are expected in the SV
    for 2000 events, so another seed can give none (roughly 15 %). It stays as written (seed 1).

### A9 (18 Nov 2026, lead): circuits as a fourth problem family; engine ngspice with OpenVAF

The device, implant and single-event families answer "what does this piece of silicon do"; a circuit answers
"what does it do in a circuit". This adds a netlist description, an ngspice translator that can load Verilog-A
compact models compiled by OpenVAF, and the first link from a device result to a circuit model.
Owner's instruction for new engines: A5.0 (and "lets go with ... ngspice + openvaf", 18 Nov).

0. **What the lead had already seen when this was written.** (a) The conda-forge ngspice (version 41, the newest
   there) is built without OSDI: `pre_osdi` and `osdi` are unknown commands, so no OpenVAF model can be loaded
   into it. (b) ngspice 47 built here from its release tarball with `--enable-osdi` loads models compiled by the
   OpenVAF-Reloaded build `osdi_0.3-31-gf47d557`. (c) Two trial runs to learn the interfaces: a Verilog-A
   Shockley diode with is = 1e-14 A, n = 1, swept 0.3 to 0.7 V in 0.1 V steps (printed currents 1.08957e-9 to
   5.67031e-3 A), and an AC run of that diode with an RC load written as an ASCII raw file. Case (c) overlaps
   R34 below; its numbers were seen. No other acceptance case had been run. (d) The constants behind the thermal
   voltage: ngspice 47 uses k = 1.38064852e-23 J/K and q = 1.6021766208e-19 C (`src/include/ngspice/const.h`),
   OpenVAF k = 1.3806488e-23, q = 1.602176565e-19 (`$vt`); CODATA 2018 k/q differs from ngspice's by 3.4e-7 and
   from OpenVAF's by 1.0e-7 (relative).
1. **Description (`simws/spec/circuit.py`, exported from `simws.spec`).** SI units in the field names. All
   classes forbid unknown fields; every violated rule raises pydantic's `ValidationError`. Names and nodes are
   case-insensitive, as in SPICE.

   ```python
   NAME = r"^[A-Za-z][A-Za-z0-9_]*$"
   NODE = r"^[A-Za-z0-9_]+$"                 # "0" is ground

   class Pulse(_M):
       v1: float; v2: float
       delay_s: float = 0.0                   # >= 0
       rise_s: float; fall_s: float; width_s: float; period_s: float      # each > 0

   class Pwl(_M):
       points: List[Tuple[float, float]]      # (time s, value); >= 2 points, times >= 0 and strictly increasing

   class Element(_M):
       name: str                              # NAME; unique among elements (case-insensitive)
       kind: Literal["resistor", "capacitor", "inductor", "vsource", "isource", "diode", "bjt", "va", "subckt"]
       nodes: List[str]                       # NODE each; 2 for resistor, capacitor, inductor, vsource, isource,
                                              # diode (anode, cathode); 3 or 4 for bjt (c, b, e[, substrate]);
                                              # >= 1 for va and subckt
       value: Optional[float] = None          # required and > 0 for resistor (ohm), capacitor (F), inductor (H);
                                              # optional DC value for sources (V or A, default 0); else forbidden
       ac: Optional[float] = None             # sources only: AC magnitude, phase 0
       pulse: Optional[Pulse] = None          # sources only
       pwl: Optional[Pwl] = None              # sources only; not together with pulse
       model: Optional[str] = None            # required for diode, bjt, va (a Model name) and subckt (a subcircuit
                                              # name from a library); forbidden for the other kinds
       params: Dict[str, float] = {}          # instance parameters, e.g. {"area": 2.0}, {"Nx": 1}

   class VerilogA(_M):
       module: str                            # NAME; the source must contain "module <module>"
       source: str                            # the Verilog-A text, not empty

   class Model(_M):
       name: str                              # NAME; unique among models (case-insensitive)
       kind: Literal["diode", "npn", "pnp", "va"]
       params: Dict[str, float] = {}
       va: Optional[VerilogA] = None          # required when kind == "va", forbidden otherwise

   class Library(_M):
       path: str                              # a model library file; "~" is expanded when it is run
       section: Optional[str] = None          # ".lib path section" when given, ".include path" otherwise

   class Op(_M):        kind: Literal["op"] = "op"
   class DcSweep(_M):   kind: Literal["dc"] = "dc"; source: str; start: float; stop: float; step: float
                        # source: a vsource or isource; step != 0 with the sign of stop - start;
                        # 2 <= number of points <= 100001
   class AcSweep(_M):   kind: Literal["ac"] = "ac"; points_per_decade: int; f_start_Hz: float; f_stop_Hz: float
                        # 1 <= points_per_decade <= 1000; 0 < f_start_Hz < f_stop_Hz
   class Transient(_M): kind: Literal["tran"] = "tran"; step_s: float; stop_s: float
                        max_step_s: Optional[float] = None      # 0 < step_s < stop_s; max_step_s > 0 when given

   class CircuitNumerics(_M):
       reltol: float = 1e-6                   # 1e-9 <= reltol <= 1e-2
       abstol_A: float = 1e-15                # > 0
       vntol_V: float = 1e-9                  # > 0
       gmin_S: float = 1e-15                  # >= 0

   class CircuitProblem(_M):
       spec_version: str = "1.0"
       kind: Literal["circuit"] = "circuit"
       title: str = "simws circuit"
       elements: List[Element]                # >= 1; at least one element has the node "0"
       models: List[Model] = []
       libraries: List[Library] = []
       analysis: Union[Op, DcSweep, AcSweep, Transient]          # chosen by "kind"
       probes: List[str]                      # >= 1; "v(<node>)", "v(<node>,<node>)" or "i(<vsource name>)";
                                              # every node appears in an element, every i() names a vsource
       temperature_C: float = 27.0            # > -273.15
       tnom_C: float = 27.0                   # > -273.15: the temperature at which model parameters are given
       numerics: CircuitNumerics = CircuitNumerics()
       def problem_id(self) -> str: ...
       def with_changes(self, changes) -> "CircuitProblem": ...
   ```

   Further rules: every diode, bjt or va element names a Model of kind diode, npn or pnp, va respectively (a bjt
   takes npn or pnp); a subckt element needs at least one Library. `simws.spec.family(problem)` returns
   `"circuit"` for a `CircuitProblem`.
2. **Library (`simws.spec.library`).** `SHOCKLEY_VA`: the Verilog-A text of an ideal diode, module
   `shockley_diode`, parameters `is` (A, default 1e-14) and `n` (default 1), `I(a, c) <+ is * (limexp(V(a, c) /
   (n * $vt)) - 1)` (the Shockley equation, W. Shockley, Bell Syst. Tech. J. 28, 435, 1949).
   `rc_lowpass(R_ohm=1e3, C_F=1e-9, analysis="ac", points_per_decade=10, f_start_Hz=10.0, f_stop_Hz=1e8,
   step_s=1e-8, stop_s=1e-5)`: source "vin" between node "in" and "0", resistor "r1" from "in" to "out",
   capacitor "c1" from "out" to "0", probe "v(out)"; for "ac" the source has ac 1 and DC 0; for "tran" it has
   pulse v1 0, v2 1, delay 0, rise and fall 1e-12 s, width 1 s, period 2 s, and the transient has max_step_s =
   step_s. `divider(R1_ohm=1e3, R2_ohm=3e3, V=1.0)`: "vin" on node "in", "r1" from "in" to "mid", "r2" from "mid"
   to "0", op, probes "v(mid)" and "i(vin)". `diode_iv(IS_A=1e-14, N=1.0, start=0.30, stop=0.65, step=0.05,
   model="builtin", temperature_C=27.0, tnom_C=27.0)`: source "vin" on node "a", diode "d1" from "a" to "0" with
   model "dmod" (kind diode, params {"is": IS_A, "n": N}) when model is "builtin", or kind va with `SHOCKLEY_VA`
   (params {"is": IS_A, "n": N}) when model is "va"; dc sweep of "vin"; probe "i(vin)".
3. **Engine layer (`simws/engines/base.py`, additive).**

   ```python
   class CircuitResult(BaseModel):            # forbids unknown fields
       engine: str
       problem_id: str
       analysis: Literal["op", "dc", "ac", "tran"]
       axis_name: str = ""                    # dc: the swept source; ac: "f"; tran: "t"; op: ""
       axis_unit: str = ""                    # "V" or "A" (dc), "Hz", "s"; "" for op
       axis: List[float] = []
       signals: Dict[str, List[float]]        # probe (as written in the description) -> values (real part for ac)
       signals_imag: Dict[str, List[float]] = {}     # ac only
       units: Dict[str, str]                  # probe -> "V" or "A"
       flags: List[str] = []
       diagnostics: Dict[str, Any] = {}

   class CircuitCapability(BaseModel):        # forbids unknown fields; every list holds the ALLOWED values
       engine: str; env: str; description: str; in_process: bool
       element_kinds: List[str]; model_kinds: List[str]; analyses: List[str]; max_elements: int
       tested_element_kinds: List[str]; tested_analyses: List[str]
       tested_temperature_C: Tuple[float, float]
       notes: str = ""
       def unsupported(self, problem) -> List[str]: ...
       def outside_tested_range(self, problem) -> List[str]: ...
   ```

   `unsupported`: reasons starting with `"element_kinds:"`, `"model_kinds:"`, `"analyses:"`, `"max_elements:"`.
   `outside_tested_range` (bounds included): `"element_kinds:"` when an element kind is not tested,
   `"analyses:"`, `"temperature_C:"`, and `"libraries:"` whenever the problem uses a library (its content is not
   part of the description, so it is never "tested"). Registry: `ENGINES` gains
   `"ngspice": "simws.engines.ngspice_engine"`, `FAMILY = "circuit"`, adapter "circuit.ngspice"; the A7.3 rules
   on families apply.
4. **ngspice translation (`simws/engines/ngspice_engine.py`; `FAMILY = "circuit"`).** ngspice 47 built with OSDI
   (Holger Vogt, Marcel Hendrix, Paolo Nenzi, Dietmar Warning, "Ngspice User's Manual, Version 47"), and
   OpenVAF-Reloaded `osdi_0.3-31-gf47d557` (P. Kuthe, A. Buermen et al.; OSDI 0.3 output, which every ngspice
   from 39 on loads), both run as executables. Where they are: the conda env prefix in `SIMWS_SPICE_PREFIX`,
   else `<envs_root>/spice`; `<prefix>/bin/ngspice` and `<prefix>/bin/openvaf`. Module functions: `available()`
   (ngspice executable present), `openvaf_available()`, `unsupported(problem)`, `build_netlist(problem,
   osdi=None) -> str` (pure; `osdi` maps a Verilog-A module name to its compiled file), `compile_va(va, cache_dir)
   -> Path`, `parse_raw(path) -> dict`, `run(problem, workdir) -> CircuitResult`.
   Capability: element_kinds all nine; model_kinds all four; analyses ["op", "dc", "ac", "tran"];
   max_elements 10000; tested_element_kinds ["resistor", "capacitor", "vsource", "diode", "va", "subckt"];
   tested_analyses ["op", "dc", "ac", "tran"]; tested_temperature_C (26.85, 27.0) (the cases of R30 to R36);
   env "spice"; in_process False.
   Netlist: the title line; one `.lib "<path>" <section>` or `.include "<path>"` per library (path expanded);
   one line per element, its SPICE name being a kind letter, an underscore and the description name (`R_r1`,
   `C_`, `L_`, `V_`, `I_`, `D_`, `Q_`, `N_` for va, `X_` for subckt), then the nodes, then the value, model or
   subcircuit name, then `key=value` for each instance parameter; a source writes `DC <value>`, `AC <ac>` when
   given, and `PULSE(v1 v2 delay rise fall width period)` or `PWL(t1 v1 t2 v2 ...)`; one `.model <name>
   <D|NPN|PNP|module>(key=value ...)` per model; `.options reltol=.. abstol=.. vntol=.. gmin=.. temp=..
   tnom=..` from the numerics and temperatures; a `.control` block with `pre_osdi <file>` per Verilog-A model,
   `set filetype=ascii`, the analysis (`op`; `dc V_<src> start stop step`; `ac dec n f_start f_stop`;
   `tran step stop 0 max_step`), `write result.raw` with the probes translated to ngspice names (`v(node)`,
   `v(a,b)`, `i(V_<src>)`), then `.endc` and `.end`.
   Verilog-A models are compiled by OpenVAF into a cache `<SIMWS_CACHE or ~/.cache/simws>/osdi/<key>/` where the
   key is the SHA-256 of the OpenVAF identity (the SHA-256 of its executable) and the source; a second run with
   the same source reuses the file (diagnostics `osdi_cached` True). `run` raises `EngineNotAvailable` when
   ngspice, or (for a Verilog-A model) OpenVAF, is missing; `Unsupported` as the other engines; and
   `RuntimeError` with the last 20 lines of the ngspice log when ngspice fails or writes no raw file. Current
   probes keep the SPICE sign: `i(<vsource>)` is the current that enters the source at its first node and
   leaves at its second. Every result whose problem uses a library carries a flag
   `"circuit:external_library: <expanded path> sha256 <first 12 hex digits>"` per library.
5. **Adapter `circuit.ngspice`** (`simws/adapters/circuit.py`: base `CircuitAdapter`, subclass
   `CircuitNgspice`; manifest version "1.0", env "tcad", inputs ["problem"], citations "ngspice47" and
   "openvaf"). Invalid description, refused problem and missing engine as in A7.5. Quantities, one per probe:
   `circuit.v.<node>` or `circuit.v.<node1>-<node2>` (unit V), `circuit.i.<source>` (unit A), names in lower
   case; scalars for op; for dc, ac and tran an axis `{"name": <dc source name in lower case> | "f" | "t", "unit":
   "V" | "A" | "Hz" | "s"}`; for ac four quantities per probe with the suffixes `.re`, `.im`, `.mag` (same unit)
   and `.phase` (rad, numpy.angle). Plus the declared output `circuit.n_points` (dimensionless: the number of
   analysis points; 1 for op). No uncertainty is attached (a deterministic solve; its error is set by the
   numerics); the result carries the flag `"budget:none"`. Self-check (status `check_failed`): a non-finite
   value; for dc, a number of points different from the description's. Each `outside_tested_range` string s
   gives `"extrapolated:" + s`.
6. **Device to circuit, first link (`simws/chain/diode.py`).** `fit_shockley(V, J, T_K)` fits ln J = ln Js +
   q V / (N k T) by least squares over all given points (V in volts, J in A/m^2, all J > 0; CODATA 2018 k and
   q) and returns `{"Js_A_m2", "N", "T_K", "max_rel_residual"}`, the residual being max |J_fit / J - 1|.
   `circuit_from_fit(fit, area_m2=1e-12, start, stop, step)` returns `diode_iv(IS_A = Js x area, N, start,
   stop, step, model="builtin", temperature_C = tnom_C = T_K - 273.15)`. Choosing the area is a modelling choice
   (currents scale with it); the temperature must equal the device's, and `tnom_C` must equal it too, or ngspice
   rescales IS with temperature.
7. **Limits, stated.** No noise, distortion, sensitivity or pole-zero analyses; one analysis per problem; AC
   phase 0 only; no subcircuit definitions in the description itself (only from libraries). Trust stays
   "unverified": R30 to R34 compare with closed forms (code verification of the translation), and no second
   circuit engine is connected.
8. **Acceptance (tolerances fixed here, before any of these runs).** Library problems with their defaults
   unless stated. Closed forms use CODATA 2018 k and q.
   R30 (divider): `divider()`: v(mid) within 1e-9 V of 0.75 V and i(vin) within 1e-12 A of -2.5e-4 A.
   R31 (RC, AC): `rc_lowpass()`: at every point the frequency is within 1e-9 relative of 10^(1 + k/10) Hz,
   |v(out)| within 1e-6 relative of 1/sqrt(1 + (2 pi f R C)^2) and its phase within 1e-6 rad of
   -atan(2 pi f R C).
   R32 (RC, transient): `rc_lowpass(analysis="tran")`: at every output time t, |v(out) - (1 - exp(-t / (R C)))|
   <= 1e-3 V.
   R33 (built-in diode): `diode_iv()`: at every point -i(vin) within 1e-4 relative of IS (exp(q V / (N k T)) -
   1), T = 300.15 K.
   R34 (Verilog-A diode through OpenVAF): `diode_iv(model="va")`: the same, within 1e-4 relative.
   R35 (device to circuit): `pn_diode_1d(V=[0.30, 0.35, ..., 0.60])` through `device.devsim`; `fit_shockley` on
   its anode current density; `circuit_from_fit(fit, start=0.30, stop=0.60, step=0.05)` through
   `circuit.ngspice`: at every point -i(vin) within 1e-4 relative of Js x area x (exp(q V / (N k T)) - 1).
   Reported, not gated: N, Js, and the largest relative difference between the circuit current / area and the
   DEVSIM current density (how well an ideal Shockley diode describes this device).
   R36 (foundry model, regression; skipped when the IHP SG13G2 PDK is not at
   `~/sim_workspace/ihp_sg13g2_pdk`): the legacy Gummel set-up of `benchmarks/ihp_npn13g2_baseline` (library
   `cornerHBT.lib` section `hbt_typ`, subckt `npn13G2` with Nx = 1, collector and base at 0 V, emitter swept
   from -0.35 V to -1.02 V in -0.005 V steps): the peak of I_C / I_B over 1 uA < I_C < 5 mA within 1 % of 800.17,
   and V_BE at the point with I_C nearest 1 uA within 0.0051 V of 0.665 V (the stored legacy result, made with
   another ngspice version and the same model; a regression check, not an independent one).
   R37 (cache and repeatability): running `diode_iv(model="va")` twice gives identical signals, and the second
   run reports `osdi_cached` True.
   Why these tolerances: R30, R31 and R33 to R35 are solved to reltol 1e-6 and differ from the closed forms only by
   the solver tolerance and the constants of 0d (at most 1e-5 relative at 0.65 V); a wrong element value, unit,
   temperature or sign fails them by far more. R32 is a time integration: 1e-3 V is 0.1 % of the step, while a
   1 % error in R C gives 3.7e-3 V. R36 allows the change between ngspice versions, not a change of model.
9. **Laptop install.** `tools/install_spice.sh` creates a conda env `spice` (conda-forge c-compiler,
   cxx-compiler and make), downloads ngspice 47 from SourceForge (SHA-256
   894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f), builds it with `--enable-osdi` into that
   env, and puts OpenVAF-Reloaded `osdi_0.3-31-gf47d557` from https://fides.fe.uni-lj.si/openvaf/download/
   (SHA-256 9bc343d52e00529ddba452b90358cdd14da14ec7437d75dec2dbd621eda54bc8) into the env as `openvaf`. It does
   not touch the `eda` or `tcad` envs or the OpenVAF already in the workspace. Engine tests are skipped where
   ngspice is not available.
10. **Correction during implementation (18 Nov, before any acceptance run).** A9.4 said every model is written
    `.model <name> <type>(key=value ...)`. For an OSDI (Verilog-A) model ngspice 47 then loses the first parameter:
    a smoke run (not an acceptance case: is = 2e-14 A, n = 1.1, 0.4 V) gave the current of the default
    is = 1e-14 A, and `showmod` listed is = 1e-14. Written without parentheses, `.model <name> <module> key=value
    ...`, both parameters arrive. Verilog-A models are written that way; built-in models keep the parentheses.
11. **After the checker's report (18 Nov).** The checker wrote 445 tests from the text of A9 alone; 431 passed,
    R36 was skipped (no IHP PDK on the build machine). Acceptance: R30 to R34 and R37 passed (worst differences:
    R30 0; R31 |H| 5.6e-16, phase 2.2e-16 rad; R32 2.9e-6 V; R33 8.5e-6; R34 2.6e-6, all at 0.65 V; R37 identical
    signals, cache used on the second run).
    **Recorded FAIL: R35.** The circuit current differs from the fitted Shockley diode by 1.84e-3 at 0.30 V
    (gate 1e-4): ngspice puts gmin (1e-15 S, the description's default) across every junction, and with
    IS = Js x area = 2.6e-18 A the gmin current (gmin x 0.3 V = 3.0e-16 A) is 0.18 % of the diode current. The
    reasoning in A9.8 left gmin out. The gate is unchanged; R35 stays in the suite, run as specified, as an
    expected failure. Reported values of R35: N = 1.0514, Js = 2.6308e-6 A/m^2; an ideal Shockley diode differs
    from the DEVSIM diode by up to 16 % over 0.30 to 0.60 V (model-form error).
    **Change made because of the FAIL (after the result, so listed here):** `circuit_from_fit` takes
    `gmin_S: float = 0.0` and sets the problem's `numerics.gmin_S` to it, so the circuit holds the fitted diode
    and nothing else. *Post hoc* R35b: the same link with that default, same 1e-4 gate (the checker's diagnostic
    run gave 7.5e-6).
    **Defects fixed:** the dc axis name is now put in lower case by the adapter (it passed the description's
    case through); a non-finite probe value made the store refuse the result before the self-check could mark
    it, so such a probe is now left out of the quantities and the self-check fails the result (`check_failed`,
    message "<probe>: non-finite values"); `compile_va` returns the path, as A9.4 says (it returned a pair).
    **Correction to the text, chosen by the lead:** `fit_shockley`'s `max_rel_residual` compares the data with
    the fitted line, J_fit = Js exp(q V / (N k T)), without the -1 (the -1 changed it by 1e-5 on exact data).
    **Clarifications:** the number of dc points is round((stop - start) / step) + 1. `parse_raw` returns
    {"plotname", "complex", "names" (lower case, in file order), "types", "values" (one list per point, one
    entry per variable; complex numbers for an AC plot)}. `compile_va(va, cache_dir)`: `cache_dir` replaces the
    cache root `<SIMWS_CACHE or ~/.cache/simws>/osdi`, and the file is `<cache_dir>/<key>/<module>.osdi` with
    key = SHA-256 of (SHA-256 of the executable, a newline, the source). A transient without `max_step_s` uses
    `step_s` as the maximum step. `CircuitResult.axis_name` for dc is the description's source name; the
    adapter writes it in lower case. `problem_id` uses the same recipe as `Problem.problem_id`.
12. **On the owner's laptop (18 Nov, update 7, commit cd6670f there).** The same acceptance numbers as on the
    build machine for R30 to R35, R35b and R37, and R36 ran there (the IHP SG13G2 PDK is installed): peak beta
    800.170 against the stored 800.17 (4.7e-7 relative, gate 1 %), V_BE at the point nearest 1 uA 0.6650 V
    against 0.665 V (gate 0.0051 V). PASS: ngspice 47 with OSDI reproduces the legacy npn13G2 Gummel result of
    the older ngspice, so the foundry model path carries over. Fast suite there: 1292 passed, 19 skipped,
    3 xfailed.

### A10 (18 Nov 2026, lead): single event to circuit; critical charge and upset probability

The second link between families (the first is A9.6): the energy a particle leaves in a sensitive volume (A8)
becomes a current pulse injected into a circuit node (A9); bisection over the pulse charge finds the critical
charge of that node, and the single-event histogram gives the probability that one particle deposits at least
that much. Owner's instruction: "lets do 1 and 2", 18 Nov (1 = this link).

0. **What the lead had already seen.** Only closed-form numbers, computed from the formulas in 4 before any run:
   the references of R42 to R44. No circuit or single-event run of any case below.
1. **Module `simws/chain/see.py`.** Constants and functions:
   `PAIR_ENERGY_SI_EV = 3.66`: mean energy per electron-hole pair in silicon, F. Scholze, H. Rabus, G. Ulm,
   J. Appl. Phys. 84(5), 2926-2939, 1998, doi 10.1063/1.368398 (measured 3.66 +- 0.03 eV for photons of 50 to
   1500 eV; the abstract states no temperature; used here as the room-temperature value).
   `charge_C(edep_eV, pair_energy_eV=PAIR_ENERGY_SI_EV)` = edep_eV / pair_energy_eV x q (q = 1.602176634e-19 C)
   and its inverse `edep_eV(charge_C, pair_energy_eV=...)`; both accept arrays.
   `double_exponential(charge_C, tau_rise_s, tau_fall_s, delay_s=0.0, n_points=401) -> Pwl`: the current pulse
   of G. C. Messenger, "Collection of charge on junction nodes from ion tracks", IEEE Trans. Nucl. Sci. 29(6),
   2024-2031, 1982: I(t) = Q / (tau_f - tau_r) (exp(-(t - d) / tau_f) - exp(-(t - d) / tau_r)) for t >= d.
   Rules (ValueError otherwise): charge > 0, 0 < tau_rise < tau_fall, delay >= 0, 50 <= n_points <= 5000.
   Times t_j = d + 20 tau_f (j / (n - 1))^2, j = 0 .. n-1 (dense where the pulse is steep); values from the
   formula, the last one set to 0, then all multiplied by one factor so that the trapezoidal integral of the
   PWL equals Q to rounding. The time constants are inputs: they come from TCAD or measurement, never from
   this module.
   `inject(problem, node, pulse, polarity="discharge", name="set") -> CircuitProblem`: a copy of a transient
   CircuitProblem with an isource `name` carrying `pulse` (DC 0); "discharge" puts it from `node` to "0" (it
   removes charge from the node, as a strike collected at a node held high), "charge" from "0" to `node`;
   `v(<node>)` is added to the probes when missing. ValueError when the analysis is not a transient, the node is
   not in the circuit, or an element already has that name.
   `critical_charge(hub, problem, node, threshold_V, tau_rise_s, tau_fall_s, q_low_C, q_high_C, delay_s=0.0,
   polarity="discharge", rel_tol=1e-3, max_runs=60) -> dict`: an upset is the node voltage reaching the
   threshold during the transient (discharge: the minimum of v(node) <= threshold; charge: the maximum >=
   threshold). Each trial is a `circuit.ngspice` hub job of `inject(problem, node, double_exponential(q, ...))`.
   It first runs q_low (must not upset) and q_high (must upset), else ValueError; then bisects until
   (q_hi - q_lo) / q_hi <= rel_tol. Returns {"qcrit_C": (q_lo + q_hi) / 2, "q_lo_C", "q_hi_C", "runs",
   "result_ids", "node", "threshold_V", "polarity", "tau_rise_s", "tau_fall_s"}. A failed hub job raises
   RuntimeError with its message.
   `upset_probability(see_result, qcrit, pair_energy_eV=PAIR_ENERGY_SI_EV) -> dict`: `see_result` is a stored
   `see.*` result, `qcrit` the dict above. E_crit = edep_eV(qcrit_C); P = the result's `see.p_above` linearly
   interpolated in the threshold at E_crit (the exact value when the events are spread evenly inside a bin);
   P_low and P_high the same at edep_eV(q_hi_C) and edep_eV(q_lo_C); sigma = sqrt(p (1 - p) / N), p =
   (P N + 1) / (N + 2), N the result's `n_events`. Returns {"P", "P_low", "P_high", "sigma", "E_crit_eV",
   "n_events", "flags"}; flags hold `"upset:outside_histogram"` when E_crit is outside the histogram's range
   (then P is 1 below it and the overflow fraction above it).
2. **Limits, stated.** All charge generated in the SV is collected at one node (the SV model's assumption; no
   funnelling, diffusion or collection efficiency); the pulse shape is the double exponential with given time
   constants; the critical charge is for one node, one pulse shape and one threshold; P is per particle that
   crosses the SV the way the single-event problem's source does (G4SEE's point source: always through the
   SV's centre). A cross-section needs the fluence and area on top. Trust stays "unverified".
3. **Tests by the checker** cover the functions above from this text (fake hub where useful).
4. **Closed form used as reference (R42, R43).** A node held at Vdd through R with C to ground, the pulse
   removing charge: C du/dt = I(t) - u / R for the droop u = Vdd - v. For the Messenger pulse with tau = R C:
   u(t) = Q / (C (tau_f - tau_r)) [h(tau_f) - h(tau_r)], h(a) = a tau (exp(-t / tau) - exp(-t / a)) / (tau - a).
5. **Acceptance (tolerances fixed here, before any run).** Circuit "rc_node(C, R)": vsource "vdd" 1.0 V on node
   "vdd", resistor "r1" (vdd-n) R, capacitor "c1" (n-0) C, transient step 0.1 ps, stop 300 ps, max step 0.1 ps,
   default numerics; pulse tau_r = 1 ps, tau_f = 10 ps, delay 0; threshold 0.5 V, discharge.
   R40 (conversion): `charge_C(1e6)` equals 1e6 / 3.66 x 1.602176634e-19 C to 1e-12 relative (4.37753e-14 C),
   and `edep_eV(charge_C(x))` returns x to 1e-12 relative.
   R41 (pulse): for Q = 2 fC the PWL's trapezoidal integral equals Q to 1e-12 relative; its largest value is within
   1 % of the analytic peak 1.548527e-4 A, at a time within 2 % of the analytic peak time 2.558428 ps.
   R42 (charge reaches the node): rc_node(10 fF, 10 kOhm) with a 2 fC pulse: the minimum of v(n) is within 1 % of
   the closed-form droop (closed form: 0.8452308 V, droop 0.1547692 V) of that droop.
   R43 (critical charge): rc_node(10 fF, 10 kOhm), q_low 1 fC, q_high 20 fC: `qcrit_C` within 1 % of the closed form
   6.461234e-15 C, and (q_hi - q_lo) / q_hi <= 1e-3.
   R44 (end to end, needs G4SEE): `see_si_slab("alpha", 5.0, sv_thickness_um=10.0)` through `see.g4see`;
   critical charges of rc_node(60 fF, 100 kOhm) (closed form 3.032237e-14 C; q_low 10 fC, q_high 100 fC) and
   rc_node(200 fF, 100 kOhm) (closed form 1.003812e-13 C; q_low 50 fC, q_high 300 fC); then P >= 0.999 for the
   first and P <= 0.001 for the second (the alphas leave about 1.61 MeV, 70.6 fC). Also P_low <= P <= P_high in
   both.
   Why these tolerances: R42 and R43 differ from the closed form by the PWL sampling of the pulse, the time
   integration and the bisection (1e-3); 1 % leaves room for those and catches a wrong sign, unit or charge.
   R44 checks the chain end to end against a case whose answer is not in doubt.
6. **Amendment before any run (18 Nov, lead).** With 401 points the grid spacing near the pulse peak (t about
   2.56 ps for tau_r = 1 ps, tau_f = 10 ps) is about 0.11 ps, so the PWL peak time could miss the analytic one by
   up to about 2.2 %, more than R41 allows, by construction rather than by error. The default of
   `double_exponential` becomes `n_points=2001` (spacing near the peak about 0.022 ps). Tolerances unchanged.

### A11 (18 Nov 2026, lead): a different-physics device engine, electron energy balance in DEVSIM ("devsim_eb")

Drift-diffusion assumes electrons are always at the lattice temperature. Where the field is strong they are not:
they heat up, their mobility falls and, over short distances, their velocity overshoots. A second device engine
that solves the same Problem with an electron energy balance says how much a drift-diffusion result depends on
that assumption (model-form error). It runs on the same code (DEVSIM), so it is a different-physics check, never
a verification. Owner's instruction: "lets do 1 and 2", 18 Nov (2 = a second engine for trust).

0. **What the lead had already seen.** Nothing of this engine: no run of any kind. The silicon saturation velocity
   that simws already uses (Canali 1975, `simws.tcad.physics.saturation_velocity`) is 1.07e7 cm/s for electrons
   at 300 K.
1. **Model (electrons; holes stay isothermal drift-diffusion).** Unknown: the electron temperature Tn (K) at each
   node, with u = k Tn / q.
   Current: J_n = q mu_n (grad(n u) - n grad(psi_c)), psi_c the electron band potential used by the
   drift-diffusion translator (potential plus the band-gap-narrowing shift).
   Energy flux: S_n = -(5/2) u J_n - kappa_n grad(Tn), kappa_n = (5/2) (k^2 / q) mu_n n Tn.
   Energy balance (steady state): div(S_n) = J_n . E - W_n, W_n = (3/2) n k (Tn - T) / tau_w.
   Relaxation times of G. Baccarani and M. R. Wordeman, "An investigation of steady-state velocity overshoot in
   silicon", Solid-State Electron. 28, 407-416, 1985, without the kinetic-energy term (an energy-balance, not a
   full hydrodynamic, model): mu_n = mu_0 T / Tn and tau_w = (3 k mu_0 / (2 q v_s^2)) T Tn / (T + Tn), so
   W_n = n q v_s^2 (Tn^2 - T^2) / (mu_0 T Tn). mu_0 is the low-field mobility the drift-diffusion translator uses
   at that node (constant or Masetti), v_s the saturation velocity above. No new parameter is introduced.
   Consequence used for checking (homogeneous material, field E, no gradients): Tn = T sqrt(1 + (mu_0 E / v_s)^2)
   and J_n = q n mu_0 E / sqrt(1 + (mu_0 E / v_s)^2) (the Caughey-Thomas form with beta = 2).
   Left out, stated: the energy carried by recombining carriers, hole heating, impact ionization driven by Tn,
   lattice heating.
2. **Discretization.** On an edge from node 0 to node 1 (length h): u_e = (u_0 + u_1) / 2, Tn_e = (Tn_0 +
   Tn_1) / 2, mu_e = (mu_0(node 0) + mu_0(node 1)) / 2 x T / Tn_e, delta = (psi_c,0 - psi_c,1) / u_e,
   J = q mu_e / h (n_1 u_1 B(-delta) - n_0 u_0 B(delta)) (B the Bernoulli function; for Tn = T it is the
   drift-diffusion translator's Scharfetter-Gummel current), S = -(5/2) u_e J - kappa_e (Tn_1 - Tn_0) / h with
   kappa_e = (5/2) (k^2 / q) mu_e ((n_0 + n_1) / 2) Tn_e; J . E is assembled on the edge volume and W_n on the
   node volume. Ohmic contacts hold Tn = T.
3. **Engine `devsim_eb` (`simws/engines/devsim_eb_engine.py`, family "device", adapter "device.devsim_eb",
   `CODE = "devsim"`).** It translates a Problem exactly as `devsim` does, solves the drift-diffusion equilibrium,
   adds the energy balance, solves the equilibrium again, and then runs the analysis. Capability: dimensions [1];
   materials ["silicon"]; no grading; contacts ["ohmic"]; statistics ["boltzmann"]; mobility ["constant",
   "masetti"]; velocity_saturation [False] (the energy balance gives the field dependence; adding drift-diffusion
   velocity saturation would count it twice); srh ["off", "constant", "scharfetter"]; auger [False, True];
   band_gap_narrowing ["off", "slotboom"]; incomplete_ionization ["off"]; impact_ionization ["off"]; analyses
   ["equilibrium", "dc_sweep"]; tested_temperature_K (300, 300); mesh_study True; env "tcad"; in_process True.
   `EngineResult.diagnostics` adds "eb_vsat_n_cm_s" (the v_s used), "eb_Tn_max_K" (the largest Tn at the last
   point) and "eb_model" ("baccarani_wordeman_1985_energy_balance"). Every result carries the flag
   `"physics:energy_balance_electrons"`. Module function `profiles(problem) -> dict`: runs the problem and returns,
   at its last point, {"x_um", "Tn_K", "n_cm3", "potential_V"} node by node (1D, increasing x).
   The adapter `device.devsim_eb` (version "1.0") is `device.devsim`'s adapter with this engine (same quantities,
   units, mesh study, flags). Library: `simws.spec.library.n_bar_1d(N_D_cm3=1e16, length_um=2.0, mu_n=1000.0,
   mu_p=400.0, nodes=2001, T=300.0, V=None)`: uniform n-type silicon, contacts "left" (x_min) and "right"
   (x_max), constant mobilities, SRH, Auger and band-gap narrowing off, Boltzmann statistics, uniform mesh; a dc
   sweep of "right" over V (0.05 V steps) or equilibrium when V is None; numerics as `pn_diode_1d`.
4. **Independence (amends A2.3).** An engine module may set `CODE` (its name when absent); `simws.engines.code(name)`
   returns it. `compare_results` counts two results as independent only when their engines' codes differ, so a
   comparison of `devsim` with `devsim_eb` never verifies either. Existing pairs are unchanged (devsim and charon
   have different codes).
5. **Acceptance (tolerances fixed here, before any run).**
   R50 (equilibrium): `n_bar_1d()` and `pn_diode_1d()` at equilibrium through `devsim_eb`: max |Tn - T| / T <=
   1e-6 over all nodes (from `profiles`) and every current below the A3 floor.
   R51 (low-field limit): `n_bar_1d(V=[0.04])` (200 V/cm): the `devsim_eb` current within 1e-3 relative of the
   `devsim` current of the same problem.
   R52 (hot electrons, homogeneous): `n_bar_1d(V=[2.0, 6.0, 20.0])` swept to each value in turn (three problems
   with V=[2.0], V=[6.0], V=[20.0]): at the node nearest the middle, with E the central-difference field there
   and n the density there, Tn within 1 % of T sqrt(1 + (mu_0 E / v_s)^2) and the current density within 1 % of
   q n mu_0 E / sqrt(1 + (mu_0 E / v_s)^2) (mu_0 = 1000 cm^2/Vs, v_s from the diagnostics).
   R53 (mesh): `n_bar_1d(V=[6.0])` through `device.devsim_eb` at paper fidelity: the mesh error bar of
   dc.I.right below 1 % of its value.
   R54 (different-physics check, reported): `pn_diode_1d(V=[0.30, 0.35, ..., 0.60])` through `device.devsim` and
   `device.devsim_eb`: the largest relative difference of dc.I.anode is printed, not gated; recorded with
   `compare_results(..., tolerance=0.01)`, the comparison has independent False, and the trust level of the
   `devsim` result is the same before and after it is recorded.
   Why these tolerances: R50 and R51 are limits in which the two models agree exactly, so the gates sit just
   above the solver tolerance; R52 compares with a closed form valid in the middle of a long bar (the energy
   relaxation length, about v_s tau_w, is some 10 nm against 1 um to the contacts); 1 % allows the
   discretization and catches a wrong factor in any term (a factor 5/2 or 3/2 misplaced changes Tn by more).
7. **After the checker's report (18 Nov).** 61 tests written from A10 alone; 58 passed, then all after one fix.
   **All acceptance checks passed:** R40 exact; R41 integral 2.2e-16, peak value 8.3e-7, peak time 1.8e-3
   (2001 points); R42 minimum of v(n) 0.8452317 V, droop off by 5.7e-6 relative; R43 qcrit 6.462036e-15 C
   (+1.2e-4) with a bracket of 7.2e-4 after 14 runs; R44 (10 000 alphas, mean deposit 1.61364 MeV = 70.64 fC):
   P = 1.000000 for the 60 fF node (qcrit 3.0314e-14 C) and P = 0.000400 for the 200 fF node (qcrit
   1.0038e-13 C), P_low <= P <= P_high in both.
   **Defect fixed:** `upset_probability` set the outside-histogram flag when the bracket ends q_lo or q_hi fell
   outside the histogram, not only E_crit; now only E_crit sets it (P_low and P_high keep their values: 1 below
   the range, the overflow fraction above).
   **Clarifications:** `see_result` is a stored result envelope (`hub.result(id)`), N its diagnostics'
   `n_events`; between the last threshold and high_MeV, P is interpolated to the overflow fraction at
   high_MeV; `critical_charge` raises ValueError when q_low_C >= q_high_C or the polarity is unknown, and
   RuntimeError when max_runs is reached; node names and probes are case-insensitive. "About 1.61 MeV, 70.6 fC"
   in R44 means the R22 deposit, 1.6137 MeV.
6. **After the checker's report (18 Nov).** 46 tests written from A11 alone; 44 passed.
   Passed: R50 (|Tn - T| and every current exactly 0 for the bar and the diode); R51 (1.7e-4 against 1e-3); R52 at
   2 V (Tn 8.0e-6, current 5.7e-6 against 1e-2); R53 (mesh error bar 6.0e-7 of the current, `passed` True);
   R54 (largest difference 0.22 % at 0.60 V, rising from 0.09 % at 0.30 V; independent False; the devsim result's
   trust unchanged). Registry, capability, refusals, `profiles`, independence and the adapter as stated.
   **Recorded FAIL: R52 at 6 V and 20 V.** Tn 909.97 K against 920.43 K (1.14 %) at 6 V; 2736.8 K against
   2820.5 K (3.0 %) and the current 1.29 % off at 20 V. The gate is unchanged; both cases stay in the suite as
   expected failures. *Post hoc diagnosis* (the checker's independent script, after the result): the solution is
   self-consistent (rebuilt from the profiles, the discrete energy balance leaves a residual of 1e-10 of J.E), but
   the bar is not homogeneous: electrons enter cold at the cathode, their mobility falls as they heat, and the
   bar becomes space-charge limited (n = 1.06 N_D and 1.46 N_D in the middle; the field changes by 64 % and
   142 % across it; the currents exceed q N_D v_s). R52's premise rested on the energy relaxation length alone
   and missed this space-charge gradient. So the failure says the check was set up wrongly at those biases, not
   that the model or code is wrong; the 2 V case, where the premise holds, agrees to 1e-5.
   **Changes:** the plan summary marks a checker on the primary's code as "(same code: model-form check only)";
   section 4.3 still lists it among the checkers (a plan is about capability), and A11.4 keeps it from verifying.
   The energy-balance bias ramp saves and restores the node solutions around a failed step (a diverged Newton
   step can leave Tn overflowing); the solver's "matrix is singular" lines in the log are such failed steps,
   each followed by a smaller one. **Clarifications:** "the A3 floor" in R50 is the A3.2 floor, 1e-6 A/cm^2;
   R54 compares a = devsim with b = devsim_eb; R51 holds for both contacts.
7. **On the owner's laptop (18 Nov, update 8, commit 155f1e7 there).** A10 R40 to R44 and A11 R50, R51, R52 at
   2 V, R53 and R54 passed with the same numbers as on the build machine; R52 at 6 V and 20 V are the recorded
   expected failures. The energy-balance suite took 10 minutes there.

### A12 (18 Nov 2026, lead): electronic stopping from measured data for RustBCA ("empirical")

A7.9 recorded that RustBCA's Lindhard-Scharff (LS) electronic stopping is too weak for boron in silicon and that a
correction factor needs a source other than the range table that failed. This amendment takes the factor from
measured stopping powers and keeps the 1975 range tables of R16 out of its derivation, so R16 can be run again as
a test that the factor did not see. Owner's instruction: "lets do all except 3", 18 Nov (1 = a stopping
correction from sourced data).

0. **What the lead had already seen when this was written.** (a) A7.9: multiplying the LS stopping by 1.6 (a
   post hoc trial, not sourced) brought boron at 100 keV to +1.7 % in Rp. (b) Which data sets the database below
   holds for B, P and As in Si (authors, numbers of points, energy ranges); no stopping value had been divided
   by LS when this section (items 0 to 2) was committed. Items 3 onward were written after the factor of item 2
   was computed and before any RustBCA run with it.
1. **Source.** IAEA Electronic Stopping Power of Matter for Ions database, release 2026-09, the file
   `StoppingPower.csv` from https://www-nds.iaea.org/stopping/download/tuples (downloaded 18 Nov 2026; zip
   SHA-256 0813d4137aa571eb8d8d48e96ed7ca3dd2b7ef10e510d627e7d01029379ba01b, CSV SHA-256
   dacc1c57e39a8187129d9b6593e5aa91276ca317a7f0c7530869446347f9af0c). Citation: C. C. Montanari et al., Nucl.
   Instrum. Methods B 551 (2024) 165336, doi 10.1016/j.nimb.2024.165336. The rows used are copied into
   `simws/data/stopping/iaea_<ion>_si.csv` with the database's own columns, so the derivation can be repeated
   without the network.
2. **Rule (fixed before computing).** For an ion X in silicon:
   (a) rows with projectile X, target Si, phase "s", energy unit MeV/u, energy at most 0.010 MeV/u (10 keV/u:
   the whole slowing-down of the beams R16 tests, boron at 100 keV is 9.1 keV/u), and a reference that is a
   measurement: rows whose `ref_id` is SRIM2013 (a fitted table) are excluded, every other row is kept;
   (b) for each row, r_i = S_i / S_LS(E_i), S_i the row's `stopping_power_converted` in MeV/(mg/cm^2), S_LS the
   value of RustBCA 3.0.0's own `electronic_stopping_cross_sections(Z1, 14, E, M1, 1.0, 1.0)[0]` (eV m^2 per
   atom) converted with the silicon atom mass 28.0855 u, E = (MeV/u value) x M1 x 1e6 eV, M1 the row's
   `ion_isotope` (u);
   (c) weights w_i = 1 / (r_i e_i)^2 with e_i the row's `percentage_error` / 100, or 0.10 where it is empty;
   ck = sum(w_i r_i) / sum(w_i); its standard uncertainty u = sqrt(1 / sum w_i) x max(1, sqrt(chi2 / (n - 1)))
   with chi2 = sum w_i (r_i - ck)^2 (the Birge ratio inflates it when the sets disagree);
   (d) an entry exists only when at least 3 rows from at least 2 publications pass (a); otherwise the ion has no
   empirical stopping in silicon and the mode is refused for it;
   (e) reported with the entry, never used to change it: the leave-one-publication-out values of ck, and the
   slope of r against E / M1 from an unweighted straight-line fit (a constant ck is a model choice; the slope
   says how good a choice it is).
3. **The factor that the rule gives (computed 18 Nov, after items 0 to 2 were committed as 251fce4).** Boron:
   11 rows from 4 publications (Grahmann and Kalbitzer 1976, 4 rows; Hofker et al. 1975, 4; Hoffmann et al.
   1976, 1; Siketic et al. 2010, 2), 0.40 to 9.09 keV/u: **ck = 1.381, u = 0.195**. The sets disagree: chi2 =
   1190 for 10 degrees of freedom (Birge ratio 10.9); Grahmann and Kalbitzer give r of 0.98 to 1.10, the others
   1.40 to 1.55, and Siketic's 0.40 keV/u row gives 3.58. Leave one publication out: 1.761 (without Grahmann),
   1.373 (without Hofker), 1.380 (without Hoffmann), 1.072 (without Siketic). Slope of r: -0.073 per keV/u.
   Phosphorus: no row passes (a), so no entry. Arsenic: 6 rows from one publication (Tan et al. 1988), so no
   entry. So "empirical" exists for boron only, with a large stated uncertainty.
4. **Description.** `ImplantModels.electronic_stopping` becomes `Literal["lindhard_scharff", "empirical"]`,
   default unchanged, so no existing problem id changes (`implant_si("P", 50)` is
   5cfbc9d5e496882805061234f5349fe1 and `implant_si("B", 100)` is 7141925e1068b3eda07be7f312f0690d, before and
   after). "empirical" means: electronic stopping taken from measured data, each engine with its own stated
   source; a result in this mode carries a flag starting `"implant:empirical_stopping:"` that names it.
5. **RustBCA in "empirical" mode.** `simws.radiation.empirical_stopping` implements A12.2 (`entry(ion)`,
   `selected(ion)`, `lindhard_scharff_cross_section_eV_m2`, `lindhard_scharff_MeV_cm2_mg`) from the CSV in
   `simws/data/stopping`. The translator uses mode LOW_ENERGY_NONLOCAL as before and sets every layer's
   `electronic_stopping_correction_factors` to the entry's ck. It refuses (reason starting
   `"electronic_stopping:"`) when the ion has no entry, or when a layer holds anything other than Si (RustBCA
   applies one factor to every species of a layer, and the factor is for silicon). The BORON_FLAG of A7.9 is not
   set in this mode; the flag is `"implant:empirical_stopping: Lindhard-Scharff x <ck> +- <u> from <n> measured
   points (IAEA database, A12); the data sets disagree (Birge ratio <b>)"` with three decimals for ck and u and
   one for b. Diagnostics gain `stopping_ck` and `stopping_ck_u`. `run(problem, workdir=None, ck=None)`: a
   given ck replaces the entry's (for R16s; the hub never passes it). `ImplantCapability` gains
   `tested_ion_symbols_by_stopping: Dict[str, List[str]] = {}`: when the problem's electronic_stopping is a key,
   that list is used instead of `tested_ion_symbols`. It stays empty until R16e has been run.
6. **Acceptance (tolerances fixed here, before any RustBCA run with the factor).** Library problems, 20 000 ions,
   seed 1, ZBL, as A7.7.
   R16e (the held-out table): `implant_si("B", 50)` and `implant_si("B", 100)` with electronic_stopping
   "empirical": `implant.Rp` and `implant.dRp` each within 25 % of the R16 table values (B 50 keV 0.1608 /
   0.0504 um; B 100 keV 0.2994 / 0.0710 um). The gate is R16's, unchanged. Only if all four pass, the RustBCA
   capability gets `tested_ion_symbols_by_stopping = {"empirical": ["B"]}`.
   R16s (sensitivity, reported, no gate): the same two problems with ck - u and ck + u; the spread of Rp says how
   much the disagreement of the data sets matters.
   R17 (the rule, repeated): ck, u, n and the publications recomputed from the CSV files and the text of A12.2
   alone agree with `entry("B")` to 1e-9 relative; `entry("P")` and `entry("As")` are None.
   R18 (the LS formula): `lindhard_scharff_cross_section_eV_m2` equals RustBCA's own
   `electronic_stopping_cross_sections(Z1, 14, E, M1, 1, 1)[0]` to 1e-12 relative for B, P and As at 1, 10 and
   100 keV (skipped where libRustBCA is not installed).
   R19 (refusals and no side effects): "empirical" for P and for As is refused; a boron problem with a layer
   holding O is refused; the default mode gives `depth_um` identical to a run of the same problem before A12
   (the same seed and input dictionary: `build_input` of a default problem has every correction factor 1.0).
7. **Trust.** Neither mode makes a RustBCA result verified (A7.6 still holds); R16e is evidence against a theory
   table that the factor did not see, not a validation against measured ranges.
8. **After the checker's report (18 Nov).** 26 tests written from A12 alone. R17 passed (the checker's own
   recomputation agrees with `entry("B")` to 2e-16; every number of item 3 reproduced), R18 passed (1e-12), R19
   passed. **R16e passed, all four:** B 50 keV Rp 184.2 nm against 160.8 (+14.6 %), dRp 57.4 against 50.4
   (+13.8 %); B 100 keV Rp 334.6 against 299.4 (+11.8 %), dRp 81.0 against 71.0 (+14.0 %). So, as item 6 says,
   the RustBCA capability now has `tested_ion_symbols_by_stopping = {"empirical": ["B"]}`. R16s (reported):
   with ck - u the boron Rp is +23.8 % (50 keV) and +22.8 % (100 keV), with ck + u +6.8 % and +2.8 %; dRp at
   ck - u is +27 % and +31 %, outside 25 %. The disagreement between the measured data sets is therefore the
   largest error left in boron ranges: about +-8 % in Rp.
   **Clarifications (no tolerance involved):** the atomic mass unit is RustBCA's (1.66053906660e-27 kg); the
   keys of the entry are those of the module (`leave_one_out`, `slope_per_keV_u`, `birge_ratio`).

### A13 (18 Nov 2026, lead): a second implant engine on different code, Geant4 ("g4implant")

A7.6 says an implant result stays "unverified" until a second, different implant engine is compared with it.
This engine runs the same ImplantProblem through Geant4 11.2.2 (the toolkit already installed for G4SEE, A8):
different code, a different nuclear-scattering algorithm, and two choices of electronic stopping. Owner's
instruction: "lets do all except 3", 18 Nov (2 = a second implant engine).

0. **What the lead had already seen when this was written** (scouting on 18 Nov, so the reader can judge the
   tolerances). (a) Arsenic at 80 keV into the `implant_si` target, 2000 to 20 000 ions, Lindhard-Scharff
   stopping: Geant4 Rp 70.1 nm with the physics cutoff of item 3 at 10 eV, 62.4 nm at 1 eV, 59.3 at 0.2, 55.8 at
   0.05, 46.1 at 0.01 and 34.2 at 0.002; RustBCA Rp 56.4 nm. Counting the two branches of
   G4ScreenedNuclearRecoil's impact-parameter sampling showed that at 10 eV only the long-flight branch is used,
   at 1 eV 8 % of the collisions take the short-flight branch, at 0.05 eV 37 %; the short-flight branch puts
   more collisions per unit area than there are atoms, which is why the range keeps falling. (b) Geant4's
   electronic stopping in "empirical" mode (G4IonParametrisedLossModel) divided by Lindhard-Scharff: 1.34 for
   boron and 1.05 for phosphorus, constant from 1 to 200 keV (Geant4 scales its lowest table value with the
   square root of the energy); the value is the same in a material built from atom densities and in G4_Si.
   (c) Arsenic at 80 keV with recoils (A14), displacement energy 15 eV: RustBCA 2186 vacancies per ion, mean
   vacancy depth 42.0 nm (100 ions; 2.4 GB of memory for those 100 ions, with the silicon cutoff 1.5 eV);
   Geant4 1889 per ion and 48.8 nm (200 ions, cutoff 10 eV). (d) G. Hobler et al. (SISDEP 1991) fitted boron
   electronic stopping in silicon to channeling data as 1.5 times Lindhard's; noted, not used (A12.2 is fixed).
   No boron or phosphorus range or damage case had been run with this engine.
1. **Program.** `tools/g4implant/g4implant.cc` (simws code, about 350 lines) compiled against Geant4 together
   with TestEm7's G4ScreenedNuclearRecoil.cc and G4LindhardPartition.cc from the Geant4 11.2.2 examples
   (`share/Geant4/examples/extended/electromagnetic/TestEm7` in the env). It reads a keyword input file and
   writes the end point of every ion, in launch order. Geometry: the layers as boxes 2 m wide along y and z
   (laterally infinite for these ranges) in a vacuum world 1 nm thicker on each side; the ion starts 0.5 nm in
   front of the surface. Serial run manager, CLHEP seed = the problem's seed, so a run is repeatable.
   Physics, GenericIon only: G4ScreenedNuclearRecoil (Mendenhall and Weller, NIM B 227 (2005) 420) with the
   problem's screening function (zbl, or mol for "moliere"; Kr-C is refused), no multiple-scattering process
   (as TestEm7 below 100 MeV); production cut 1 km (no delta electrons). Electronic stopping:
   "lindhard_scharff": a continuous process written in g4implant.cc, dE/dx = sum over elements of n_i
   S_LS(Z1, Z2_i, E, M1), S_LS exactly RustBCA's formula (A12.5), integrated exactly over each step, a step
   losing at most 5 % of the energy; Z1 and M1 of the moving atom (Geant4's nuclear mass in u);
   "empirical": G4ionIonisation with G4IonParametrisedLossModel, Geant4's own data and scaling.
   The app writes `# dedx_e <E> keV <value> MeV_cm2_g` header lines for requested energies in both modes.
2. **Install.** `tools/install_g4implant.sh` compiles the program into the g4see env (A8.8) with that env's
   compiler and geant4-config, and installs `<env>/bin/g4implant`. It stops with a message when the env, the
   compiler or the TestEm7 sources are missing. Tests that need it are skipped where it is not installed.
3. **Engine `g4implant`** (`simws/engines/g4implant_engine.py`, family implant, code "geant4", env "g4see",
   not in process; prefix as the G4SEE engine: SIMWS_G4SEE_PREFIX or `<envs_root>/g4see`). Fixed settings:
   physics cutoff 10 eV (G4ScreenedNuclearRecoil's own default, the value its header calls typical, and per
   0(a) the one at which only the branch its sampling was written for is used); recoil cutoff per nucleon =
   the ion's `cutoff_eV` / its mass (so the ion stops below its cutoff) in range mode, A14.3 in damage mode.
   The ion is Geant4's ion (Z, A = round(mass_amu)). Capability: structures ["amorphous"], potentials ["zbl",
   "moliere"], electronic_stopping ["lindhard_scharff", "empirical"], recoils [False, True] (A14), max_layers
   20, max_keV_per_amu 25 (as RustBCA, so both engines take the same problems), tested ranges set after the
   acceptance runs (until then: tested_keV_per_amu (0, 0), tested_target_symbols ["Si"], tested_ion_symbols
   ["none"], so every result is flagged `extrapolated:`). Flags on every result: the amorphous flag of A7 and
   `"implant:geant4_nuclear_cutoff: collisions transferring less than 10 eV are ignored (G4ScreenedNuclearRecoil
   physics cutoff, A13.3)"`; in "empirical" mode `"implant:empirical_stopping: Geant4 G4IonParametrisedLossModel
   (its ion data, scaled with the square root of the energy below the lowest table energy; A13)"`. The surface
   and bulk binding energies are not used by this engine (stated in its notes; A14.3 refuses bulk binding with
   recoils). Diagnostics: geant4 version line, ion mass (u), physics cutoff, recoil cutoff per nucleon,
   material densities. Errors: the program not installed -> EngineNotAvailable with the install command;
   a non-zero exit -> the last lines of its log in the message.
4. **Adapter `implant.g4implant`**: the A7.5 adapter with ENGINE "g4implant" (same quantities, error bars,
   split halves, self-checks), env "tcad" (corrected 18 Nov before any acceptance run: as the G4SEE adapter, the
   adapter runs in the tcad env and calls the program in the g4see env; "g4see" is the engine's env), engine_version "11.2.2", citations geant4_2016 and
   mendenhall2005 (added to the citation list).
5. **Acceptance (tolerances fixed here, before any of these runs).** Library problems, 20 000 ions, seed 1, ZBL,
   unless stated. "Within 4 sigma" as A7.7.
   R60 (repeatability): `implant_si("P", 50, n_ions=2000)` twice gives identical `depth_um`; seed 2 differs.
   R61 (counting): implanted + reflected + transmitted = N in every run; a 0.05 um target under P at 100 keV
   (n_ions=2000) has transmitted ions and the transmitted flag.
   R62 (honest error bars): `implant_si("P", 50)` with seeds 1 and 2 agree in Rp, dRp and lateral straggle
   within 4 sigma.
   R63 (the LS formula in the program): the `dedx_e` lines in "lindhard_scharff" mode for B, P and As in the
   `implant_si` target at 10, 50 and 100 keV equal `lindhard_scharff_MeV_cm2_mg` within 1e-3 relative (the
   ion's nuclear mass differs from the isotope mass by Z electron masses).
   R64 (the published tables, "empirical" mode): B 50, B 100 and P 50 keV: Rp and dRp each within 25 % of the
   R16 table values (R16's gate). Reported without a gate: the same in "lindhard_scharff" mode.
   R65 (the cross-code verification, "lindhard_scharff" mode, the same problem on both engines): B 50, B 100
   and P 50 keV: `implant.Rp` within 10 % and `implant.dRp` within 15 % of RustBCA's, with
   `simws.compare.compare_results` (relative metric) reporting `independent` True. Why 10 % and 15 %: both
   engines then use the same screening function and the same electronic stopping formula, so what remains is
   the collision algorithm (free-flight path, impact-parameter sampling, scattering integral, the 10 eV
   cutoff); a range that two such implementations give differently by more than 10 % is not known to 10 %.
   0(a) shows arsenic at 80 keV is 24 % apart; it is reported, not gated.
   R66 (reported, no gate): Geant4 "empirical" against RustBCA "empirical" for B 50 and 100 keV (different
   stopping data, 1.34 against 1.38 times LS); and B 50 keV with the physics cutoff at 1 eV (run through the
   program directly), to show what the cutoff choice moves.
   R67 (trust): a RustBCA result for which R65's comparison passed and was recorded becomes "verified" by the
   rules of section 6 (A2.3); one whose comparison failed stays "unverified". Nothing is changed in the trust
   rules.
6. **After the acceptance runs.** The tested ranges of item 3 are set to what passed, and the RustBCA
   capability's notes say which ions R65 verified.
7. **After the checker's report (18 Nov).** 31 tests written from A13 alone.
   **A defect, fixed, and the affected checks run again:** the program aborted (Geant4 "Failed root find" in
   G4ScreenedNuclearRecoil) when an ion that had left the target collided with the hydrogen of the vacuum world
   (G4_Galactic). It broke R61 and the 20 000-ion "empirical" runs of B 50 and P 50 keV. Fix: an atom that enters
   the world volume is stopped there and counted as reflected or transmitted by the side it left; no process acts
   on the world's material. Ions that stay in the target are unaffected (the Lindhard-Scharff R65 numbers are
   the same before and after). After the fix, R61 passed (67 implanted, 1 reflected, 1932 transmitted; the hub
   status is outside_validity with the transmitted flag).
   Passed: R60, R61, R62 (0.02, 0.50 and 0.68 sigma), R63 (program LS stopping +1.4e-4 to +1.5e-4 of the formula).
   **R65 (the verification):** B 50 keV Rp +8.5 % (gate 10 %), dRp +10.0 % (15 %): passed; B 100 keV Rp +6.4 %,
   dRp +8.4 %: passed; **P 50 keV recorded FAIL:** Rp 80.6 nm against RustBCA's 71.3 (+13.0 %), dRp 34.2 against
   29.5 (+15.9 %). The gates are unchanged. The two codes agree to 6 to 10 % where electronic stopping decides the
   range (boron) and differ by 13 to 24 % where nuclear collisions matter (phosphorus; arsenic in 0(a)): the
   nuclear-collision algorithms disagree, and nothing here says which is right.
   **R64 recorded FAIL, all three:** Geant4 "empirical" against the tables: B 50 keV Rp +25.6 %, dRp +28.4 %;
   B 100 keV Rp +20.9 % (within), dRp +25.8 %; P 50 keV Rp +31.6 %, dRp +32.3 %. In Lindhard-Scharff mode
   (reported): B 50 +45.9 % / +57.7 %, B 100 +44.7 % / +63.7 %, P 50 +32.8 % / +33.5 %.
   R66 (reported): "empirical" against "empirical", boron: Geant4 deeper by 9.6 % (50 keV) and 8.2 % (100 keV) in
   Rp; the physics cutoff at 1 eV instead of 10 eV moves B 50 keV by -3.5 % in Rp and -3.7 % in dRp.
   **A decision on trust (stricter only, made after the results and so stated here):** R65 would have made the two
   boron Lindhard-Scharff RustBCA results "verified", although they carry `implant:known_failure:` and are 35 %
   too deep: both codes share the too-weak stopping, so their agreement verifies the codes, not the number. The
   trust rules (section 6.1) gain two rules: a result carrying a flag containing ":known_failure:" is never above
   "unverified"; and one failed, independent engine comparison blocks "verified" (as A1.10 does for
   "validated"). R67 is amended accordingly: no RustBCA result of R65 is verified (boron: known failure;
   phosphorus: R65 failed). Today no implant result is verified; that is the honest state.
   **Changes:** tested ranges of g4implant set to what passed (item 6): `tested_keV_per_amu` (4.5, 9.1),
   `tested_ion_symbols` ["B"], and nothing in "empirical" mode (`{"empirical": ["none"]}`); a g4implant boron
   result in Lindhard-Scharff mode carries `implant:known_failure:` as RustBCA's does. The adapter's env is "tcad"
   (item 4, corrected before the runs; the checker had read the earlier text).
   **Clarifications:** the program's input keywords are those of `build_input`'s text (`physics_cutoff_eV` and
   the others in the header of g4implant.cc); status 0 in its output means stopped in the target.

### A14 (18 Nov 2026, lead): damage (vacancy) profiles from both implant engines

A7.6 refused recoil tracking: following every recoil in memory exhausted an 8 GB machine. A14 follows the
recoils in batches and streams them through files, so the memory is bounded by a batch, and counts vacancies.
Owner's instruction: "lets do all except 3", 18 Nov (4 = damage profiles).

0. **What the lead had already seen.** A13.0(c) (arsenic only). E. Holmstrom, A. Kuronen and K. Nordlund,
   Phys. Rev. B 78, 045202 (2008), doi 10.1103/PhysRevB.78.045202: the threshold displacement energy of silicon
   from DFT molecular dynamics is 12.5 eV at its minimum (<111>) and 36 eV averaged over directions; the paper
   says the 13 to 25 eV commonly used are too low. So the displacement energy is an explicit input with no
   default; the library and demo state their value and its source.
1. **What is counted.** A vacancy is made wherever a target atom is given more than its displacement energy Ed
   by a collision; that atom then moves on as a recoil and may displace others (a full cascade). No
   recombination, no replacement collisions, an amorphous static target: the count is the BCA "full cascade"
   number, an upper bound for what survives (the ion-beam literature finds such counts above what anneals out
   at room temperature). Reported: the vacancy depth profile (vacancies per ion per metre, on the problem's depth
   bins, by the depth where each vacancy was made) and the vacancies per ion.
2. **Description.** `Atom` gains `displacement_eV: Optional[float] = None` (> 0). It is left out of the
   serialised description when None, so no existing problem id changes (R75). `ImplantModels.recoils` True asks
   for damage; then every atom used in a layer needs `displacement_eV`, else the problem is refused (reason
   starting `"recoils:"`). `implant_si(..., displacement_eV=None)`: a value sets the silicon atom's
   `displacement_eV` and `models.recoils` True.
3. **Engines.**
   RustBCA: the damage run uses RustBCA's own command-line program (the same crate and commit, built by
   `tools/install_rustbca.sh` next to the Python module: `<tcad env>/bin/RustBCA`, or SIMWS_RUSTBCA_BIN),
   because the Python call returns only the incident ions. The input is `build_input` written as TOML with
   `track_recoils` and `track_displacements` true and the atoms' Ed. RustBCA writes a displacement row for
   every recoil whose starting energy exceeds its species' Ed (with its start and end points). The ions are run
   in B = 20 batches (sizes differing by at most one), batch b with seed `seed * 1000 + b` (so the seed must be
   below 2 000 000) and `num_chunks` such that a chunk holds at most 2 x threads ions; each batch's files are
   read, histogrammed and deleted before the next batch starts. Refused: "empirical" stopping with recoils
   (RustBCA would scale the recoils' stopping by the boron factor too), and an atom whose Ed is below its own
   `cutoff_eV` (RustBCA creates no recoil below the cutoff, so displacements would be lost).
   Geant4: `recoils 1` in the program: G4ScreenedNuclearRecoil creates a recoil when it is given more than
   (recoil cutoff per nucleon) x (its mass); the recoil cutoff is set to Ed / m of the target atom, so a recoil
   is made exactly when a vacancy is (and the ion stops below Ed m_ion / m_target, which is flagged). The
   program counts each recoil's creation depth into B = 20 batch histograms (event i in batch floor(20 i / N)).
   Refused: more than one atomic species in the target (one recoil cutoff), Ed below 10 eV (the physics cutoff:
   a smaller transfer is never made), and bulk binding above 0 (Geant4 has none).
   Both engines return, in `ImplantResult`, `vacancy_bin_um` (the problem's `depth_bin_um`),
   `vacancy_batches` (B lists of counts on the depth bins) and `vacancy_batch_ions` (ions per batch); vacancies
   made outside the target (none are expected) are counted in `diagnostics["vacancies_outside"]`.
4. **Adapter.** With recoils, the implant adapters add `implant.vacancies_per_ion` (dimensionless),
   `implant.vacancy_profile` (1/m, on the depth axis) and `implant.vacancy_depth_mean` (m), and with a dose
   `implant.vacancy_concentration` (1/m^3). Uncertainties from the batch means: for each batch, the vacancies
   per ion (and per bin, and the mean depth) of that batch; the standard error over the 20 batches. (A cascade
   is not a set of independent vacancies, so Poisson errors would be too small.) A result with recoils carries
   `"implant:damage_full_cascade: no recombination or annealing; an upper bound on surviving defects (A14.1)"`.
5. **Acceptance (tolerances fixed here, before any of these runs).** `implant_si(ion, E, n_ions=2000,
   displacement_eV=15.0)`, seed 1, ZBL, "lindhard_scharff", unless stated. 15 eV is used because it is a
   common value in BCA work (A14.0), not because it is right; the checks are about the codes.
   R70 (counting): RustBCA: the vacancies counted equal the displacement rows whose start lies in the target
   (checker reads one batch directly); both engines: vacancies per ion x N equals the sum of all batch
   histograms; profile x bin x N sums to that number.
   R71 (Ed scaling): B and P at 50 keV, Ed 15 and 30 eV: the ratio of vacancies per ion lies in [1.6, 2.4] for
   each engine (NRT gives exactly 2).
   R72 (NRT band): vacancies per ion / N_NRT in [0.8, 3.0] for B and P at 50 keV on both engines.
   N_NRT = 0.8 T_dam / (2 Ed), T_dam = E / (1 + k g(eps)), k = 0.1337 Z1^(1/6) (Z1/A1)^(1/2),
   g = 3.4008 eps^(1/6) + 0.40244 eps^(3/4) + eps, eps = E / E_L, E_L = 30.724 Z1 Z2 (Z1^(2/3) + Z2^(2/3))^(1/2)
   (A1 + A2) / A2 eV (Norgett, Robinson and Torrens, Nucl. Eng. Des. 33, 50 (1975), with the ion as the
   primary). A full cascade without recombination is expected at or above NRT; a count far outside the band
   means a counting error.
   R73 (cross-code): B and P at 50 keV: Geant4 against RustBCA, vacancies per ion within 20 % and
   `implant.vacancy_depth_mean` within 15 %, with `compare_results` reporting `independent` True. Why: the same
   potential, stopping formula and Ed; the counts then differ through the collision algorithm, and A13.0(c)
   shows -14 % and +16 % for arsenic (reported, not gated).
   R74 (memory): the RustBCA damage run for P at 50 keV, 2000 ions, with the default thread count, has a peak
   resident memory of its child processes below 1.5 GB (`resource.getrusage(RUSAGE_CHILDREN)` in a fresh
   process). Reported: its run time.
   R75 (no id changes): `implant_si("P", 50)` and `implant_si("B", 100)` keep the ids of A12.4; the range-only
   result of a problem is unchanged by A14 (RustBCA in process, `depth_um` identical).
   R76 (refusals): as listed in items 2 and 3, each with its reason prefix ("recoils:" or
   "electronic_stopping:").
6. **Limits, stated.** Amorphous, static target: no channeling, no dose build-up, no recombination; the
   damage scales linearly with dose. Ed is the user's choice and the count scales roughly as 1 / Ed.
7. **After the checker's report (18 Nov).** 23 tests written from A14 alone; run again after the A13.7 fix.
   Passed: R70 (the checker ran batch 0 itself with seed 1000: 123 803 displacement rows, the engine's count;
   batches count from 0), R72 (vacancies / NRT: RustBCA 1.26 for B and 1.60 for P, Geant4 1.08 and 1.38), R74
   (peak memory 90 MB, 56 s for P 50 keV with 2000 ions), R75, R76. RustBCA R71: Ed 15 against 30 eV gives 2.17
   (B) and 2.15 (P).
   **R71 recorded FAIL for Geant4:** the ratio is 1.38 (B) and 1.37 (P), outside [1.6, 2.4].
   **R73:** B 50 keV passed (vacancies -14.8 %, mean depth +9.2 %); **P 50 keV recorded FAIL** on the mean depth
   (+15.2 % against 15 %; vacancies -13.9 % within 20 %).
   *Post hoc diagnosis of R71* (run after the result, so it explains and does not excuse): B 50 keV, 400 ions,
   Geant4 vacancies per ion at Ed 15 / 30 / 60 eV: 613 / 439 / 191 with the 10 eV physics cutoff (ratios 1.39 and
   2.30), 684 / 316 / 156 with 1 eV (2.16 and 2.03). So near its cutoff the Geant4 model under-samples the
   transfers just above it (the long-flight impact-parameter sampling thins the collisions at the edge of the
   cross section): counts with Ed below about three times the cutoff are wrong. **Change:** a g4implant damage
   result with Ed below 30 eV carries `implant:damage_near_cutoff:`, which says to use the RustBCA count. The
   R73 comparisons at 15 eV were therefore made in the region where Geant4 is weakest.
   *Reported after the acceptance runs (the demo, Ed 36 eV, 2000 ions):* Geant4 counts 32.5 % (B 50 keV) and
   33.3 % (P 50 keV) more vacancies than RustBCA, where at 15 eV it counted 14 % fewer; the mean depths differ by
   +7.7 % and +16.3 %. RustBCA's count falls by 2.65 from 15 to 36 eV, Geant4's by 1.7. So the two codes agree on
   damage counts to about 30 % at best, in either direction depending on Ed; the vacancy depth profile's shape
   is the more robust result.
   **Clarifications:** `implant.vacancy_depth_mean` is the pooled mean over all vacancies, its uncertainty the
   standard error of the 20 batch means (ddof 1); the seed limit is refused with the prefix "recoils:".

### A15 (18 Nov 2026, lead): channeling, not done

The owner's option 5 was channeling through an analytic dual-Pearson model with published parameters. The lead
looked for a published, openly readable parameter set on 18 Nov: the dual-Pearson tables for boron in (100)
silicon (Tasch et al., J. Electrochem. Soc. 136, 810 (1989)) and the Suzuki et al. database (IEEE Trans.
Electron Devices 47, 44 (2000)) are behind publishers' paywalls; the implant file shipped with SUPREM-IV.GS
holds only the 1975 amorphous (single Pearson) tables, the same source as R16; the TU Wien theses found show
their tables as images without values. No open-source code that models a crystal (channeling) was found
either (MARLOWE, Crystal-TRIM, IMSIL and MDRANGE are not openly distributed). So nothing is built and no number
is invented. It can be done when the owner supplies one of the two papers (through the university library):
the model is a weighted sum of two Pearson IV curves, a few hours of work plus acceptance against the
paper's own example profiles.

### A12 to A14, laptop run (19 Nov 2026)

Update 9 on the owner's laptop (WSL2, committed as 53dd2a9 on arch/v1): fast tests 1449 passed, 12 skipped,
6 xfailed; the RustBCA command-line program and g4implant built without sudo. Every acceptance number of A12.8,
A13.7 and A14.7 was reproduced to the printed digits (A12: 26 passed; A13: 27 passed, 4 recorded FAILs; A14: 20
passed, 3 recorded FAILs). Only R74 differs, as expected with more threads: peak memory 154 MB (limit 1.5 GB), 69 s.
The demo's table is identical to A14.7's.

### Master becomes the v1 line (19 Nov 2026)

Owner's instruction: "lets do these two: the 23 legacy scripts; replacing master with arch/v1". The original list
of 23 was lost when the lead's notes were compacted, so every runnable entry point of the old master was checked
instead. v1 modified 5 files that existed on master (pytest.ini, simws/adapters/__init__.py, simws/hub/citations.py,
simws/hub/model_form.py, tools/nightly.sh: all additions) and deleted none, so the standalone scripts that do not
import simws (pipeline/legacy, sige_hbt_study, 03_tcad_devices, benchmarks) cannot change. The old workflow
(`rule all`, 14 targets) was run with the old master code and with the v1 code and compared number by number
(tools/legacy_compare.py, relative 1e-9, timestamps skipped): 11 targets on the build machine (28 output files) and
3 on the owner's laptop (geant4_track, si_film; qe_si_scf rerun with the real pseudopotential folder, which git
does not carry): every output identical. Then `simws_master_switch.sh` tagged the old master as
legacy-master-2026-11-19 (a79994f) and fast-forwarded master to arch/v1 (53dd2a9) in ~/sim_workspace; the quick
check passed (49 tests). Not merged yet: the ws/* branches (ws0 to ws4 one commit each, ws6 five).

### A15, continued (19 Nov 2026, lead): channeling from the two papers the owner supplied

The owner supplied both papers named in A15 (Sabanci University library copies) with "there find them here". The
PDFs are not put in the repository (copyright); their SHA-256 and the extracted numbers are, in
`simws/data/channeling/tasch1989_boron.json` and `suzuki2000_arsenic.json`, each with its page, table or figure and
the method by which a figure was read.

0. **What the lead had already seen.** Both papers in full. Tasch: Table I (B+) and Table II (BF2+) typed; Fig. 5
   (fraction of the dose in the main peak, D1/DT, against dose) digitised by the lead (21 points). Suzuki: Fig. 4(a)
   (Rpm, dRp1, dRp2, L, alpha against energy) and 4(b) (channel dose against dose) digitised by the lead; the
   equations (3)-(9). No profile had been computed from either model when items 0 to 7 were committed.
1. **What the two models are.** Tasch (boron, B+): N(x) = D1 f1(x) + D2 f2(x), f1 the main ("non-channeled")
   Pearson density and f2 the channeling Pearson density, both depending on energy only (Table I), D1 / DT on
   energy and dose (Fig. 5); fitted by the authors to SIMS profiles of 68 implants into bare (100) silicon at 7 deg
   tilt and 30 deg rotation. Suzuki (arsenic): N(x) = (Phi - Phi_chan) n_a(x) + Phi_chan n_c(x), n_a a joined half
   Gaussian (peak Rpm, front dRp1, back dRp2), n_c the same with an exponential-type tail exp(-((x - Rpm)/L)^alpha)
   joined continuously at Rpm + dRp2 (factor kappa, eq. 7); each density normalised on [0, inf); parameters from
   crystal-TRIM Monte Carlo that the authors tuned to SIMS (so a fit to simulation, not to measurement), 7 deg tilt,
   22 deg rotation, no cover oxide. Eq. (5) as printed lacks the squares; the joined half Gaussian it cites
   (Gibbons and Mylroie 1973) is h_m exp(-(x - Rpm)^2 / (2 dRp^2)), which is used.
2. **Interpolation and the region each model covers (fixed now).** Boron: Table I parameters linearly in energy;
   D1/DT linearly in energy between the 15, 25, 35 and 50 keV curves and linearly in log10(dose) along each curve.
   Covered: 15 to 50 keV and 2e13 to 2e15 cm^-2 (where all four curves have points). The 80 keV column of Table I
   and the 80 keV and 1e16 points of Fig. 5 are kept in the data file but not used. Arsenic: Fig. 4(a) linearly in
   energy between 40, 80, 120, 160 and 200 keV; Phi_chan / Phi linearly in energy between the 40, 120 and 200 keV
   curves of Fig. 4(b) and linearly in log10(dose). Covered: 40 to 200 keV, 1e11 to 1e16 cm^-2.
   Both: one layer of crystalline silicon, beam tilt 7 deg and rotation 30 deg (boron) or 22 deg (arsenic), each
   within 0.5 deg, dose given. Anything else is refused: the papers say the channeled part depends strongly on the
   angles and no rule for other angles is given.
3. **The Pearson density.** From (Rp, dRp, gamma, beta) through the Pearson differential equation
   d ln f / dx = (x - a) / (b0 + b1 x + b2 x^2), x measured from Rp, with A = 10 beta - 12 gamma^2 - 18,
   a = b1 = -dRp gamma (beta + 3) / A, b0 = -dRp^2 (4 beta - 3 gamma^2) / A, b2 = -(2 beta - 3 gamma^2 - 6) / A
   (the standard Pearson system; Tasch's eq. 4 with the signs that give a density); integrated numerically on the
   region where the denominator keeps its sign around the mode, so every Pearson type the four moments give is
   covered (Table I has beta = 2.80 with gamma = 0, a bounded type II). Truncated to x >= 0 and normalised there.
4. **Description.** `Target.structure` (written `Layer.structure` at first; corrected before any code ran: the field
   is on the target) becomes Literal["amorphous", "crystalline_si_100"], default unchanged.
   `Beam` gains `rotation_deg: Optional[float] = None` (0 to 360), left out of the serialised description when
   None, so no existing id changes. The BCA engines keep refusing anything but "amorphous".
5. **Engine `channeling`** (`simws/engines/channeling_engine.py`, family implant, code "published_fits", in
   process, no environment beyond numpy): evaluates the model on the problem's depth bins (bin-averaged density
   per ion, 1/um). `ImplantResult` gains `profile_um` (bin centres) and `profile_per_um`, and
   `channeled_fraction` (D2 / DT for boron, Phi_chan / Phi for arsenic); depth_um and lateral_um stay empty (no
   ions are followed; no lateral information). The problem's potential, stopping and statistics are not used: a
   result carries `"implant:published_fit: <cite>; <conditions>; the models and statistics of the description are
   not used (A15)"`. Model uncertainty from the reading of the figures: the profile and quantities are also
   evaluated with the figure values moved by their stated reading uncertainty (D1/DT -/+ 0.002; Fig. 4(a) lengths
   -/+ 1 nm, alpha -/+ 0.01, channel dose -/+ 5 %), each group in the direction that widens the result; half the
   spread is the reported sigma. The paper's own fit error to SIMS is not included (Tasch quotes 15 % for the SIMS
   concentrations), and a result says so in `budget:digitisation_only`.
   Adapter `implant.channeling` (env tcad): `implant.profile` (1/m, on the depth axis), `implant.concentration`
   (1/m^3), `implant.Rp` and `implant.dRp` (mean and standard deviation of the whole profile, m),
   `implant.channeled_fraction` (dimensionless). Citation keys tasch1989, suzuki2000.
6. **Acceptance (tolerances fixed here, before any profile has been computed).**
   R80 (the Pearson code): for every column of Table I and Table II, both components, the untruncated density's
   numerical mean, standard deviation, skewness and kurtosis reproduce the table: Rp within 0.5 %, dRp within 1 %,
   gamma within 0.02, beta within 0.05.
   R81 (normalisation): the integral of N over the depth equals the dose within 1e-4 relative, boron at (15, 2e13),
   (35, 5e14), (50, 2e15) and arsenic at (40, 1e11), (120, 1e13), (200, 1e16) (keV, cm^-2), with a target deep
   enough to hold the profile (the engine refuses one that cuts off more than 1e-4 of it).
   R82 (the figures, read again independently): the checker digitises Tasch Fig. 5 and Suzuki Fig. 4 from the PDFs
   by a method of its own; D1/DT within 0.005 at every point used; Fig. 4(a) lengths within 2 nm or 3 % (the
   larger), alpha within 0.02; Fig. 4(b) within 10 %. Typed table values within 0 (copied exactly).
   R83 (Suzuki's joins): n_c continuous at Rpm + dRp2 within 1e-9 relative; n_a and n_c have their maximum at Rpm
   (within one depth bin).
   R84 (what the papers state, as checks of the implementation): boron, the channeled fraction falls with dose at
   15, 25, 35 and 50 keV; arsenic, Phi_chan / Phi falls with dose above 1e13 cm^-2 and Phi_chan(1e16) /
   Phi_chan(1e14) < 1.5 at 40, 120 and 200 keV (saturation); both, the depth at which the profile falls to 1e-3 of
   its peak is larger than that of the main component alone, for every energy and dose of R81.
   R85 (reported, no gate): boron, the main component's Rp (Table I R1) against RustBCA's "empirical" amorphous Rp
   at 15, 25, 35 and 50 keV; arsenic, Rpm against RustBCA's Lindhard-Scharff amorphous peak position at 40, 80,
   120, 160 and 200 keV. These compare a crystal's non-channeled peak with an amorphous calculation; they are
   context, not verification.
   R86 (no id changes): `implant_si("P", 50)` and `implant_si("B", 100)` keep their ids.
   R87 (refusals, reason prefixes): wrong tilt or rotation ("angles:"), an ion other than B or As ("ion:"), energy
   or dose outside item 2 ("energy:", "dose:"), no dose ("dose:"), more than one layer or a layer that is not
   crystalline silicon ("structures:"); the BCA engines refuse "crystalline_si_100" ("structures:").
7. **Trust.** There is no second channeling engine: every result is "unverified" by rule. The boron model is a fit
   to measured SIMS profiles at one pair of angles; the arsenic model is a fit to tuned Monte Carlo. Phosphorus
   needs a third source (Suzuki et al., Solid-State Electron. 42, 1671 (1999), ref. [1] of the arsenic paper).
8. **After the checker's report (19 Nov).** 121 tests written from A15 and the two papers (tests/v1/test_channeling.py).
   **Two implementation defects, found by the checker and fixed** (the code now does what items 1 and 2 say; no
   tolerance and no rule was changed): (a) the arsenic components were normalised over the target instead of over
   [0, inf), so a profile depended on the target's thickness and a too-thin target was not refused (200 keV,
   1e11: 1.2e-2 of the profile lies beyond 0.5 um); (b) the arsenic channel share was interpolated as
   log10(Phi_chan) against log10(dose) (straight lines on the log-log figure) instead of Phi_chan / Phi linearly in
   log10(dose) as item 2 says; between the digitised doses the two differ by up to 41 %. The thin-target refusal is
   now also returned by `unsupported()`, so the planner sees it (reason prefix "structures:").
   **Results (all pass):** R80 the Pearson code reproduces all 24 columns of Tables I and II to 4 decimals (types I,
   II, IV and VII among them); R81 normalisation within 1.4e-5; R82 the checker's own digitisation of the figures:
   Tasch Fig. 5 within 0.0014 (limit 0.005), Suzuki Fig. 4(a) lengths within 0.38 nm (limit 2 nm), alpha within
   0.006 (limit 0.02), Fig. 4(b) within 5.4 % (limit 10 %), Tables I and II identical to the PDF text; R83; R84 (the
   arsenic tail at 200 keV and 1e16 is deeper than the main part by only 0.5 nm: at that dose 0.04 % of the dose
   is channeled); R86; R87 (23 refusal cases).
   **R85 (reported):** boron, RustBCA "empirical" amorphous Rp against Tasch's main-peak R1: -6.4 % (15 keV),
   -0.6 % (25), +2.7 % (35), +5.9 % (50); arsenic, RustBCA Lindhard-Scharff amorphous peak position against
   Suzuki's Rpm: +25 % to +40 % (the means, 32 to 127 nm, are closer). So for boron the amorphous BCA (with the A12
   factor) and the SIMS fit's main peak agree to 6 %; for arsenic RustBCA's peak is deeper than the fit's.

### A15, laptop run (19 Nov 2026)

Update 10 on the owner's laptop, committed as f373ddb on arch/v1; master in ~/sim_workspace fast-forwarded to it.
Fast tests 1573 passed, 9 skipped, 6 xfailed. Channeling acceptance 115 passed, 6 skipped (the checks that re-read
the PDFs, which are not on the laptop). Every R84 and R87 line and the demo table are identical to the build
machine's. The backup remote winbackup received master 53dd2a9, the tag and arch/v1 (before update 10); the push to
GitHub (offsite) stopped at the username prompt.

### A16 (23 Nov 2026, lead): thermal anneal (dopant diffusion) as a fifth problem family; DEVSIM and FiPy; the process chain implant -> anneal -> device -> circuit

The owner asked for "Implant -> anneal -> device: the chain that connects the engines, with diffusion as a new family
checked by two solvers".

0. **What the lead had seen.** The EE212 class handout #12 (Plummer, Deal, Griffin, Stanford 1999-00, "Dopant
   diffusion"; the basis of Table 7-5 of Plummer, Deal and Griffin, *Silicon VLSI Technology*, 2000), as hosted at
   dunham.ece.uw.edu/ee528/notes/Chapter7.pdf, SHA-256 1eadb697...845da63, pages 9-18 read as text: the intrinsic
   table (eq. 13, page 10), the charge-state table (eq. 19, page 18), the worked boron design of pages 12-14.
   F. J. Morin and J. P. Maita, Phys. Rev. 96, 28 (1954), eq. (3): n_i^2 = 1.5e33 T^3 exp(-1.21 eV / kT), fitted
   to points below 700 K. FiPy 3.4.5 installed on the build machine; one 1D diffusion solved to see that it runs.
   No anneal of any kind had been computed by simws when items 0 to 9 were committed.
1. **Description** (`simws/spec/anneal.py`, `AnnealProblem`, kind "anneal"): `dopant` one of B, P, As, Sb;
   `initial` a depth profile (`depth_um` strictly increasing, >= 0; `concentration_cm3` >= 0; 2 to 20000 points;
   linear interpolation between points, zero beyond the last; the first value holds from 0 to the first point);
   `background_cm3` (>= 0) and `background_type` ("donor" or "acceptor"), uniform; `thickness_um` (> 0, at least
   the last profile depth); `steps`: one or more isothermal steps (`temperature_C` 600 to 1400, `time_s` > 0);
   `models.diffusivity` "intrinsic" (default) or "fermi"; `surface` "reflecting" (default: no flux, the dose is
   kept) or "fixed" with `surface_concentration_cm3` (a constant source, e.g. a predeposition); the back side is
   reflecting; `numerics.dx_um` (default 0.001, at most 20000 intervals) and `numerics.steps_per_anneal` (default
   200). Each step's time points are t_k = time_s (k / N)^2, k = 1..N (the same points for every engine).
2. **The physics (both engines solve exactly this).** dC/dt = d/dx (D h dC/dx), x the depth.
   "intrinsic": h = 1 and D = D0 exp(-E / kT) from the handout's eq. (13): B 1.0 cm^2/s, 3.5 eV; As 9.17, 3.99;
   Sb 4.58, 3.88; P 4.70, 3.68. "fermi": D = D0 + D- (n/ni) + D= (n/ni)^2 for donors (P, As, Sb) and
   D = D0 + D+ (p/ni) for B, each term D.0 exp(-D.E / kT) from eq. (19): B D0 0.05/3.5, D+ 0.95/3.5; As D0
   0.011/3.44, D- 31.0/4.15; Sb D0 0.214/3.65, D- 15.0/4.08; P D0 3.85/3.66, D- 4.44/4.0, D= 44.2/4.37.
   n (or p for B) from charge neutrality with the background: N = C - Nb (background of the other type) or
   C + Nb (same type), n = N/2 + sqrt(N^2/4 + ni^2); the field factor h = 1 + C / sqrt(N^2 + 4 ni^2) (the flux
   of the ionised dopant in the built-in field, for one diffusing species on a uniform background).
   ni = 3.87e16 T^1.5 exp(-0.605 eV / kT) cm^-3 (Morin and Maita, the square root of their eq. 3), k = 8.617333e-5
   eV/K. Every "fermi" result carries `model:ni_morin_maita_extrapolated` (their fit stops at 700 K). Every result
   carries `anneal:not_modelled: transient enhanced diffusion from implant damage, clustering, solid solubility,
   oxidation and segregation`: the classical equation only.
3. **Engines.** `devsim_anneal` (`simws/engines/devsim_anneal_engine.py`, family anneal, CODE "devsim"): DEVSIM 1D
   mesh with nodes at k dx, the flux on each edge with the arithmetic mean of D h at its two nodes, backward
   Euler on the time points of item 1, Newton to relative 1e-10. `fipy` (`simws/engines/fipy_engine.py`, family
   anneal, CODE "fipy"): FiPy Grid1D cells of width dx (cell centres at (k + 1/2) dx), DiffusionTerm with D h
   at the faces as FiPy interpolates it, TransientTerm (backward Euler) on the same time points, sweeps until the
   largest relative change is below 1e-10 (at most 50). The initial profile is interpolated onto each engine's own
   points. Both report the concentration at x_k = k dx, k = 0..M (FiPy: its face values). Two codes, so the two
   are independent (A11.4). Capability: tested 800 to 1200 C (the range of the handout's plot); outside,
   `extrapolated:`.
4. **Result and adapters** `anneal.devsim_anneal`, `anneal.fipy` (env tcad): `anneal.concentration` (1/m^3 on the
   depth axis, m), `anneal.dose` (1/m^2, trapezoid over the reported points), `anneal.Dt` (m^2, the sum over the
   steps of the intrinsic D times the time, for context), `anneal.junction_depth` (m; only with a background of
   the other type: the first depth where C falls to the background, linear between points; none, and the flag
   `anneal:no_junction`, when C stays above it or starts below it). Deterministic: `budget:none` (the uncertainty
   of the diffusivity tables is not included and the flag says so: `budget:model_not_included`). Citations
   plummer2000, morin1954, devsim, fipy (Guyer, Wheeler, Warren, Comput. Sci. Eng. 11(3), 6 (2009)).
5. **Profile doping for devices.** `DeviceSpec.profiles`: a list of `ProfileTerm` (species donor or acceptor;
   `axis` "x" or "y"; `position_um` strictly increasing, `concentration_cm3` >= 0, 2 to 20000 points; linear
   interpolation, zero outside [first, last]). Terms add to the boxes in `doping_at`. Left out of the serialised
   description when empty, so no existing id changes. DEVSIM evaluates it at every node (node solutions added to
   the doping expressions); Charon refuses it ("doping:"); devsim_eb inherits DEVSIM's handling.
6. **Chain** (`simws/chain/process.py`): `initial_from_implant(result, dose_cm2)` (implant.concentration, or
   implant.profile x dose); `anneal_problem(...)`; `diode_from_anneal(result, ...)` (a 1D device: the annealed
   dopant as a ProfileTerm on the background box, contact at the surface and at the back, a DC sweep);
   then `chain.diode.fit_shockley` and `circuit_from_fit` (A9.6) as before. A demo, `tools/v1_process_demo.py`.
7. **Acceptance (tolerances fixed here, before any anneal has been computed).**
   R90 (Gaussian, "intrinsic", reflecting, both engines): B at 1000 C for 1800 s from (a) a half Gaussian at the
   surface, sigma0 = 0.02 um, Q = 1e14 cm^-2 and (b) a Gaussian at Rp = 0.1 um, sigma0 = 0.03 um with its image
   in the surface, Q = 1e14; thickness 1 um, default numerics. Against the exact solution (the same form with
   sigma^2 = sigma0^2 + 2 D t): largest deviation at most 1 % of the exact peak; the standard deviation of (a)
   (about the surface) within 0.5 %.
   R91 (erfc, "intrinsic", fixed surface 1e19, zero initial, B at 1000 C 1800 s, both): C = Cs erfc(x / 2 sqrt(D
   t)) within 1 % of Cs at every point; the dose within 1 % of 2 Cs sqrt(D t / pi).
   R92 (successive steps): 1000 C 900 s then 1100 C 600 s against one Gaussian with the summed Dt (eq. 14),
   case R90(a): within the R90 tolerances.
   R93 (dose): reflecting surface, every case here and in R94: the final dose equals the initial dose of the
   engine's own interpolated profile within 1e-6 relative.
   R94 (the two engines, "fermi", no exact answer): As (Rp 0.03 um, sigma 0.012 um, 1e15 cm^-2) into p-type 1e16 at
   1000 C 600 s; P (0.05, 0.02, 5e15) into p-type 1e16 at 1000 C 600 s; B (0.1, 0.035, 1e15) into n-type 1e16 at
   1000 C 1800 s; thickness 1 um. Largest difference of the profiles at most 1 % of the peak; junction depths within
   1 %; doses within 1e-6.
   R95 (the Fermi model's limits): at C << ni the "fermi" D equals the sum of the eq. (19) terms with n = ni
   (h = 1 within 1e-6); for C >> ni with no background h tends to 2; the R94 "fermi" junctions are deeper than the
   "intrinsic" ones for As and P at 1000 C (concentration-enhanced diffusion).
   R96 (the handout's worked design, pages 12-14): Q = 4.3e13 cm^-2 of B in a half Gaussian with sigma0 = 0.005
   um, n-type background 1e15, 1100 C for 6.8 h, "intrinsic", thickness 10 um, dx 0.005 um. Gate: the junction
   depth within 1 % of the closed form x_j = sqrt(4 D t ln(Q / (sqrt(pi D t) C_B))) with the table's D (the
   sigma0 term is negligible). Reported: the depth against the handout's 3 um, and D(1100 C), D(950 C) against
   the handout's 1.5e-13 and 4.2e-15. (Written first with a 3 % gate against the handout's 3 um; the lead's own
   arithmetic, before any code ran, gives D(1100 C) = 1.42e-13 from the table, so x_j = 2.90 um, 3.5 % below the
   handout's rounded design: that gate tested the handout's rounding, not the code, and was replaced.)
   R97 (profile doping): for a device with a ProfileTerm (and boxes), DEVSIM's NetDoping at every node equals
   doping_at within 1e-12 relative; the ids of `pn_diode_1d()`, the HBT and every library device are unchanged.
   R98 (refusals, reason prefixes): a device engine or an implant engine given an AnnealProblem ("problem kind:");
   Charon given a ProfileTerm ("doping:").
   R99 (chain, reported): B 35 keV 1e14 cm^-2 into n-type 1e16 (RustBCA, "empirical") -> 1000 C 1800 s "fermi"
   on both engines -> DEVSIM diode -> Shockley fit -> ngspice: junction depths of the two engines (gate: 1 %), Js
   and N of the fit, the ngspice current at 0.6 V against the device's.
8. **Trust.** Two independent engines: an anneal result whose `anneal.junction_depth` and `anneal.dose` were
   compared with the other engine's and passed (1 %, 1e-6) is "verified". Nothing here is "validated": no
   measured profile is used (SIMS after anneal would be the next source).
9. **Not in A16.** Implant damage and transient enhanced diffusion, two-dimensional diffusion, ramps, oxidation,
   several dopants at once. Each is a refusal ("dopant:", "steps:") or the flag of item 2.
10. **After the checker's report (23 Nov).** 184 tests written from A16 and the two PDFs (tests/v1/test_anneal.py);
   178 passed and 6 failed at first. The PDFs' numbers equal item 2 (eq. 13, eq. 19, Morin and Maita eq. 3, the
   design numbers of pages 12-14).
   **One implementation defect, found by the checker and fixed:** for a profile that drops to zero, devsim_anneal
   returned round-off values just below zero ahead of the diffusion front (e.g. -2e-316 cm^-3), and the hub refused
   the valid result ("must be non-negative"). Values below zero by less than 1e-12 of the peak are now reported as
   0 (their count is in the diagnostics, `negative_roundoff_set_to_zero`); larger negatives are left for the hub's
   check. FiPy was not affected.
   **One contract defect, found by the checker, decided by the lead:** R94 and item 8 compared the two engines'
   doses within 1e-6. The engines sample the starting profile at different points (DEVSIM at k dx, FiPy at
   (k + 1/2) dx, item 3), so for a profile with a slope at the surface their starting doses already differ (R94
   read literally, a Gaussian cut at the surface: As 3.83e-5, P 1.38e-5, B 1.97e-6), and the final doses differ by
   exactly that while each engine keeps its own dose to 1e-13. That gate measured the sampling of the input, not
   the solvers. Now: R94's dose gate is "the engines' final doses differ by at most the difference of their starting
   doses plus 1e-6" (each engine's own conservation stays gated at 1e-6 by R93 and by the adapter's self-check);
   item 8's trust comparison is the junction depth only (1 %). R94 is run both ways (cut at the surface, and with
   its image as in R90(b) and `chain.process.gaussian_profile`); every other R94 number passes both ways.
   **Results:** R90 0.22 % of peak (limit 1 %), sigma 0.002 % (0.5 %); R91 0.09 % of Cs, dose -0.08 % (1 %); R92
   0.19 %; R93 5.3e-13 (1e-6); R94 profiles within 0.018 % of peak, junctions within 0.043 % (1 %); R95 h - 1 <=
   1e-6 at C = 1e-6 ni, h = 2 within 1e-6 at C = 1e4 ni, Fermi junctions deeper (As 144 against 111 nm, P 404
   against 255 nm); R96 x_j 2.914 um on both engines against the closed form 2.899 um (+0.52 %, limit 1 %; the
   handout's design says 3 um; D(1100 C) 1.426e-13 against its 1.5e-13, D(950 C) 3.79e-15 against 4.2e-15); R97
   NetDoping equal to doping_at within 2e-16, every library id unchanged; R98 refusals as stated. R99: junctions
   431.38 and 431.39 nm (0.0004 %), Js 1.111e-6 A/m^2, N 1.0089, ngspice -1.12 % against the device at 0.6 V.
   Gaps the checker noted, recorded as the behaviour: no junction quantity and no flag when the background is of
   the dopant's own type or zero; morin1954 is cited on intrinsic results too; both budget flags are set.

### A17 (23 Nov 2026, lead): a TCAD strike current as the circuit pulse (no assumed pulse shape)

The owner asked for "TCAD strike -> circuit pulse, to remove the assumed pulse shape from the upset calculation".
Until now (A10) the critical charge used Messenger's double exponential with time constants given by hand.

0. **What the lead had seen.** The legacy DEVSIM strike (`simws/tcad/particle_strike.py`, a 2D replica of Charon's
   particleStrike test, agreeing with Charon to 0.9 % of peak) and the generation hook of `simws.tcad.physics.
   setup_region` (`extra_generation`, pairs/cm^3/s added to both continuity equations). DEVSIM's contact current
   is the particle current (no displacement current). No strike had been run on the v1 path when items 0 to 8
   were committed.
1. **Description.** A device analysis `StrikeTransient` (kind "strike"), 1D devices only: `bias` (a list of Bias,
   ramped first in the order given; other contacts at 0 V); `pairs_cm2` (> 0, electron-hole pairs per cm^2 of
   device area); `depth_um` (start < stop, inside the device): the pairs are generated uniformly over that depth
   segment; `t_peak_s` and `sigma_s` (> 0, t_peak >= 4 sigma): the time profile is a Gaussian truncated to
   [t_peak - 4 sigma, t_peak + 4 sigma] and renormalised to unit area; `t_stop_s` (> t_peak + 4 sigma);
   `steps_per_sigma` (default 10) and `points_per_decade` (default 20). Time points (the same for every engine,
   `time_points_s()`): 10 points geometric from 1e-3 (t_peak - 4 sigma) to t_peak - 4 sigma; then uniform steps of
   sigma / steps_per_sigma to t_peak + 4 sigma; then geometric at points_per_decade to t_stop_s. A new kind in the
   Analysis union changes no existing id.
2. **Physics.** The device's drift-diffusion models unchanged, plus G(x, t) = pairs_cm2 / (stop - start) x g(t) in
   pairs/cm^3/s inside the segment, g the truncated unit-area Gaussian. Backward Euler with the source at the end
   of each step (as the legacy replica). On the mesh, the footprint (1 inside, 0 outside, 1/2 at a node exactly on
   an end) is scaled so that its node-volume integral equals stop - start exactly: the generated pairs are exact
   up to the time quadrature, whose sum is reported.
3. **Engine.** DEVSIM only (`analyses` gains "strike"); Charon and devsim_eb refuse ("analyses:"); no mesh study
   for "strike" (as for small_signal, A6.6). One engine: every strike result is "unverified" by rule.
4. **Result.** `EngineResult` gains `time_s`; analysis "strike"; `currents` per contact per time point (the first
   point is t = 0, at the bias, before any generation). Adapter quantities (1D, canonical): `tr.I.<contact>`
   (A/m^2 on the axis "time", s, positive into the device), `tr.Q.<contact>` (C/m^2, the trapezoid integral of
   I(t) - I(0) over the time points), `tr.Q_generated` (C/m^2, q x pairs_cm2 x 1e4, exact), diagnostics
   `generated_pairs_quadrature_cm2` (the engine's time sum). Flags on every strike result: `strike:fixed_bias: the
   contacts are held at their bias; a circuit node's voltage change does not act back on the collection (no
   mixed mode)` and `strike:1d_column: the pairs are spread over the device area (no track radius, no 2D
   funnelling)`.
5. **Chain** (`simws/chain/see.py`): `strike_problem(device_problem, charge_C, area_m2)` (pairs_cm2 = charge /
   (q area)); `tcad_pulse(device_result, contact, area_m2)`: a Pwl of (I(t) - I(0)) x area at the contact (sign as
   collected: positive = current into the device, i.e. drawn from the circuit node); `critical_charge_tcad(hub,
   circuit, node, threshold_V, device_problem, contact, area_m2, q_low_C, q_high_C, rel_tol=1e-3,
   rerun=True)`: the same bisection as A10, each step running the device at that charge (`rerun`) or scaling one
   device pulse linearly (`rerun=False`, flagged `strike:linear_scaling`); returns the A10 fields plus the device
   result ids and the collected charge at q_crit. `critical_charge` (Messenger) is kept unchanged.
6. **Acceptance (tolerances fixed here, before any strike has been computed on this path).** The reference
   device: 1D, 5 um, n+ 1e19 on [0, 0.2) um over p 1e16, contacts "top" (x = 0) and "back"; ModelSpec defaults
   (Masetti, Scharfetter SRH, Auger, Slotboom); top at +1 V (reverse bias); generation on [0.25, 0.6] um (inside
   the depletion region); t_peak 20 ps, sigma 5 ps, t_stop 1 us; mesh 1 nm to 1 um and 20 nm after.
   R100 (collection): pairs 1e11 cm^-2: Q_top / Q_generated within 1 % of 1, Q_back / Q_generated within 1 % of -1.
   R101 (time steps): steps_per_sigma 20 and points_per_decade 40 against the defaults: Q_top within 0.5 %, peak
   |I_top| within 2 %.
   R102 (mesh): mesh hints scale 0.5 against 1: Q_top within 0.5 %, peak within 2 %.
   R103 (low-injection linearity): pairs 1e10 against 1e11: the collected fractions within 1 % of each other and
   the peak currents in the ratio 10 within 2 %. Reported (no gate): the collected fraction and the peak per pair
   at 1e12 and 1e13 (high injection, the reason for `rerun`).
   R104 (Kirchhoff): at every time point |I_top + I_back| at most 2 % of the largest |I_top| (the displacement
   current at the contacts is not in the particle current; this bounds what is missing).
   R105 (quadrature): the reported quadrature sum within 1e-3 of pairs_cm2.
   R106 (the chain, reported): a node held at 1 V by 10 kOhm from a 1 V supply with 1 fF to ground, threshold
   0.5 V, the reference device at 1 um^2 on the node: q_crit with the TCAD pulse (rerun), with the TCAD pulse
   scaled linearly, and with Messenger's pulse at (1 ps, 10 ps) as in A10; the collected charge at q_crit.
   R107 (refusals and ids): a 2D device ("strike:"), Charon and devsim_eb ("analyses:"), a mesh study ("mesh_study:");
   descriptions with t_peak < 4 sigma, t_stop <= t_peak + 4 sigma, or a segment outside the device are invalid;
   every library problem keeps its id.
7. **Trust.** One engine: "unverified". The model-form limits are the two flags of item 4. Fully coupled mixed mode
   (the legacy `simws/tcad/mixed_mode.py`) is the next step and the natural check of the fixed-bias assumption.
8. **Not in A17.** 2D or axisymmetric strikes, a track radius, LET along a track from G4SEE (the deposit enters as
   a charge), mixed mode on the v1 path.
9. **Two corrections after the lead's first development runs (23 Nov), before the checker wrote any test; no
   tolerance changed.** (a) Item 4's `tr.Q` was the trapezoid integral of I(t) - I(0). Backward Euler conserves
   charge with the sum over steps of I_k (t_k - t_{k-1}) (each step's current is that step's mean); the trapezoid
   adds half of sum (I_{k-1} - I_k)(t_k - t_{k-1}), which on the reference device overstated the collected charge
   by 5.7 % (top) and 6.3 % (back) with the backward sum at 0.99998 for both. `tr.Q` is now that backward sum, and
   `tcad_pulse` is a staircase (I_k held over (t_{k-1}, t_k], corners 1e-4 of the step apart) so the circuit
   receives the same charge. (b) Item 0 noted that DEVSIM's contact current is the particle current only. On the
   reference device the particle currents at the two contacts differ by up to 14 % of the peak during the pulse
   (the displacement current eps dE/dt in the neutral p region carrying the pulse is not small), so the particle
   current is not the terminal current a circuit sees. `tr.I.<contact>` is now the particle current plus the
   displacement current, the backward difference of the contact charge (DEVSIM's get_contact_charge on the
   potential equation) over each step; with it the two contacts agree to 4e-10 of the peak. The particle
   currents stay in the diagnostics (`particle_currents`). R104 keeps its 2 % gate.
10. **After the checker's report (23 Nov).** 57 tests written from A17 (tests/v1/test_strike_pulse.py); 56 passed,
   1 failed. No implementation defect was found.
   **Recorded FAIL, R103 (peak ratio):** the peak current at 1e11 pairs/cm^2 is 9.615 times that at 1e10 (limit: 10
   within 2 %). The collected fractions agree to 8e-6 (limit 1 %). The checker ran more deposits outside the gate:
   peak per pair 6.047e-5, 6.044e-5, 6.023e-5, 5.974e-5, 5.791e-5 A/m^2 per pair/cm^2 at 1e8, 1e9, 1e10, 3e10, 1e11,
   with the peak time unchanged (25.5 ps) and R101 / R102 showing the peak converged to 0.3 % and 0.004 %. So the
   code is linear in the low-injection limit (1e8 against 1e9: 0.035 %), and the failure is the lead's choice of
   1e11 as "low injection": 1e11 over 0.35 um is about 0.3 of the p doping. The gate is kept as written and the
   test is a strict xfail; the 1e8 / 1e9 numbers are reported, not a gate fixed after the fact.
   **Results:** R100 Q_top / Q_generated 0.999982 and Q_back -0.999982 (limit 1 %); R101 Q 8.1e-7, peak 3.1e-3 (0.5 %,
   2 %); R102 Q 2.7e-7, peak 4.3e-5; R103 fractions 8.3e-6 (1 %); R104 at most 3.9e-9 of the peak (2 %); R105
   quadrature 8.9e-7 (1e-3); tcad_pulse integral against tr.Q x area 5.7e-6; R107 refusals as stated, every library
   id unchanged. R103 reported: collected fraction / peak per pair 0.99989 / 2.61e-5 at 1e12, 0.9714 / 3.30e-6 at
   1e13 (high injection: the peak saturates and the pulse spreads, which is why `rerun` is the default).
   **R106 (reported):** with the node of R106 (10 kOhm, 1 fF) and A10's circuit window (step 0.1 ps, stop 300 ps),
   Messenger's pulse gives q_crit 1.367 fC; neither TCAD pulse upsets the node up to 100 fC: at 1 um^2 the device's
   current saturates near 49 uA (100 fC: collected 79 %, peak at 564 ps), and 49 uA x 10 kOhm < 0.5 V, so the node
   cannot reach the threshold whatever the window (3 ns: v_min 0.510 V). The lead's demo adds, as context, the same
   node through 100 kOhm: q_crit 0.754 fC (TCAD rerun), 12.29 fC (one 100 fC pulse scaled down), 0.646 fC
   (Messenger). The fixed-shape assumptions are where the answers part.
   **Wording, recorded as implemented:** the mesh "1 nm to 1 um and 20 nm after" is the hint lines (0, 1 nm), (1 um,
   1 nm), (5 um, 20 nm), so the spacing grades from 1 nm to 20 nm between 1 and 5 um; the geometric tail has
   ceil(decades x points_per_decade) points; with t_peak = 4 sigma the 10 geometric head points are absent;
   `critical_charge_tcad` returns the A10 fields except the Messenger time constants.

### A16 and A17, laptop run (23 Nov 2026)

Update 11 on the owner's laptop, committed as 30aec7a on arch/v1; master in ~/sim_workspace fast-forwarded to it.
FiPy 3.4.5 installed into the tcad env by the update. Fast tests 1809 passed, 13 skipped, 7 xfailed. Anneal
acceptance 180 passed, 4 skipped (the checks that re-read the source PDFs). Strike acceptance 57 passed, 1 xfailed
(R103's peak ratio, A17.10). Every gated R90 to R105 number and both demo tables are identical to the build
machine's: R90a 0.2168 %, R91 0.0918 %, R99 junctions 431.38 / 431.39 nm, R100 0.999982, R103 ratio 9.61544, the
100 kOhm critical charges 0.754 / 12.294 / 0.646 fC.

### A18 (23 Nov 2026, lead): mixed mode, the strike with the circuit node solved together with the device

The owner chose "mixed mode" and "G4SEE track -> TCAD" (A19) as the next two steps. A17 held the struck contact at
a fixed bias; at 10 kOhm / 1 fF that decided the answer (no upset with the TCAD pulse, 1.4 fC with Messenger's).

0. **What the lead had seen.** The legacy `simws/tcad/mixed_mode.py` (DEVSIM circuit elements on a contact,
   elements scaled per unit depth, fully coupled Newton; its `rc_check` matched the analytic RC exponential); the
   v1 DEVSIM engine's circuit contacts (A6.4) and `ramp_source`; A17's strike path and results. No mixed-mode run on
   the v1 path when items 0 to 8 were committed.
1. **Description.** `StrikeTransient.load` (optional, left out of the serialised description when absent, so no
   A17 id changes): `contact`, `supply_V`, `R_ohm` (> 0), `C_F` (>= 0), `area_um2` (> 0, the area the 1D device
   stands for), `supply_step_V` (default 0.1). The circuit: a source supply_V from node "vdd" to ground, R from
   "vdd" to the contact, C from the contact to ground. The load contact may not be in `bias` (other contacts keep
   their fixed bias).
2. **Engine (DEVSIM).** The load contact is a circuit contact; the elements are scaled to the 1D device's per-cm^2
   currents: R_eff = R A, C_eff = C / A (A in cm^2). The supply is ramped from 0 (steps of supply_step_V, halved on
   failure), then the fixed biases, then the transient of A17 unchanged. The node voltage is read from DEVSIM's
   circuit solution after every step.
3. **Result.** `EngineResult.node_V` (the load contact's voltage at every time point; empty without a load).
   Adapter: `tr.V.<contact>` (V on the axis "time") and `tr.V_min.<contact>` (V). The flag `strike:fixed_bias` is
   replaced by `strike:mixed_mode: the device and its load (supply, R, C) are solved together in one Newton
   system`; `strike:1d_column` stays.
4. **Chain** (`simws/chain/see.py`): `with_load(device_problem, contact, supply_V, R_ohm, C_F, area_m2)`;
   `critical_charge_mixed(hub, device_problem, threshold_V, area_m2, q_low_C, q_high_C, rel_tol=1e-3)`: the A10
   bisection, each trial a device run at that charge; upset when tr.V_min of the load contact is at or below the
   threshold; returns the A10 fields (no Messenger time constants), the device result ids and v_min at q_crit.
5. **Acceptance (tolerances fixed here, before any mixed-mode run).** The reference device of A17.6 with the top
   contact on a load (supply 1 V, area 1 um^2); its fixed biases empty.
   R110 (Kirchhoff at the node): R 10 kOhm, C 1 fF, pairs 1e11: at every time point k >= 1,
   |(V_s - v_k)/R - C (v_k - v_{k-1})/(t_k - t_{k-1}) - I_k A| <= 1e-3 x max_k |I_k A|, with I the reported
   tr.I.top (current into the device).
   R111 (the fixed-bias limit): R 10 kOhm, C 1 fF, pairs 1e9 (the node moves by about 0.2 % of the supply):
   tr.Q.top within 1 % of the A17 fixed-bias run of the same device at top = +1 V, and the peak current within 2 %.
   R112 (collection): R 100 kOhm, C 1 fF, pairs 1e11: tr.Q.top / tr.Q_generated within 1 % of 1.
   R113 (convergence): R 100 kOhm, C 1 fF, pairs 5e11: the dip V_s - V_min with steps_per_sigma 20 and
   points_per_decade 40, and with mesh scale 0.5, within 1 % of the default's.
   R114 (DC point): with no deposit to speak of (pairs 1e6), v at t = 0 equals V_s - I_0 A R within 1e-6 V
   (I_0 the reported current at t = 0).
   R115 (reported): q_crit mixed for the node of A17 R106 (10 kOhm, 1 fF) and for 100 kOhm / 1 fF, next to A17's
   fixed-bias TCAD (rerun) and Messenger (1 ps, 10 ps); v_min at q_crit.
   R116 (refusals and ids): the load contact also in bias, a load on an unknown contact ("analysis:"), a 2D device
   ("strike:"), Charon and devsim_eb ("analyses:"); `reference_strike_diode()` keeps its A17 id
   ff0714fb4325f7f403ef4969c97f233b (computed by the lead on the A17 code before this amendment).
6. **Trust.** One engine: "unverified". R111 ties the mixed-mode path to A17's fixed-bias path where they must
   agree; there is no second code.
7. **Not in A18.** Loads other than supply-R-C on one contact; transistors; 2D mixed mode.
7a. **A18 clarification by the lead (23 Nov, during development, before the checker).** Item 2's supply and R are
   built as their Norton equivalent: a current source supply_V / R into the contact and R from the contact to
   ground. For the node this is the same circuit (the same v(t), the same currents in the device and in C; the
   current "in R" of R110 is (V_s - v)/R either way). Built literally, the voltage source's current unknown is the
   leakage (about 1e-16 A here) and Newton stalled on it at 1e-9 to 1e-6 relative in round-off; in the Norton form
   the only circuit unknown is the node voltage and the coupled solves reach the description's 1e-10. The ramp of
   item 2 ramps that current in steps of supply_step_V / R.
8. **After the checker's report (23 Nov).** 41 tests written from A18 (tests/v1/test_mixed_mode.py); 39 passed and
   2 failed at first.
   **Results:** R110 worst residual 6.9e-10 of the peak (limit 1e-3); R111 tr.Q.top 1.1e-4 (1 %); R112 0.999977
   (1 %); R113 mesh 3.7e-5 (1 %); R114 3.7e-12 V and 3.7e-11 V (1e-6 V); R116 as stated, the A17 id kept; trust
   "unverified". R115: 10 kOhm / 1 fF, no upset in mixed mode up to 100 fC (v_min 0.5915 V at 100 fC, 78 %
   collected); 100 kOhm / 1 fF, q_crit 0.9136 fC in mixed mode against 0.7537 fC (fixed-bias TCAD pulse) and
   0.6461 fC (Messenger).
   **Recorded FAIL, R111 (peak):** the mixed-mode peak is 5.5 % below the fixed-bias one (limit 2 %) while the
   node moves by 0.04 % and the charge agrees to 1e-4. With the node free, part of the fast current is supplied by
   the device's own junction capacitance (about 0.2 fF here, against C = 1 fF) and never reaches the external
   circuit; stiffening the node closes the gap (10 Ohm: 7.6e-5; 100 fF: 1.0e-3; C = 0: 7.3e-2). The gate's premise
   (a 0.04 % voltage change means a fixed-bias peak) was the lead's error. Gate kept; strict xfail.
   **Recorded FAIL, R113 (time steps):** the dip with steps_per_sigma 20 and points_per_decade 40 is 1.27 % larger
   than with the defaults (limit 1 %); with 40 / 80 about 2 %. The minimum falls in the geometric tail (63 ps),
   and the tail's points per decade drive the error. Mixed-mode dips and q_crit therefore carry a time-step error
   of about 2 % at the defaults, in no budget. Gate kept; strict xfail.
   **Clarified:** item 4's "v_min at q_crit" is returned at the two bracket ends (`v_min_at_q_lo_V`,
   `v_min_at_q_hi_V`); q_crit itself is not run. `C_F` defaults to 0 (no capacitor).
   **Fixed after A19's checker (see A19.7):** DEVSIM's error type and the transient retry, which apply here too.

### A19 (23 Nov 2026, lead): the G4SEE deposit's depth profile as the TCAD generation, and the upset probability

1. **Description.** `StrikeTransient.profile` (optional, left out when absent): `depth_um` (strictly increasing)
   and `weight` (>= 0, not all 0; relative pairs per unit depth, linear between points, zero outside); with a
   profile, `depth_um` of the strike must equal (first, last) of the profile. The footprint of A17.2 becomes the
   weight at each node (with the 1/2 rule at the ends), scaled so its node-volume integral equals stop - start:
   pairs_cm2 stays exact.
2. **Depth profile from G4SEE** (`simws/chain/see.py`, `deposit_profile(hub, see_problem, edges_um)`): one G4SEE run
   per slice [e_k, e_k+1] (the SV of the given problem moved to that depth and thickness, everything else
   unchanged); returns the slice centres, `see.edep_mean` per slice and its uncertainty, and the mean deposit
   per micrometre. `profile_from_deposit(...)` turns it into a GenerationProfile (weights = mean MeV per um at
   the slice centres, extended flat to the slice edges at both ends).
3. **Chain.** `strike_from_see(device_problem, profile)`: the strike's generation follows the profile;
   `upset_probability(see_result, qcrit)` (A10, unchanged) with q_crit from `critical_charge_mixed` run on that
   device: every event's deposit is taken with the mean depth shape (flag `strike:mean_profile` on the chain's
   result), its size from G4SEE's spectrum in the SV covering the whole profile.
4. **Acceptance (tolerances fixed here, before any run of this amendment).**
   R120 (slices add up): proton 5 MeV, normal incidence, bulk 20 um wide and 50 um thick, SV 1 um x 1 um laterally,
   5 slices of 1 um from 0 to 5 um, 20 000 events each, and one run with the SV 0 to 5 um: the sum of the slice
   means within 4 combined standard errors of the 0 to 5 um mean.
   R121 (against NIST, as A8.7): alpha 5.0 MeV, the same geometry with a 10 um wide SV, slices of 1 um from 0 to
   10 um: each slice mean within 3 % of the CSDA deposit of `simws.radiation.nist_star` for that slice.
   R122 (a profile in TCAD): (a) a flat profile over A17's segment reproduces the A17 uniform run (tr.Q.top and
   every tr.I.top value within 1e-9 relative); (b) a linear profile (weight 1 at 0.25 um, 3 at 0.6 um): the
   quadrature of generated pairs within 1e-3 of pairs_cm2 and tr.Q.top / tr.Q_generated within 1 % of 1.
   R123 (refusals): a profile whose ends differ from depth_um, an all-zero weight, decreasing depths.
   R124 (chain, reported): proton 5 MeV (and 2 MeV) on the reference device of A18.5 at 1 um^2, R 100 kOhm, C 1 fF:
   the G4SEE profile over 0 to 5 um in 0.5 um slices, q_crit mixed with that profile, the upset probability per
   proton with it, and with A10's Messenger q_crit (1 ps, 10 ps) on the same node, 20 000 events.
5. **Trust.** The deposit side keeps G4SEE's standing (A8, checked against NIST); the device side is A18's
   ("unverified"). The chain's result carries both.
6. **Not in A19.** Per-event depth shapes (each event takes the mean shape), lateral spread of the track, tilted
   tracks in the device (the 1D column is normal incidence).
7. **After the checker's report (23 Nov).** 33 tests written from A19 (tests/v1/test_see_profile.py); 30 passed and
   3 failed at first. Histograms chosen by the checker: protons 0 to 1 MeV, alphas 0 to 5.25 MeV, 20 000 bins; no
   overflow in any run.
   **Results:** R120 slice sum 0.067262 MeV against 0.067071 +- 0.000071 MeV, 1.86 combined sigma (limit 4);
   R121 worst slice -0.45 % against NIST (3 %); R122a identical to A17's uniform run (0 difference); R122b
   quadrature 8.9e-7 (1e-3), Q_top / Q_gen 0.999973 (1 %); R123 refusals as stated.
   **Two defects, found by the checker and fixed.** (1) R124 stopped on "Convergence failure!" at one trial charge
   (5 MeV, 2.3082e12 pairs/cm^2). DEVSIM raises its own error type, which the strike path caught only as
   RuntimeError (so the failure escaped as an engine error rather than "not converged", on the A17 path as well),
   and the transient had no retry (A18.2 named none). Newton stalled on the potential equation at about 1e-7
   relative near the end of the pulse. Now a failed step is retried once at relative 1e-6 (flag
   `newton_rel_tol:1e-06 at N transient steps`, count in the diagnostics `relaxed_steps`), then halved up to 8
   times (flag `strike:step_split:N`; substeps are recorded as time points, so the backward charge sum stays
   exact). The failing case now runs, with one relaxed step. (2) The flag `strike:mean_profile` of item 3 was on
   no result: `critical_charge_mixed` now adds it when the device has a profile, `upset_probability` carries it
   from that q_crit, and the q_crit result gives `device_trust`.
   **R124 (reported; the checker's run before the fix, with q_high 50 fC):** proton 5 MeV, mean 0.06707 MeV in
   0-5 um, q_crit 3.6886 fC (E_crit 84.3 keV), P per proton 0.0505 +- 0.0016; proton 2 MeV, q_crit 3.7069 fC,
   P 0.9994. With Messenger's q_crit (0.6461 fC, E_crit 14.8 keV), P = 1 for both.

### A18 and A19, laptop run (24 Nov 2026)

Update 12 on the owner's laptop, committed as 304a0e9 on arch/v1; master in ~/sim_workspace fast-forwarded.
Fast tests 1880 passed, 13 skipped, 9 xfailed (35 min). Mixed-mode acceptance 41 passed, 2 xfailed (R111 peak,
R113 time, as recorded); G4SEE-profile acceptance 33 passed. Identical to the build machine: R110 to R113 dips,
R115 (0.9136 / 0.7537 / 0.6461 fC at 100 kOhm), R120, R121, R122, R124 (5 MeV: q_crit 3.6874 fC, P 0.0508;
2 MeV: 3.7057 fC, P 0.9994). One difference, all within the gates: the leakage at t = 0 with 10 kOhm is
round-off on both machines (laptop I_0 A = -3.4e-15 A, here +4.2e-16 A), and tr.Q subtracts it over 1 us, so
R111's charge differs by 2.3e-3 on the laptop (1.1e-4 here; limit 1e-2) and R114 at 10 kOhm by 3.5e-11 V (limit
1e-6 V). At small deposits (1e9 pairs/cm^2, about 1.6 aC per um^2) tr.Q therefore carries about 0.2 % from the
leakage's round-off.

### A20 (24 Nov 2026, lead): the crystal (channeling) implant in the process chain

The owner asked for the next three items: the channeling profiles in the chain (A20), phosphorus channeling from
Ghante et al. 1995 (A21), and implant damage in the anneal (A22).

0. **What the lead had seen.** A15 (the channeling engine: implant.concentration with a digitisation sigma per bin,
   normalised over [0, inf), refused when the target cuts off more than 1e-4), A16 (`initial_from_implant`, the
   anneal, `diode_from_anneal`) and their results. No crystal profile had been annealed when this was committed.
1. **Chain.** `initial_from_implant(result, dose_cm2=None, band=0)`: `band` +1 or -1 takes the concentration plus
   or minus its reported sigma at every point (clipped at 0): the digitisation envelope, not a 1-sigma profile (the
   reading errors are correlated along the profile). Nothing else changes: a channeling result goes through the
   A16 chain as an implant result.
2. **Acceptance (tolerances fixed here, before any run).**
   R130 (dose): B 35 keV, 1e14, tilt 7, rotation 30, crystal, target 1.5 um, depth bins 2 nm; As 120 keV, 1e13,
   tilt 7, rotation 22, crystal, target 1.0 um: the dose of `initial_from_implant` within 0.5 % of the beam dose.
   R131 (the two anneal engines on a crystal profile): each of R130's profiles annealed "fermi", B on n-type 1e16 at
   1000 C 1800 s, As on p-type 1e16 at 1000 C 600 s, thickness 2 um: junction depths of devsim_anneal and fipy
   within 1 % (A16.8).
   R132 (channeling deepens the junction): the same anneals from the amorphous RustBCA profile of the same beam
   (boron in "empirical" stopping, as A15's demo): the crystal junction deeper than the amorphous one, for both
   ions, on both engines.
   R133 (reported): junction depths from band -1, 0, +1 (devsim_anneal); the DEVSIM diode of the crystal B profile:
   J(0.6 V) and the Shockley fit, next to the amorphous one.
3. **Trust.** The anneal result is "verified" when the two engines agree (A16.8); the starting profile is A15's,
   "unverified" by rule. Each keeps its own level; the chain reports both.

### A21 (24 Nov 2026, lead): phosphorus channeling from the SIMS of Ghante et al. 1995 (our own fit)

0. **What the lead had seen.** V. Ghante, L. M. Lam, S. Morris, S.-H. Yang, A. F. Tasch et al., "The detailed dependence of
   implanted phosphorus profiles in (100) single-crystal Si on key implant parameters", MRS Symp. Proc. 396, 715
   (1995), DOI 10.1557/PROC-396-715, supplied by the owner (scan, 400 dpi, bitonal): SIMS only, no fitted
   parameters; the authors give the SIMS accuracy as better than 50 % in concentration and 10 % in depth; tilt
   angles corrected for the wafers' miscut. Figure 1 (15, 30, 50, 80, 120, 180 keV; 1e14; tilt 4, rotation 22),
   Figure 2 (50 keV; 1e13, 1e14, 5e14, 2e15, 8e15; tilt 10, rotation 0), Figures 3 to 6 (tilt and rotation series).
   The lead traced Figures 1 and 2 (a slope-continuity tracer, every trace checked on an overlay; tools/digitise
   to be committed with the data) and saw, while reading, that each trace integrates to its nominal dose within
   6 % (Figure 1: 0.94 to 1.01 x 1e14) and 2 % (Figure 2): the SIMS were normalised to the dose, and the reading
   holds. That is recorded here, not gated. No model had been fitted when items 0 to 7 were committed.
1. **Data** (`simws/data/channeling/ghante1995_phosphorus.json`): the traced points (depth um, cm^-3) of Figure 1,
   Figure 2, Figure 3 (the 10 deg curve: 30 keV, 1e14, tilt 10, rotation 0), Figure 6 (the 0 deg curve: 120 keV,
   1e14, tilt 10, rotation 0) and Figure 5 (the 0 deg curve: 50 keV, 5e14, tilt 10, rotation 0), the pixel
   calibration, the PDF's SHA-256. Points below the curve's noise floor (Figure 1: 2e17; Figures 2, 3, 5, 6: the
   floor named in the file per curve) are kept but not fitted.
2. **Model.** The form of A15's arsenic model (Suzuki 2000): (1 - f) n_a + f n_c, n_a a joined half Gaussian (Rpm,
   dRp1, dRp2), n_c the same with the tail exp(-((x - Rpm)/L)^alpha) beyond Rpm + dRp2, each normalised on
   [0, inf). The six parameters are fitted by the lead to each traced curve (least squares in log10 C over the
   points above the floor) and stored in the data file with the fit's residuals. Two conditions, and nothing
   between them: (A) tilt 4, rotation 22, dose 1e14, 15 to 180 keV (Figure 1), the parameters linear in energy;
   (B) tilt 10, rotation 0: at 50 keV the doses 1e13 to 8e15 (Figure 2), the parameters linear in log10 dose; at
   1e14 the energies 30, 50, 120 keV (Figures 3, 2, 6), linear in energy. In (B) a dose other than 1e14 is taken
   only at 50 keV. Every result carries `implant:own_fit: simws's fit to the SIMS of Ghante et al. 1995 (not a
   published model)` and `budget:sims_accuracy: the SIMS are good to 50 % in concentration and 10 % in depth`.
3. **Engine.** The channeling engine (A15) gains phosphorus (31P) under these two conditions (angles within 0.5
   deg). The uncertainty (A15.5's convention): every length moved by -/+ 10 % (the SIMS depth accuracy); the
   reported sigma is half the spread.
4. **Acceptance (tolerances fixed here, before any fit).**
   R140 (fit): every fitted curve within a factor 1.5 of its traced points above the floor at 95 % of them, and
   the depth where it falls to 1e18 cm^-3 within 10 % of the traced depth.
   R141 (energy, leave one out, A): the parameters at 50, 80 and 120 keV interpolated from their neighbours only
   (30 / 80, 50 / 120, 80 / 180): peak concentration within a factor 1.5, depth at 1e18 within 15 %.
   R142 (dose, leave one out, B at 50 keV): 5e14 from 1e14 / 2e15 and 2e15 from 5e14 / 8e15 (in log10 dose): the
   same limits as R141.
   R143 (the replicate): the model at (B) 50 keV, 5e14 against Figure 5's 0 deg curve (not fitted): peak within a
   factor 1.5, the depths at 1e18 and at 3e17 within 10 %.
   R144 (engine): normalisation and the A15 rules (thin target, refusals with "energy:", "dose:", "angles:");
   phosphorus outside both conditions refused.
   R145 (chain, reported): P 50 keV 1e14, condition (B), into p-type 1e16, 1000 C 600 s "fermi" on both anneal
   engines (junctions within 1 %), next to amorphous RustBCA.
5. **Trust.** "Unverified": one source, and Figure 5's curve is from the same study (R143 checks the reading and
   the fit, not the physics).
6. **Not in A21.** Other tilts and rotations (Figures 3 to 6 are checks only), doses away from 1e14 except at 50 keV
   in (B), 7 deg tilt.
7. **Corrections by the lead before any acceptance number (24 Nov).** (a) The first authors' initials were wrong in
   item 0: V. Ghante, L. M. Lam, S. Morris, S.-H. Yang, A. F. Tasch, D. Kamenitsa, J. Sheng, C. Magee.
8. (b) **The model has one component, not two (item 2 changed).** Fitting the six parameters of item 2 to each curve
   gave f = 1 (the upper bound) for 9 of the 13 curves and f between 0.19 and 0.86 for the others, with L and
   alpha trading against f: f is not determined by a single curve, and its jumps from curve to curve would make
   the interpolation of item 2 arbitrary. The fitted form is therefore Suzuki's n_c alone (the joined half
   Gaussian with the tail beyond Rpm + dRp2, kappa-joined), five parameters, normalised on [0, inf). Fit quality
   seen while deciding (the R140 quantity, before R141 to R143 were computed): rms of log10(model / trace) 0.014
   to 0.065, and 97.4 % to 100 % of the points within a factor 1.5. With one component there is no channeled
   share to report; `implant.channeled_fraction` for phosphorus is the share of the profile deeper than
   Rpm + 2 dRp2 (information only). Condition (B)'s energy set at 1e14 uses Figure 2's 1e14 curve at 50 keV.
   L and alpha still trade against each other between energies (L from 5 to 75 nm, alpha from 0.38 to 0.90);
   R141 and R142 test what that does to interpolation.
9. **After the checker's report (24 Nov).** 179 tests written from A20 and A21 (tests/v1/test_crystal_chain.py,
   tests/v1/test_phosphorus_channeling.py); 176 passed and 3 failed at first, all three from one defect.
   **Defect, found by the checker and fixed:** condition (A) was not normalised on [0, inf) at 30, 120 and 180 keV
   (the integral was 1 + 2.9e-4, 4.1e-4, 2.6e-4; limit 1e-4): the normalisation grid stopped at Rpm + 60 max(L,
   dRp2), but with alpha 0.38 to 0.47 and L 5 to 14 nm the tail carries 3e-4 beyond it. The grid now reaches Rpm +
   L 80^(1/alpha) (geometric beyond the core), as does the information-only tail share. Also fixed: phosphorus
   results carried A15's `implant:published_fit` flag next to `implant:own_fit` (now only the own-fit and SIMS
   flags; the conditions are in the diagnostics); condition (B) took doses other than 1e14 within 0.5 keV of 50
   keV, against "only at 50 keV" (now within 1e-6 keV); the adapter's description of channeled_fraction now says
   what it is for phosphorus. A15's R87 test expected "ion:" for phosphorus; since A21 it is "angles:" (amended).
   **Results:** A20: R130 B +0.0008 %, As -0.0020 % (0.5 %); R131 0.0013 % and 0.0007 % (1 %); R132 crystal deeper
   for both ions on both engines (B 490.8 against 431.3 nm, As 326.3 against 179.9 nm); R133 bands B 489.8 / 490.8 /
   491.7 nm, As 318.7 / 326.3 / 333.8 nm, diodes J(0.6 V) 1.0986e4 against 1.1000e4 A/m^2. A21: R140 at least 97.4 %
   of points within a factor 1.5 (95 %), depths at 1e18 within 2.8 % (10 %); R141 peaks 0.955 to 1.035, depths
   within 2.5 % (15 %); R142 5e14: peak 0.932, depth +14.2 % (15 %: close), 2e15: 0.990, +9.6 %; R143 peak 0.952,
   depths +2.8 % and +2.7 % (10 %); R144 refusals as stated; R145 junctions 442.86 nm on both engines against
   255.5 nm amorphous. The traced points lie within 2 px of the ink on the scan.

### A22 (24 Nov 2026, lead): transient enhanced diffusion (TED) from implant damage, an effective "+1" model

0. **What the lead had seen.** (i) P. A. Stolk, H.-J. Gossmann, D. J. Eaglesham et al., "Physical mechanisms of
   transient enhanced dopant diffusion in ion-implanted silicon", J. Appl. Phys. 81, 6031 (1997), the copy at
   www.ele.uva.es/~simulacion/StolkJAP97.pdf (SHA-256 52783d9f4a39...): {311} defects hold about one interstitial
   per implanted ion (the "+1" rule, Figs. 6-7); they dissolve with an activation energy of 3.8 +- 0.2 eV, decay
   times about 40 s at 815 C and about 4 h at 670 C (40 keV Si, 5e13 cm^-2, Sec. III); B marker layers show a
   time-averaged diffusivity enhancement near the implanted region of about 200 after 10 min and about 60 after
   30 min at 790 C (Sec. IV, Fig. 12a); Fig. 1, boron 30 keV 1.5e14 cm^-2 as implanted and after 800 C 35 min
   (SIMS). (ii) EE212 class handout #14 (Plummer, Deal, Griffin, "Ion implantation", 1999, pages 19-25; the same
   site as A16's handout, SHA-256 6dfe1738282d...): the +1 model, eq. (20) (the supersaturation during {311}
   dissolution, a function of temperature only) and its plot (about 1.3e5 at 600 C and 3.5e2 at 1100 C), eq.
   (21) (tau_enh proportional to Rp Q exp(Eb/kT)). (iii) Handout #12 (A16's), eq. (22) D = D* (f_I C_I/C_I* +
   f_V C_V/C_V*) and its table f_I: B 1.0, P 1.0, As 0.4, Sb 0.02 (f_V = 1 - f_I). No TED had been computed when
   items 0 to 7 were committed.
1. **Description.** `AnnealModels.ted`: "none" (default; left out of the serialised description, so no A16 id
   changes) or "plus_one". With "plus_one", the implant's damage is taken from the initial profile itself: Q its
   dose and Rp its mean depth.
2. **Model (both engines).** While the {311} clusters last, the interstitial supersaturation is S(T), uniform in
   depth, and every diffusivity of A16.2 is multiplied by f_I S + f_V / S (C_V/C_V* = 1/S, local I-V equilibrium);
   afterwards by 1. The clusters last tau(T) = tau_ref (Q / Q_ref) (Rp / Rp_ref) exp[E_tau/k (1/T - 1/T_ref)] with
   tau_ref = 40 s, T_ref = 815 C, Q_ref = 5e13 cm^-2, Rp_ref the mean depth of 40 keV Si in amorphous Si from
   RustBCA (A7, kr_c, Lindhard-Scharff; computed once and stored with the parameters), E_tau = 3.8 eV (Stolk;
   the Q Rp scaling is eq. 21's). Over several steps the clusters are used up in proportion: each step consumes
   its duration over its tau, and TED ends when the sum reaches 1. S(T) = S_ref exp[E_S/k (1/T - 1/T_S)], T_S =
   790 C, E_S from the handout's plot (its two end points: ln(1.3e5 / 3.5e2) k / (1/873.15 - 1/1373.15) K, = 1.22
   eV), S_ref calibrated on Stolk's two marker numbers at 790 C for his implant: for an anneal of length t longer
   than tau, the time-averaged enhancement is (S tau + (t - tau)) / t, so S = (E t - t + tau) / tau; S_ref is the
   mean of the values from (200, 10 min) and (60, 30 min). Every TED result carries `anneal:ted_effective: the
   +1 model with a uniform supersaturation; tau and S from Stolk et al. 1997 and the EE212 handout (simws's
   calibration); no clustering (boron above about 1e18 cm^-3 is immobile in reality), no depth dependence`.
   The time points of A16.1 plus the moment TED ends, when that falls inside a step.
3. **Engines.** devsim_anneal and fipy, both.
4. **Acceptance (tolerances fixed here, before any TED run).**
   R150 (against Stolk's Figure 1, the validation): the as-implanted SIMS of Fig. 1 (traced) annealed at 800 C
   for 35 min, "fermi" with "plus_one", on an n-type 1e15 background, thickness 1.5 um: the depths where the
   result falls to 3e17 and to 1e17 cm^-3 within 15 % of the diffused SIMS of Fig. 1 (traced). Reported: the same
   without TED, and the peak (not gated: clustering is not modelled).
   R151 (the two engines with TED): R150's case and B 35 keV 1e14 (A20's crystal profile) at 1000 C 10 s: junction
   or 1e17 depths within 1 %; doses within the R94 rule of A16.10.
   R152 (limits): "plus_one" with tau -> 0 (an anneal far longer than tau at high T, 1100 C 1 h) moves the
   junction by less than 2 % compared with "none" (TED is transient); the enhancement factor equals f_I S + f_V/S
   at the step's temperature (checked through a public function); tau and S at 815 C / 790 C equal their
   reference values.
   R153 (reported): B 35 keV 1e14 crystal (A20) on n-type 1e16, 900 C 10 s, 1000 C 10 s, 1000 C 30 min: junction
   with and without TED.
5. **Trust.** Both engines: "verified" as A16.8 (they agree). R150 compares against a measurement not used for
   the calibration (Fig. 1 is a different implant: B 30 keV 1.5e14 against Si 40 keV 5e13, 800 C against 790 C):
   a passed R150 recorded as a validation (A13.7) makes the R150 result "validated"; nothing else here is.
6. **Not in A22.** Boron clustering, depth-dependent interstitials, dislocation loops (doses above about 1e14 Si
   equivalent), amorphising implants and end-of-range damage, oxidation-enhanced diffusion.
7. **Clarifications by the lead during development, before the checker (24 Nov).** (a) Stolk's Fig. 1 traced
   (simws/data/ted/stolk1997_fig1.json; tools/digitise/tracer.py on the figure's image, checked on an overlay):
   the traces stop where they first fall below 4e16 cm^-3 past 0.3 um (the SIMS noise floor); for the anneal's
   initial profile the as-implanted trace is extended to 0.7 um with an exponential fitted to its last part (0.33
   um to the end: -3.14 decades per um), recorded in the file. The traced as-implanted dose is 1.14e14 cm^-2
   against the nominal 1.5e14 (SIMS calibration and the trace; recorded, not gated); A22.1 takes Q from the
   profile, so tau uses 1.14e14. (b) Rp_ref computed as item 2 says: 59.02 +- 0.18 nm (RustBCA, 40 keV 28Si,
   Kr-C, Lindhard-Scharff, 20 000 ions, seed 1). S_ref = 1088 (1152 from 10 min, 1025 from 30 min); E_S = 1.2227 eV.
8. (c) **R150's record.** So that R150 can be stored as a comparison with a measurement (item 5), the anneal
   adapters also report `anneal.depth_at_1e17` and `anneal.depth_at_3e17` (m; the deepest depth at or above the
   level, linear between points; only when the profile starts at or above the level), and the demo ingests the two
   traced SIMS depths as a measured result (`simws.measured.ingest`, label "validation") and records the
   comparisons (relative, 15 %).
9. **After the checker's report (24 Nov).** 44 tests written from A22 (tests/v1/test_ted.py); 42 passed and 2
   failed at first, both from one defect. **Defect, fixed:** `ted.damage` took the mean depth as trapezoid(c x) /
   trapezoid(c) over the listed points, leaving out the part held from 0 to the first point and approximating the
   linear segments (a flat 1e18 profile on 0-0.1 um gave 75 nm, not 50 nm). It now uses the exact moments of
   A16.1's profile. Effect on the acceptance cases: Stolk's Rp -0.34 % (tau 343.7 s -> about 342.5 s), the crystal
   profile -0.05 %; no gated number moves noticeably. **Fixed with it:** TED results carried A16.2's flag "not
   modelled: transient enhanced diffusion ..." next to the TED flag; with TED on it now reads "not modelled:
   clustering, solid solubility, oxidation and segregation (TED by the effective model of A22)"; the engines'
   capability notes and chain.process's docstring say the same.
   **Results:** R150 both engines -8.8 % at 3e17 and -10.9 % at 1e17 (limit 15 %); without TED -21.8 % and -21.5 %;
   the model moves the tail 0.60 (3e17) and 0.50 (1e17) of the measured shift: half the measured TED. The 15 % gate
   is not very discriminating at this temperature (no TED misses by only 22 %); the shift ratio is the honest
   figure. R151 junctions within 0.052 %, x(1e17) within 0.0012 %, doses by the A16.10 rule; R152 1100 C 1 h with
   TED moves the junction by 0.023 % (2 %), enhancement, tau(815 C) = 40 s and S(790 C) = S_ref exact; R153: 900 C
   10 s 422.5 -> 431.0 nm, 1000 C 10 s 422.9 -> 425.5 nm, 1000 C 30 min 490.8 -> 493.0 nm. Trust: the devsim TED
   result of R150 is "validated" after its two measurement comparisons are recorded; a wrong measurement blocks it
   (tested). Sources: the checker re-read every constant (handout #14's curve end points read at 400 dpi: 1.30e5 at
   600 C, 3.6e2 at 1100 C, E_S 1.20 eV from raw pixels against 1.2227 from the rounded values; Rp_ref re-run:
   59.020 nm). Reported: with E_tau = 3.8 eV anchored at 40 s / 815 C, tau(670 C) is 5.6 h against Stolk's "about
   4 h" (his two decay times alone imply about 3.6 eV, inside his +- 0.2).

### A20, A21 and A22, laptop run (24 Nov 2026)

Update 13 on the owner's laptop, committed as b7aaadf on arch/v1; master in ~/sim_workspace fast-forwarded. Fast
tests 2076 passed, 32 skipped, 9 xfailed (11 min 45 s). Crystal-chain and phosphorus acceptance 163 passed, 16
skipped; TED acceptance 41 passed, 3 skipped. Identical to the build machine: R130 (B +0.0008 %, As -0.0020 %), R131
(0.0013 %, 0.0007 %), R132 (B 490.76 / 431.29 nm, As 326.32 / 179.90 nm), R133 bands and diodes, R140 to R143 (R142
5e14: peak 0.932, depth +14.22 % against 15 %), R145 (442.86 nm), R150 (-8.9 % and -11.0 % with TED, -21.8 % and
-21.5 % without; shift ratios 0.594 and 0.496), R151 (0.0517 %, 0.0018 %), R152 (0.023 %; tau(670 C) 5.64 h), R153
(900 C 10 s 422.5 -> 431.0 nm). The R150 DEVSIM TED result is "validated"; the FiPy one "verified".

### A23 (24 Nov 2026, lead): validation data for the IBM 5AM SiGe HBT, and the 5AM device at 300 K

The owner chose validation against measurement over new features (the plan "simws validation plan: IBM 5AM SiGe
HBT", 24 Nov 2026: one device, the IBM 5AM, in A23 to A29; each stage fits at most three named knobs to one
dataset and is judged on data it did not see). A23 stores the data and builds the device; A24 runs it cold.

0. **What the lead had seen.** (i) J. A. Pellish, PhD thesis, Vanderbilt, Dec 2008 (free copy, SHA-256 in the data
   file): Fig. 3, Gummel and beta of a 0.5 x 2.5 um^2 5AM HBT (peak beta about 165 near VBE 0.62 V; beta about 125
   at 0.8 V); Fig. 4, forced-IB output curves (IC about 0.78 mA at IB 6 uA and 1.35 mA at 12 uA; a rise from about
   3 V). (ii) L. Luo, PhD thesis, Auburn, 2011 (free copy): the device is a 5AM 0.5 x 2.5 um^2 HBT (his Figs. 6.53,
   7.3); Fig. 1.1 cross section; Fig. 1.2 SIMS of As, B, P and Ge with **no numbers on the depth axis** (Ge peak
   about 0.098, B peak about 2.5e18 cm^-3, collector P about 5e17); Fig. 4.1 intrinsic base sheet resistance vs T
   (about 1.4e4 ohm/sq at 300 K, rising steeply below about 100 K); Fig. 4.2 M-1 vs VCB at a small forced VBE,
   43 to 300 K; Fig. 4.7 beta vs VBE at VCB = 0, 43 to 300 K (peak about 130 near 0.77 V at 300 K, about 190
   near 1.03 V at 93 K); Fig. 5.7 IC vs VBE, 43 to 393 K, and his extracted ideality (1 above about 200 K, 1.35 at
   43 K; "default Mextram fails below 110 K"); Fig. 3.7(a) fT vs JE at 300 K (peak about 47 GHz near 1 mA/um^2;
   the colder panels have no numbers on their fT axis); Fig. 6.38, an extracted RE of order 10 ohm. (iii) Sutton
   thesis (Georgia Tech 2009) section 2.4.2 and Table 2 (the SiGe film: 10-20 nm buffer, 70-100 nm boron layer,
   10-30 nm cap; effective emitter width 0.42 um), Fig. 72 (pre-irradiation 77 K and 300 K Gummel of a 0.5 x 1.0
   5AM). (iv) Cressler et al. 2001 Table I (beta 115, VA 60 V; another lot); Lourenco thesis Table 6.1 (BVCEO 3.3 V,
   BVCBO 10.5 V). The two chips' peak beta differ by about 25 % (Pellish 165, Luo 130). No 5AM simulation had been
   run when items 0 to 6 were committed.
1. **Data** (`simws/data/validation/`). One JSON file per source (`pellish2008.json`, `luo2011.json`,
   `sutton2009.json`), each with the source's citation and the PDF's SHA-256, and per figure: the traced points with
   units, the axis calibration, the method (`vector`: read from the PDF's drawing operators, the axes calibrated on
   the tick marks in the same stream; or `raster`: tools/digitise as A21), and the trace error (vector: the line
   width; raster: as A21). Every trace is checked on an overlay of the page. Noisy SIMS traces carry the floor below
   which points are not used. `catalog.json` lists every dataset of the plan (paper, figure, device, conditions,
   role: "calibration", "validation", "trend" or "input"). `simws.validation.catalog`: `load(dataset)` and
   `ingest(hub, dataset)` (`simws.measured.ingest` with the catalog's role as the label; trend and input datasets
   are refused).
2. **Device** (`simws.spec.library.ibm5am_hbt(s_um, k_B, k_E, L_um=2.5, T=300.0, VBE=None, VCB=0.0, ...)`). A 2D
   half cell (x = 0 is the symmetry plane through the emitter's centre; currents x 2 x L), x from 0 to 1.2 um, y
   from the top of the monocrystalline emitter (Luo's dashed poly/mono line). Doping: Luo Fig. 1.2 traced, every
   depth u of the figure (0 to 1 across the axis) placed at y = s_um (u - u_poly); As (x <= 0.21 um only, x k_E) as
   the emitter; B (x k_B) as the base, across the cell; P as the collector; the buried As held at its last traced
   value for 0.5 um below the trace, to the collector contact on the bottom. Ge from the traced Ge curve (linear
   GradeTerms between its points). Extrinsic base: acceptors 1e20 for x >= 0.36 um (a 0.15 um spacer) down to the
   depth where the traced B falls below 1e17 on the collector side. Contacts: emitter on top over x <= 0.21 um
   (ohmic, the poly is not modelled), base on top over x >= 0.8 um, collector on the bottom. Physics, at every
   temperature (A24 changes only T): Fermi-Dirac, Klaassen mobility, Altermatt incomplete ionization, SRH
   "scharfetter", Auger, Slotboom; impact ionization (van Overstraeten) only where an item says so. Flags:
   `device:5am_reconstruction: depths of Luo Fig. 1.2 scaled by a fitted s; lateral layout chosen by the lead`
   and `device:no_parasitic_resistance: no poly emitter, no RE, no external RB, RC`.
3. **Knobs (three, fitted together).** s_um (micrometres per unit of Fig. 1.2's depth axis; 0.3 to 1.2), k_B (0.5
   to 2), k_E (0.3 to 3). The fit: least squares in log10 of IC (Luo Fig. 5.7a, 300 K, VBE 0.60 to 0.78 V), IB =
   IC / beta (Fig. 4.7, 300 K, VBE 0.70 to 0.78 V) and RsBi (Fig. 4.1 at 300 K), each set weighted equally. RsBi of
   the model: 1 / (q integral of mu_p p dy) over the equilibrium solution of the vertical cut at x = 0, between the
   two metallurgical junctions, mu_p and p from the device's own models. Stored in
   `simws/data/validation/ibm5am_calibration.json` with the residuals. Above 0.78 V the emitter resistance (not
   modelled; about 10 ohm x IE) moves VBE by more than 1.5 mV: those points are reported, not fitted or gated.
4. **Acceptance (tolerances fixed here, before any 5AM run).**
   R160 (data): every dataset of the catalog loads, has a source, units and a role; the files' SHA-256 match the
   PDFs the lead read; `ingest` labels as the catalog says and refuses trend and input datasets.
   R161 (the fit, calibration data): IC within 15 % at every traced VBE in 0.60-0.78 V; IB within 20 % in
   0.70-0.78 V; RsBi(300 K) within 10 %.
   R162 (another chip, Pellish Fig. 3, held out): IC within 25 % at every traced VBE in 0.60-0.78 V. Reported:
   IB and beta (the chips' beta differ by about 25 %: item 0).
   R163 (Early voltage, Pellish Fig. 4, held out): V_A = IC / slope from a straight line through the traced IC
   over VCE 0.6 to 2.0 V at IB 6 and 12 uA; the model's V_A (the same fit over the same VCE at the VBE that gives
   that IB at VCE 1 V) within a factor 1.5.
   R164 (avalanche, Luo Fig. 4.2 at 300 K, held out): impact ionization on, VBE fixed where IC is 10 uA at VCB 0:
   M - 1 = (IB(0) - IB(VCB)) / IC(VCB) at VCB 2, 3, 4 and 5 V within a factor 2.
   R165 (fT, Luo Fig. 3.7a, held out): small signal at VCB 0 V, fT = f |h21| at 2 GHz: the peak fT within 25 %;
   the JE at the peak within a factor 2.
   R166 (mesh): IC and IB at VBE 0.70 V change by less than 3 % with mesh scale 0.5.
   R167 (reported): IC at 0.60 and 0.65 V next to the generalized Moll-Ross / Kroemer integral of the vertical cut
   (`simws.validation.sige_checks`), x the emitter width; the SiGe film the fitted s implies (Ge above 1 %) next to
   Sutton's 90 to 150 nm.
   R168 (refusals and ids): knobs outside their ranges refused ("device:"); the problem id fixed by the knobs, L
   and T; a changed data file changes the id.
   R169 (trust): the 300 K results whose comparisons R162 to R165 pass become "validated" once those comparisons
   are recorded against the ingested "validation" data; R161 is recorded against "calibration" data and validates
   nothing; a failed comparison keeps a result below "validated".
5. **Trust.** One engine (Charon has no grading): no engine pair. "Validated" only through R169.
6. **Not in A23.** Strikes, dose, substrate and deep trench (A25 to A27), the poly emitter and parasitic
   resistances, self-heating, 3D effects.
7. **Clarifications by the lead during development, before the checker (24 Nov).**
   (a) **Tracing.** Every figure is vector: the data are the plotted paths' vertices read from the PDFs' drawing
   operators, the axes calibrated on tick marks in the same stream (`tools/digitise/vector.py`,
   `tools/digitise/trace_5am.py`, which writes the three data files); every trace checked on an overlay. Read
   while tracing: Pellish Fig. 3's peak beta is 157 at 0.62 V (item 0's "about 165" was by eye); at 1 uA the two
   chips' VBE agree within 2 mV (Pellish 0.700 V, Luo 0.698 V). Luo Fig. 1.2: the depth axis has 8 equal tick
   intervals; the Ge film (Ge > 1 %) spans 1.10 of them (110 nm at 0.1 um per interval); the poly / mono line is
   at u = 0.0484; the As trace ends at u = 0.075 at 2.3e20 cm^-3. Fig. 5.7(a)'s twelve curves are named by Fig.
   5.7(b)'s markers (393, 300, 262, 223, 192, 162, 136, 110, 93, 76, 60, 43 K). Fig. 4.1: the legend's own marker
   is excluded. Fig. 4.2: the 43 K curve is flat above about 3 V (Luo: instability). Fig. 3.7(a): the fT axis has
   no tick marks in the stream and is calibrated on its labels. Sutton Fig. 72: each curve is drawn one segment
   per path; the segments are chained.
   (b) **Engine.** A23.2's Ge needs several grading terms: the DEVSIM translator now takes several, in order, not
   overlapping, of one element, axis and mode, evaluated at the nodes into a node solution; one term is
   translated as before (no id or result changes).
   (c) **Emitter.** A tabulated profile cannot be confined laterally, so the emitter is a box (x <= 0.21 um, from
   the poly / mono line to the trace's end) at the trace's mean concentration there x k_E.
   (d) **Mesh.** The first mesh (0.5 nm at the emitter edge in depth, 2 nm laterally) took 330 s for 5 Gummel
   points; it was coarsened before the fit (1 nm, 4 nm). Between the two meshes at the start knobs the lead saw IC
   1.5 % and IB 5 % apart (R166's quantity, on another pair of meshes).
   (e) **RsBi** is computed from the engine's equilibrium state, not as a hub quantity: R161's and R172's RsBi
   comparisons are in the tests and not recorded in the hub; R169's records are the IC curves.
   (f) **R164 and R173, method changed.** Impact ionization in the coupled solve did not converge at equilibrium on
   this device (with the source added before the equilibrium solve, and with it added after; the second change
   was tried and reverted). M - 1 is computed instead as the electron-initiated multiplication along the cut
   x = 0, from the base's most p-type point through the collector junction: 1 - 1/M = integral alpha_n
   exp(-integral (alpha_n - alpha_p)) dy, van Overstraeten-de Man at T (`simws.tcad.models.alpha_numpy`), on the DC
   solution without impact ionization (IC has no avalanche feedback). While debugging it the lead saw R164's
   numbers on a device near the final knobs (s 0.82, k_B 0.63, k_E 2.78): model M - 1 at VCB 0 to 5 V 7.8e-6,
   0.0023, 0.021, 0.075, 0.176, 0.34; measured at 1 to 5 V 1.2e-4, 0.0022, 0.019, 0.076, 0.19. (The lead first
   read them one VCB step out of line, as agreement; corrected after the demo's R164 line: the model at 2 V is
   0.021, about 10 x the measurement.) The tolerance is unchanged.
   (g) **The fit** (R161's quantities seen): 13 evaluations of five Gummel points (0.60, 0.66, 0.70, 0.74, 0.78 V)
   with the equilibrium RsBi; it stopped on its step tolerance at s 0.8256 um, k_B 0.6256, k_E 2.7842 (103 nm per
   tick interval; a 114 nm SiGe film). IC 1.22 to 1.09 x the measurement from 0.60 to 0.78 V, IB 1.54 to 1.64 x,
   RsBi 0.89 x. k_E moves IB little (d log IB / d log k_E about -0.2): an ohmic contact at the poly / mono line
   (the poly emitter not modelled, item 2's flag) gives an emitter-side base current that k_E's range cannot bring
   down to the measured one. RsBi and IC pull k_B opposite ways (less boron raises RsBi toward the measurement and
   IC away from it): the SiGe hole mobility, provisional in simws, is the first suspect. Recorded, not refitted.
   (h) JE (R165) and JC (R174) are over the drawn emitter area, 0.5 x L.
   (i) fT bias points (R165): VBE 0.80 to 0.98 V in 20 mV steps, VCB 0.
   (j) **The public functions** (for the checker): `simws.validation.catalog`: `catalog()`, `entry(id)`,
   `role(id, T_K=None)` (a figure's curve at 300 K can be "calibration" while the others are "validation"),
   `load(id, T_K=None)` / `curve(...)` (arrays; `figure(id)` the raw block and the source), `ingest(hub, id,
   axis=(name, key, unit), columns={quantity: (key, unit)}, T_K=None, rows=None)`. Dataset ids: luo2011.fig1_2,
   .fig5_7a, .fig4_7, .fig4_1, .fig4_2, .fig3_7a; pellish2008.fig3, .fig4; sutton2009.fig72 (traced), and the
   later stages' rows (no file). `simws.spec.library`: `ibm5am_hbt(...)` (item 2), `ibm5am_scale(L_um)` (cm of
   depth -> one device), `IBM5AM_RANGES`. `simws.validation.ibm5am`: `knobs()` (the stored calibration),
   `gummel(k, VBE, T=300, L_um=2.5, VCB=0, mesh_scale=1)` -> VBE_V, IC_A, IB_A (one device), RsBi_ohm_sq;
   `rsbi(k, T)`; `early_voltage(k, IB_target_A)`, `early_voltage_measured(IB_uA)`; `avalanche(k, T=300,
   VCB=(...), VBE_guess=None)` -> M_minus_1 per VCB; `ionization_integral(y_cm, E_Vcm, T)`; `ft_curve(k, VBE)` ->
   VBE_V, JE_mA_um2, fT_GHz; `moll_ross(k, VBE)`; `luo_targets(T)`, `gummel_window(T)`, `vbe_at_current(V, I,
   target)`; `fit()` (writes the calibration file; slow). `tools/v1_5am_demo.py --part a23|a24` prints the R
   lines. `simws.engines.devsim_engine.grade_values(grading, coord_um)` (item (b)).

### A24 (24 Nov 2026, lead): the 5AM device from 77 K to 393 K, nothing refitted

0. **What the lead had seen.** As A23.0; in particular Luo's beta peaks (item 0 (ii)) and his statement that the
   IC slope departs from ideal below about 200 K; Sutton Fig. 72. No cold 5AM run when committed.
1. **Runs.** A23's device with its stored knobs, unchanged, at 393, 223, 162, 93 K (Luo's temperatures) and 77 K
   (Sutton's), plus 43 K reported. Same models (A23.2); the engine's extended precision below 200 K.
2. **Acceptance (tolerances fixed here, before any cold run).**
   R170 (IC, Luo Fig. 5.7a): VBE at IC = 1 uA and at 100 uA within 20 mV of the trace, at 393, 223, 162 and 93 K.
   R171 (beta, Luo Fig. 4.7): the peak beta within 30 % at 223 and 162 K and within a factor 2 at 93 K. Reported:
   the VBE of the peak.
   R172 (base sheet resistance, Luo Fig. 4.1): RsBi within 30 % at 223, 162 and 100 K and within a factor 2 at 75 K;
   reported down to the trace's lowest temperature (incomplete ionization).
   R173 (avalanche, Luo Fig. 4.2): A23's R164 at 162 and 93 K, VCB 2, 3 and 4 V, within a factor 2.
   R174 (another chip, Sutton Fig. 72, pre-irradiation, 0.5 x 1.0 um^2, L_um 1.0): VBE at JC = 10 nA/um^2 within
   30 mV of the trace at 300 K and at 77 K.
   R175 (reported): 43 K (or its failure category); the model's IC ideality next to Luo's (Fig. 5.7b) at 93 and
   43 K.
3. **Trust.** As A23.5: each temperature's result is "validated" when its comparisons pass and are recorded.
4. **Expected trouble.** Below about 110 K, base current gains a non-ideal part (Luo: trap-assisted tunnelling)
   that drift-diffusion does not hold; a FAIL there is recorded, not tuned away.

### A23 and A24, build-machine results (24 Nov 2026, before the checker)

`tools/v1_5am_demo.py`, stored knobs s 0.8256 um, k_B 0.6256, k_E 2.7842.
- R161 (calibration): IC 1.217 to 1.095 x (FAIL at 0.60 to 0.66 V, 15 %); IB 1.54 to 1.64 x (FAIL, 20 %); RsBi
  12739 against 14284 ohm/sq, 0.892 (FAIL, 10 %).
- R162 (Pellish's chip): IC 1.198 to 1.305 x (FAIL at 0.60 to 0.70 V, 25 %); IB 2.03 to 2.08 x, model beta 86 to 98
  (reported).
- R163: V_A 28.2 V against 140.1 V (IB 6 uA) and 30.3 against 268.7 V (12 uA): 0.20 and 0.11 (FAIL, factor 1.5).
- R164: M - 1 at 2, 3, 4, 5 V: 0.0209, 0.0735, 0.173, 0.336 against 0.00222, 0.019, 0.0764, 0.19: 9.4, 3.9, 2.3, 1.8
  (FAIL at 2 to 4 V, factor 2).
- R165: peak fT 59.6 GHz at 1.22 mA/um^2 against 46.9 GHz at 0.94: +27 % (FAIL, 25 %); JE within a factor 2 (pass).
- R166: mesh scale 0.5 at 0.70 V: IC -1.52 % (pass), IB -15.9 % (FAIL, 3 %): the base current, and so k_E, is not
  mesh-converged.
- R167 (reported): IC / Moll-Ross 0.789 and 0.754 at 0.60 and 0.65 V; SiGe film 114 nm.
- R170: VBE at 1 uA / 100 uA: 393 K -8.7 / -6.2 mV, 223 K -4.0 / -2.6 mV, 162 K +6.2 / +4.9 mV (pass, 20 mV);
  93 K no converged solution (FAIL; the equilibrium solve fails within 36 s, also with 20 mV steps).
- R171: peak beta 0.728 x at 223 K (pass, 30 %), 0.981 x at 162 K (pass); 93 K no result (FAIL).
- R172: RsBi 0.925 x at 223 K, 0.874 x at 162 K (pass, 30 %); 100 K and 75 K no result (FAIL).
- R173: M - 1 at 162 K: 0.0513, 0.147, 0.314 against 0.00495, 0.0398, 0.13 (FAIL); 93 K no result (FAIL).
- R174: Sutton 300 K +1.1 mV (pass, 30 mV); 77 K no result (FAIL).
- R175: 43 K no result.

What the numbers say. The collector current and its temperature dependence hold from 393 to 162 K within 9 mV,
with three knobs fitted at 300 K only, and Sutton's other chip at 300 K within 1 mV. The rest points at three
gaps: (1) the base-collector side of the rebuilt profile: V_A 5 to 9 x too low and avalanche at 2 to 3 V 4 to 10
x too strong both say the base-collector junction is too abrupt or the collector too heavily doped there (the
traced boron tail and collector phosphorus near the junction); (2) the emitter: the base current is neither
mesh-converged (-16 %) nor reachable with k_E (no poly emitter); (3) the solver below about 100 K on this device.

### A23b (24 Nov 2026, lead): the 5AM device revised (poly-emitter contact, collector phosphorus, mesh)

The owner chose to fix the device before shipping (A23, A24 and A23b go in one update). A23's knobs, data,
tolerances and hold-outs stay; what changes is the device, its knobs and its mesh.

0. **What the lead had seen** (beyond A23.0 and A23.7): every A23 / A24 build-machine result above. Diagnostics:
   at 93 K the equilibrium converges with Boltzmann statistics and fails with Fermi-Dirac at the fitted emitter
   (N_E = 1.78e21 cm^-3, k_E 2.78), with or without incomplete ionization, Klaassen, SRH / Auger or BGN; with the
   emitter at the trace's mean (k_E = 1, 6.4e20) Fermi-Dirac converges at 93 and 77 K (43 K: a root bracket fails
   in the contact equilibrium). Mesh at the A23 knobs, VBE 0.70 V: IB -7.1 % at scale 0.7 and -15.9 % at 0.5 (IC
   -1.1 %, -1.5 %). BVCBO 10.5 V (Lourenco Table 6.1) against the traced collector phosphorus next to the base
   (about 3.8e17 cm^-3): an abrupt one-sided junction at 3.8e17 breaks down near 4 V, at 1e17 near 10.7 V. The
   new contact's closed-form check below (item 1), run before this was written.
1. **Engine: a finite hole recombination velocity at an ohmic contact.** `Contact.hole_recombination_velocity_cm_s`
   (> 0, ohmic contacts only; None = ohmic, and absent from the description so no id changes). DEVSIM: the hole
   equation at that contact keeps the bulk flux and adds q S (p - p_eq) per unit contact area (a contact node
   model is integrated over the node's volume, so the flux is divided by it). Charon refuses it. Closed form
   (1D p+ n diode, NA 1e20 / ND 1e17, n side 2 um, no recombination, constant mobility, Boltzmann, no BGN, 0.5 V):
   J_p = q p_n0 (e^{V/Vt} - 1) / (W'/D_p + 1/S), W' the neutral n width. Seen: model / closed form 1.0000,
   0.9999, 0.9992, 0.9968 at S = 1e2, 1e3, 1e4, 1e5 cm/s; at 1e8 the ohmic result within 0.05 %; ohmic 0.9951 (the
   closed form's depletion edge); the same diode in 2D gives the 1D currents to five digits.
2. **Device.** `ibm5am_hbt(..., k_P=1.0, S_p=None)`: k_P multiplies the traced collector phosphorus; S_p (cm/s) is
   the emitter contact's hole recombination velocity (the poly emitter's effective contact). A23b fixes s_um =
   0.80 (0.1 um per tick interval of Luo Fig. 1.2: the 110 nm Ge film of A23.7(a) and Sutton's 90 to 150 nm; A23's
   fit gave 0.826) and k_E = 1 (the emitter at the trace's mean). Mesh: the lead refines the mesh around the
   emitter before the fit until IB at 0.70 V changes by less than 3 % at mesh scale 0.5 (at the start knobs).
3. **Knobs (three, fitted together).** k_B (0.5 to 2), k_P (0.1 to 2), S_p (1e3 to 1e7 cm/s, fitted in log10).
   Data: A23.3's three sets (Luo 300 K IC, IB, RsBi) and BVCBO = 10.5 V (Lourenco Table 6.1, from Ahlgren et al.
   1997; a table value, weighted as one set). Model BVCBO: emitter and base at 0 V, collector swept; the voltage
   where 1 - 1/M of A23.7(f)'s integral on the cut x = 0 reaches 1 (interpolated). Stored in
   `ibm5am_calibration.json` (A23's calibration kept in the file's history in git).
4. **Acceptance.** A23.4's R161 to R169 and A24's R170 to R175, re-run with the A23b device, the same tolerances.
   New: R180 (the contact, closed form: model / closed form within 1 % at S = 1e2 to 1e5 cm/s; S = 1e8 equals the
   ohmic contact within 0.5 %); R181 (the same diode in 2D gives the 1D current within 0.1 %); R182 (refusals and
   ids: S <= 0 and S on a schottky contact refused; Charon refuses it; every library problem without S keeps its
   id); R183 (calibration: model BVCBO within 10 % of 10.5 V).
5. **Kept as history.** A23's device results above stay in this contract; the tests test the A23b device.
6. **Clarifications by the lead during development, before the checker (24 Nov).**
   (a) **Mesh.** With the A23b emitter (k_E 1, the contact's S_p) the mesh of A23 already meets item 2: at the
   start knobs (k_B 0.63, k_P 0.3, S_p 3e4 cm/s) and 0.70 V, mesh scale 0.5 moves IB +1.5 % and IC -1.4 % (beta 297
   at that S_p). The A23 sensitivity came from the 1.78e21 emitter at an ohmic contact. The mesh is unchanged.
   (b) **BVCBO.** Item 0's estimate (an abrupt junction at 3.8e17 breaks down near 4 V) was wrong for this graded
   profile: the model (item 3) gives 11.77 V at k_P = 1 and 15.86 V at k_P = 0.3 (k_B 0.63, S_p 3e4). k_P is
   therefore not expected to fix A23's low-voltage avalanche (R164) or V_A (R163), which depend on the field near
   the junction at low bias. The knob stays as committed.
   (c) The fit runs the Gummel sweep and the BVCBO ramp (1 V steps) of each evaluation in two processes; its
   start: k_B 0.63, k_P 1.0, S_p 1e5 cm/s.
   (d) Public functions added: `simws.validation.ibm5am.bvcbo(k, T=300, V_max=20, step=0.5)` -> BVCBO_V,
   `fit_b()`, `A23B_FIXED`, `BVCBO_MEASURED_V`; knob dicts carry k_P and S_p (cm/s); `knobs()` returns them when
   the calibration file has them; `ibm5am_hbt(..., k_P=1.0, S_p=None)`.

### A23, A24, A23b: the checker's report and the lead's answers (24 Nov 2026)

The checker wrote `tests/v1/test_validation_5am.py` from the contract alone (fast tier: R160, R168, R180 to R182
and unit checks of the pure functions; slow tier: R161 to R175, R183). Its first fast run: 32 passed, 2 failed.
- **Fixed (the code was wrong).** (1) Ingesting `luo2011.fig4_1` (one series over temperature) labelled every point
  with one role, so the 300 K calibration point went in as "validation" (or all 27 as "calibration"):
  `catalog.ingest` now ingests, with T_K, only that temperature's points under its role, and without T_K only the
  points of the default role. (2) `ibm5am_hbt(k_P != 1, S_p=None)` failed while writing its description. (3)
  `ingest` of a per-temperature figure without T_K, and `load` of `pellish2008.fig4` with T_K, now refuse with a
  ValueError (they raised KeyError / AttributeError). After the fixes: 34 passed.
- **The checker's readings, accepted.** "Within x %" = |model / measured - 1|; "a factor F" = the larger of the
  two ratios; "at every traced VBE" = the trace's own points in the window, the model interpolated in ln I. R163:
  V_A = intercept / slope of the straight line (the lead's `early_voltage_measured` uses IC(1.3 V) / slope, 1.3 V
  more; the verdict is the same). R180 is gated on the cathode current, where S acts, with the p side's electron
  term added to the closed form (at most 1 %). R169: the model IC scaled x 2L to one device before the comparison.
  `ionization_integral` returns M - 1. `grade_values`: half-open terms, the last closed (A23.7(b), stated now).
- **Noted, not changed.** In R180's diode at S = 1e2 cm/s the anode current is 1.6 % below the cathode's: the
  anode sits in 1e20 holes, and the tiny current (about 1e-7 A/cm^2) is below the solver's absolute resolution
  there; the cathode current matches the closed form within 0.3 %. Not tested: R167's Moll-Ross integral
  independently, the model's M - 1 independently (the field on the cut is not public), A24.3's per-temperature
  trust labels (no hub tolerance for VBE shifts), refusals of overlapping grading terms (no wording; the engine
  refuses them, A23.7(b)).

### A23b, build-machine results (24 Nov 2026)

**The fit (A23b.3).** `fit_b` from k_B 0.63, k_P 1.0, S_p 1e5 cm/s; 25 evaluations (7 optimizer steps, xtol),
about 3.2 min each with two workers. Result: **k_B 0.681, k_P 1.822, S_p 6.24e5 cm/s** (s 0.80 and k_E 1 fixed);
residuals (log10) IC 0.069 / 0.053 / 0.045 / 0.030 / 0.024 at 0.60 / 0.66 / 0.70 / 0.74 / 0.78 V, IB -0.014 /
0.000 / +0.013 at 0.70 / 0.74 / 0.78 V, RsBi -0.034; BVCBO 10.15 V. All knobs inside their ranges.

**The checker's slow tier on that calibration** (1 h 59 min on the build machine): 17 passed, 10 failed.
- R161 FAIL: IC / Luo 1.053 to 1.175 (worst 17.5 %, limit 15 %, at the low-VBE end); IB / measured 0.968 to 1.031
  (A23: 1.54 to 1.64), RsBi 0.924 (A23: 0.892): both within their limits now.
- R162 FAIL: IC / Pellish 1.152 to 1.257 (limit 25 %; A23 30.5 %). Reported: IB / Pellish 1.28 to 1.33.
- R163 FAIL: V_A 22.3 / 23.9 V against 138.8 / 267.4 V (A23: 28 / 30 V).
- R164 FAIL: M - 1 factors 21.9, 7.3, 3.8, 2.8 at VCB 2 to 5 V (A23: 9.4, 3.9, 2.3, 1.8). R173 FAIL, the same at
  162 K (20.3, 6.2, 3.8) and 93 K (16.0, 5.1, 3.1).
- R165 FAIL: peak fT 69.7 GHz at 1.24 mA/um^2 against 46.9 GHz at 0.94 (+48 %; A23 +27 %).
- R166 pass: IC -1.50 %, IB -0.70 % at mesh scale 0.5. R167 reported: IC / Moll-Ross 0.78 (0.60 V), 0.75 (0.65 V).
- R169 pass (the 300 K result stays unverified after both failed comparisons). R183 pass: BVCBO 10.15 V (-3.4 %).
- R170 pass at all four temperatures: -7.3 / -5.0 mV (393 K), -3.3 / -2.0 (223 K), +6.7 / +5.5 (162 K),
  +18.5 / +12.2 mV (93 K; A23 had no converged solution there).
- R171 pass: 0.905 (223 K), 0.785 (162 K), factor 1.72 at 93 K (new). R172 pass at 223 K (0.969) and 162 K (0.913);
  FAIL at 100 K (0.468) and 75 K (0.290, factor 3.4 > 2); reported 70 K 0.263, no solution from 60 K down.
- R174 pass: +2.1 mV (300 K), +13.5 mV (77 K, new). R175 reported: 93 K ideality 1.06 (Luo 1.23); 43 K no solution.

**What the numbers say.** A23b fixed what it was for: the base current (the emitter's hole sink) and the cold runs
down to 77 to 93 K. What fails now has one cause in two places. (1) The base-collector junction: to reach BVCBO
10.5 V the fit raised the collector phosphorus (k_P 1.8), which makes the junction field steeper at low VCB (M - 1
too large by a factor 3 to 22, falling with VCB), and the base too lightly doped and too narrow to hold the Early
voltage (V_A a factor 6 to 11 low; k_B 0.68 is the fit trading base charge for IC). The SIMS trace has no depth
scale (A23.2: 0.1 um per tick from the Ge film), so the base width and the collector profile's slope are the weakest
parts of the reconstruction; a depth-scaled 5AM / 5HP profile is what would pin them. (2) The fT peak is too high
for the same reason (a thin base and a short collector transit). Below 100 K the model's base sheet resistance rises
far too slowly (0.47 at 100 K): boron freeze-out in the 1e19 base with Altermatt's model at this doping is weaker
than Luo measures. These are recorded as xfail(strict=True) with these numbers; no tolerance changed.

**The full fast suite after A23b** (tests/v1, tests/connections): 3 failures in older tests, all from A23, fixed.
(1, 2) R97's and R107's id checks call every library function whose arguments all have defaults: `ibm5am_scale`
had a default L_um (2.5) and returns a number, not a problem. L_um is now required (every caller passes it).
(3) `test_unsupported_two_grade_terms` (written before A23) expected any second grading term to be refused; A23.7(b)
allows ordered, non-overlapping terms, so the test now checks the refusal A23.7(b) keeps (overlapping terms). The
lead should have run the whole fast suite before recording A23; it is run before every update from now on.

### A23, A24 and A23b, laptop run (update 14, 24 Nov 2026)

Update 14 committed on arch/v1 as ee80a7c and master fast-forwarded. (First attempt stopped on an untracked
`logs_install_v1/` in the worktree; it is now in the repo's local info/exclude.) Fast suite: 2108 passed, 34 skipped,
9 xfailed in 24 min 47 s (build machine: 11 min, 2.2x). Quick acceptance: 32 passed, 2 skipped (no PDFs on the
laptop, R160's hash check) in 1 min 2 s; R161 xfailed as recorded, with the build machine's numbers to every printed
digit (IC 1.053 to 1.175, IB 0.968 to 1.031, RsBi 13201 ohm/sq), as are R180 and R181. R161's single 2D Gummel sweep
took 14 min 3 s against 2 min 41 s here (5.2x, not 2x): the 5AM device's solves are slower on the laptop than the
fast suite's, so the slow tier (2 h here) should be planned at 6 to 10 hours there, not 4.

### A23b closed: the 5AM device's known limits (24 Nov 2026, owner's decision)

Mahmoud chose not to look for a depth-scaled 5AM / 5HP SIMS profile and to move on to A25 (option 2). The A23b
device and calibration (k_B 0.681, k_P 1.822, S_p 6.24e5 cm/s; s 0.80, k_E 1) are frozen for A25 to A29, and its
recorded failures stand as known limits, carried into the paper's table as FAILs.
- **Use it for** (validated against held-out or calibration data within the contract's tolerances): the 300 K
  base current and forward Gummel shape to 0.78 V (IB within 3 %, IC within 17.5 %), the intrinsic base sheet
  resistance from 393 K to 162 K, BVCBO, VBE(T) from 393 K to 77 K (within 18.5 mV), peak gain from 223 K to 93 K.
- **Do not lean on** (recorded failures): the base-collector junction and collector profile (Early voltage a factor
  6 to 11 low, M - 1 a factor 3 to 22 high, fT +48 %), the collector current's last 15 to 25 %, and the base sheet
  resistance below 100 K. Any later result that depends on these (inverse-mode base current, collector charge
  collection in A27, cold base resistance in A29) carries this in its stated uncertainty, and each contract says
  so before its runs.

### A25 (24 Nov 2026, lead): total dose on the 5AM, interface traps and oxide charge at the spacer and the STI

A25 replaces the provisional TID constants (simws.radiation.hbt_rad, v0) with a v1 feature, traps and charge on
named Si/SiO2 interfaces of a DeviceSpec, and fits three knobs to the 5AM's 63 MeV proton response at 300 K; then
predicts other doses, the JC-fixed extraction, another chip, other radiation sources, the shape of the excess base
current, and irradiation at 77 K. The device is the frozen A23b 5AM (k_B 0.681, k_P 1.822, S_p 6.24e5 cm/s; s 0.80,
k_E 1), unchanged.

0. **What the lead had seen.** A. K. Sutton, PhD thesis, Georgia Tech, Aug 2009 (the copy the owner supplied, its
   SHA-256 in sutton2009.json): the text of sections 3.4 to 4.2 and 6.2 (pages 60 to 85, 128 to 130); the page
   images, at low resolution, of **Figs. 24, 33, 34 and 73** (no values read off them beyond the following).
   Fig. 24 (63 MeV protons, 300 K, forward mode, 5AM and three later generations): the 5AM's dIB/IB0 at VBE = 0.6 V
   rises roughly from 0.3 at 0.3 Mrad(SiO2) to about 10 at 6 Mrad, near-linear in dose. Fig. 33 (5AM 0.5 x 2.5
   um^2, Co-60, 10 keV X-ray and 63 MeV protons): forward about 0.1 to 10 over 0.3 to 6 Mrad (the text: D^1.3, proton
   and gamma matched up to 3 Mrad, X-ray larger, attributed to dose enhancement by W / Cu studs near the spacer);
   inverse about 10 to 40 (D^0.3; gamma slightly above proton, both above X-ray). Fig. 73 (5AM 0.5 x 1.0, forward,
   at JC = 10 nA/um^2): 300 K about 0.3 to 20, 77 K about 0.05 to 1 over 0.1 to 6 Mrad (the text: "more than an
   order of magnitude reduction" when irradiated cold, attributed to holes self-trapped in the oxide and fewer
   interface traps). The text also gives: inverse-mode IB is set by the STI-to-SIC distance and the 5AM's STI has
   excess H2 (its inverse response is the largest of the four generations); inverse saturation is explained by
   oxide-trapped charge accumulating the base surface; ideality of dIB near 2 (Figs. 38 to 40, not viewed); dose
   rates 1 krad/s (protons), 30 rad/s (Co-60), 540 rad/s (X-ray); 1 rad(Si) = 1.8 rad(SiO2) for 10 keV X-rays;
   all terminals grounded during irradiation; anneal and stress results on 8HP (not used). Not viewed: Figs. 22,
   23, 25, 26, 39, and Fig. 72's post-irradiation curves. No A25 simulation had been run when items 0 to 6 were
   committed.
1. **Data** (vector traces of the thesis, as A23.1; `sutton2009.json` gains the keys below; catalog roles as A23
   set them: `sutton2004.figs21_26` calibration, `sutton2007.figs33_39` and `prakash2006.fig73` validation):
   (a) Fig. 22, the 5AM's (0.5 x 1.0) inverse-mode Gummel, pre-irradiation, 300 K (`fig22`, role validation, under
   sutton2004.figs21_26 as a per-curve role); (b) Figs. 24 and 26, the 5AM's forward and inverse dIB/IB0 at VBE =
   0.6 V and at JC = 1 uA/um^2 vs proton dose, with the error bars as min / max (`fig24`, `fig26`); Figs. 23 and 25
   (the same as dJB) are traced and used only to check the traces (dJB = dIB/IB0 x JB0 within the trace error);
   (c) Fig. 33, 5AM 0.5 x 2.5, forward and inverse, three sources (`fig33`); (d) Fig. 39, n of dIB vs VBE for the
   5AM, three sources (`fig39`); (e) Fig. 72's post-irradiation curves (1, 3, 6 Mrad; 300 K and 77 K; `fig72`
   gains them beside the pre-irradiation curves); (f) Fig. 73's 5AM series (`fig73`). The 8HP / 7HP / 9T series in
   the same panels are not traced. The calibration points are the 5AM's **VBE = 0.6 V** series of Figs. 24 and 26 at
   **1 and 6 Mrad(SiO2)** (four numbers); every other point of every A25 figure is validation.
2. **The interface feature (engine, spec).** `DeviceSpec.interfaces: List[Interface]`, each: `name`, `region`,
   `side` (y_min, y_max, x_min, x_max), `span` (um, along the side), `N_it_cm2` (>= 0; interface traps at midgap,
   sigma_n = sigma_p = `sigma_cm2`, default 1e-15 cm^2, the v0 constant, not a knob), `N_ot_cm2` (>= 0; a fixed
   positive sheet charge q N_ot on that boundary). Absent or empty: the problem id is unchanged (the serializer drops
   it, as A23b.1). DEVSIM: the surface recombination U_s = s (n p - n_ie^2) / (n + p + 2 n_ie), s = sigma v_th
   N_it, v_th = 1e7 cm/s x sqrt(T / 300 K), enters both continuity equations on the boundary nodes of the span as
   U_s x (boundary length / NodeVolume) (the v0 surface factor, exact as a surface integral on the mesh); N_ot enters
   Poisson as q N_ot x (boundary length / NodeVolume). Charon refuses interfaces ("interfaces: interface traps and
   oxide charge (A25) are not translated to Charon"). The engine reports, per interface, the recombination current
   q integral U_s dl at every bias point (`interface_current_A`, per cm of depth like the contact currents).
3. **The 5AM's two interfaces** (`ibm5am_hbt(..., spacer=(N_it, N_ot), sti=(N_it, N_ot))`, both default absent):
   the **EB spacer**, the top surface from the emitter edge to the extrinsic base, x in [0.21, 0.36] um (y_min);
   the **STI sidewall**, the cell's outer edge x = 1.2 um (x_max), from the surface down to 0.2 um below the
   extrinsic base (y in [0, y_xb + 0.2]), where the base-collector junction meets it. The STI position is a
   modelling choice (the STI-to-SIC distance is not published); its magnitude is absorbed by k_I below, so a chip with
   a different STI-to-SIC distance is not predicted by it. The current to one device: per-cm values x 2L for the
   bulk (as A23.2) and x the emitter perimeter P_E = 2 (L + W_E) for each interface's excess (W_E = 0.5 um drawn;
   the spacer runs round the emitter; the STI is taken to follow it, stated as an approximation). Inverse mode: the
   collector is the emitter (base swept, the physical emitter tied to the base, VCB_inverse = 0; the physical
   collector grounded); JC in inverse mode is the physical emitter's current over the drawn AE.
4. **Dose -> traps (knobs, A25.3).** N_it,spacer = k_F x D, N_it,STI = k_I x D, N_ot (both interfaces) = k_Q x D,
   with D in Mrad(SiO2), linear: interface traps build up linearly to 6 Mrad (Sutton: forward near-linear), and the
   inverse-mode sub-linearity is left to the oxide charge, as Sutton explains it (accumulation of the base surface).
   Ranges: k_F and k_I 1e9 to 1e13 cm^-2 per Mrad, k_Q 0 to 1e13 cm^-2 per Mrad (fitted in log10, k_Q in log10 of
   1e8 + k_Q). This replaces the plan's third knob (a saturation dose) because a saturation dose cannot be told from
   the oxide charge with four calibration numbers, and Sutton names the charge. Fit: least squares in log10 of
   dIB/IB0 at VBE = 0.6 V, forward and inverse, 1 and 6 Mrad, 300 K (Figs. 24, 26, the 5AM's mean points). The
   fitted knobs and residuals are stored in `simws/data/validation/ibm5am_tid.json`. Source differences (charge
   yield by LET, X-ray dose enhancement), dose rate, displacement damage and temperature-dependent build-up are not
   modelled: the knobs are per Mrad(SiO2) of 63 MeV protons, and other sources and 77 K use them unchanged.
   Measurement temperature equals irradiation temperature (Sutton measured the cold parts cold).
5. **Acceptance** (tolerances fixed here; "a factor F" = max of the two ratios):
   R190: data and catalog (each new trace loads with units, source, method; roles as item 1; ingest labels them so;
   Figs. 23 / 25 consistent with 24 / 26 within the combined trace error).
   R191: the feature against closed forms. (a) No interfaces: currents bit-identical to the A23b device, id
   unchanged. (b) A 1D n+/p diode's p side with a surface on its free boundary, low injection: the excess current
   equals the reported q integral U_s dl within 1 % (U_s from the solved n, p on the boundary: what the engine
   reports is what it solves), and the excess current is linear in N_it within 2 % from 1e9 to 1e11 cm^-2; at
   zero bias the excess current is zero within the solver's absolute resolution. (c) N_ot: the change of the total charge in
   the silicon equals -q N_ot x the span within 1 % (Gauss), at equilibrium.
   R192: refusals (negative N_it / N_ot or sigma, a span off its side, an interface on a contact's nodes, Charon).
   R193: the calibration fit: the four calibration points within a factor 1.5 (reported per point).
   R194: other doses, VBE = 0.6 V, Figs. 24 / 26 (5AM, 0.1 to 3 Mrad): within a factor 2 at every traced dose;
   the log-log slope between 0.3 and 3 Mrad reported for both modes against Sutton's D^1 / D^0.3.
   R195: the JC = 1 uA/um^2 series of Figs. 24 / 26 (5AM, every dose): within a factor 2.
   R196: Fig. 33 (5AM 0.5 x 2.5, L = 2.5): protons within a factor 2 at every dose, both modes (another chip, the
   perimeter scaling of item 3); Co-60 within a factor 2 (reported separately: the model has no source dependence);
   10 keV X-ray within a factor 2 with the dose converted as Sutton plots it (rad(SiO2)). Expected trouble: the
   forward X-ray response (dose enhancement, not modelled).
   R197: the ideality of dIB, n = (q / kT) / (d ln dIB / dVBE), at VBE 0.5, 0.6 and 0.7 V against Fig. 39 (5AM, 3
   Mrad, each source, each mode): within 0.3.
   R198: Fig. 72's post-irradiation 300 K forward IB at 1, 3 and 6 Mrad, at every traced VBE from 0.45 to 0.75 V:
   within a factor 2 (the 0.5 x 1.0 chip; the model's pre-irradiation IB there is reported first, as R174 did).
   R199: 77 K (Figs. 72 and 73): dIB/IB0 at JC = 10 nA/um^2 with the 300 K knobs, at every traced dose: within a
   factor 2; 300 K points of Fig. 73 within a factor 2 too. Expected FAIL at 77 K (Sutton: more than 10x lower
   measured; the model has no temperature-dependent build-up). A FAIL is recorded; temperature-dependent build-up
   would be a separate amendment (A25b), decided after.
   R189: the model's inverse-mode Gummel (pre-irradiation) against Fig. 22 at VBE 0.5 to 0.8 V: IC and IB reported
   per point; IB within a factor 3 gated. The A23b device's collector profile is a recorded limit (A23b closed), so
   this is expected to be the weakest check, and every inverse-mode result above inherits its error through IB0.
6. **Trust.** The calibrated 300 K results become validated only where R194 to R198 pass for that mode; 77 K
   results stay verified at most (R199). The hub records the knobs' file hash with each result, as A23.5.

**A25, clarifications (24 Nov 2026, before any 5AM total-dose run).** (a) R191(b) as written was wrong physics:
traps on a surface also lower the minority density near it, so the bulk recombination and the minority current
into the contacts change too, and the excess terminal current is not q integral U_s dl (on the lead's first trial
diode, a 2D n+/p diode with the interface on its free side, the total anode current rose by 0.66 x q integral U_s;
the hole current at the anode rose by exactly q integral U_s). R191(b) is therefore: the hole-continuity balance of
the whole device, I_p(anode) + I_p(cathode) = q integral U_bulk dV + q integral U_s dl, within 1 % with and without
interfaces (what the engine reports is what it solves, and the interface term is in the equations), and q integral
U_s dl linear in N_it within 2 % from 1e10 to 1e11 cm^-2; at zero bias the interface current is zero within the
solver's resolution. The diode is 2D (interfaces are 2D only, item 2): n+ 1e19 to 0.5 um over p 1e17, 1 x 3 um,
contacts on y_min and y_max, the interface on x_max over y 0.6 to 2.9 um. That trial run, before this
clarification: the balance held to 7 digits; q integral U_s 5.049e-10 and 5.030e-9 A/cm at N_it 1e10 and 1e11
(0.4 % from linear); Gauss for N_ot = 1e11 over 2.3 um: -2.2991e7 against -2.3e7 per cm (0.04 %).
(b) An interface's share of a node on a contact (the contact's end node) is dropped, as the contact's equations
hold there: on the 5AM spacer this is half a mesh spacing (2 nm of 150 nm) at the emitter edge.
(c) R197's VBE: Fig. 39 spans 0.55 to 0.74 V (seen after item 0 was written, with Figs. 22 and 72); the points
are 0.6 and 0.7 V, and the lowest traced, 0.55 V, in place of 0.5 V.
(d) Fig. 22's two 5AM curves are not labelled: the tracer assigns IB by the factor between Figs. 25 / 26 and the
curve at 0.6 V, the same factor (2.5) that Figs. 23 / 24 show against Fig. 72's forward IB (`fig22.assignment`):
the lower curve is IB. The ratio figures (23 to 26) use a current density 2.5 x Fig. 72's for the same 0.5 x 1.0
device (perhaps another normalization); A25 compares only their ratios dIB/IB0, which do not depend on it.
(e) R190: the JC = 1 uA/um^2 series of Figs. 23 / 24 agree within 0.6 % except the 0.3 Mrad point (14 %), and of
25 / 26 the same within 16 %; the VBE = 0.6 V series agree within 0.6 %. The gate stays on both series as
written (no narrowing after the trace): the JC series fail it, a property of the published points (the trace
error is about 2 %), recorded as a FAIL of R190's JC part.
(f) The engine's per-interface recombination current is reported as `interface_current_A_per_cm` (A per cm of
depth, as the contact currents of a 2D device; item 2 wrote `interface_current_A`): the name says the unit.

**A25, what the lead had seen before the fit (24 Nov 2026).** A smoke run of the A23b 5AM (L = 1 um, half cell,
no scaling to the perimeter) at VBE 0.5 to 0.75 V: forward, spacer N_it 1e11 raises IB at 0.6 V by 16 %, 1e12 by
162 % (linear in N_it), IC unchanged; spacer N_it 1e12 with N_ot 1e12 raises IB by 1480 % and IC by 13 %. Inverse,
STI N_it 1e11 raises IB by 7.6 %, 1e12 by 76 %; IB0 (inverse) at 0.6 V is 7.3e-10 A for the 0.5 x 1.0 device (Fig.
22's IB there corresponds to 1.6e-9 A). The checker's fast tier: 37 passed, 2 strict xfails (R190's JC series),
1 skipped (the knobs file); it found two code issues (vbe_at_jc extrapolated past the sweep; a docstring name),
fixed. The fit starts from k_F 5e11, k_I 3e12, k_Q 1e11 cm^-2 per Mrad, chosen from the smoke run.

### A25, build-machine results (24 Nov 2026)

**The fit (A25.4).** From k_F 5e11, k_I 3e12, k_Q 1e11: 24 evaluations (4 sweeps each on two workers), xtol.
Result **k_F 4.77e11, k_I 9.99e12 (the range's top), k_Q 5.45e10 cm^-2 per Mrad(SiO2)**. Calibration points
(dIB/IB0 at 0.6 V, 0.5 x 1.0 chip): forward 1.34 / 11.1 against 1.31 / 11.25 (1.05 / 6.3 Mrad); inverse 11.3 / 52.3
against 24.3 / 49.6. The inverse response of the model grows near-linearly with dose (STI traps), the measured one
as D^0.4 to D^0.6: the fit splits the difference by putting k_I at its upper limit.

**The first slow run** (the checker's suite as written: 15 to 17 doses on 23 VBE points, serially) did not finish in
4 h (one mode); the lead changed the slow tier's grids (the dose set 0.18 to 6.3 Mrad with interpolation between
neighbouring doses, a 0.04 V VBE grid plus +-10 mV around R197's points, sweeps in parallel; `tests/v1/test_tid_5am.py`
says so where it does it); no gate or tolerance changed. It found a tracing error: Fig. 72's post-irradiation trace
held the legend's line samples (1e-5 to 1e-4 A/um^2 at 0.46 to 0.50 V), fixed in the tracer before R198 was
recorded. Slow tier, 50 min on two workers (R198 re-run after the fix): **7 passed, 16 failed**.
- R189 pass: inverse JB within a factor 2.33 of Fig. 22 at 30 traced VBE (limit 3); IC within 1.24 (reported).
- R193: forward pass (1.02, 1.01); inverse FAIL (1.05 Mrad: factor 2.14; 6.3 Mrad: 1.06).
- R194: forward pass (factors 1.28, 1.15, 1.57 at 0.31, 0.63, 3.1 Mrad; slope 1.03, measured 1.34); inverse FAIL
  (2.81, 2.54, 1.41; slope 0.94, measured 0.64).
- R195 FAIL both modes: at JC = 1 uA/um^2 the model's forward ratio is 0.09 to 4.6 against 0.12 to 6.6 (factors 1.3
  to 3.1), inverse 1.1 to 13.6 against 6.0 to 30.3 (2.2 to 5.6).
- R196 (0.5 x 2.5 chip): pass for forward Co-60 (worst 1.93) and inverse X-ray (1.70); FAIL for forward protons
  (3.95 at 0.31 Mrad, where Fig. 33's proton point, 0.077, is 4 x Fig. 24's on the other chip; 3.22 at 3.15 Mrad),
  forward X-ray (up to 3.19: the dose enhancement, expected), inverse protons (3.27 to 1.09) and inverse Co-60
  (5.28 to 1.91): below 1 Mrad the inverse model is a factor 2.5 to 5 low.
- R197: inverse Co-60 and X-ray pass (worst 0.29, 0.28); FAIL forward (all sources: the model's n falls from 1.71 at
  0.55 V to 1.33 at 0.7 V, measured 1.7 to 1.9 throughout) and inverse protons (0.31 at 0.7 V).
- R198 FAIL: the model's pre-irradiation JB is within 0.91 to 1.00 of Fig. 72; after 1, 3, 6 Mrad the worst factors
  are 2.16, 3.01, 3.03 (the model low at the top of the window, where its excess current flattens: the n above).
- R199 FAIL at 77 K, as expected (Fig. 73: factors 6.4, 2.1, 2.5, 1.8, 13 at 0.1 to 6.3 Mrad; from Fig. 72: 11 to
  18); at 300 K Fig. 73 passes at four doses (1.11, 1.02, 1.22, 1.47) and fails at 3.15 Mrad by 0.5 % (2.010).
- R190's JC part (clarification e): FAIL, as recorded.

**What the numbers say.** Forward mode at fixed VBE is the model's good part: one knob (the spacer traps) with the
oxide charge carries the 0.5 x 1.0 chip from 0.3 to 6 Mrad within a factor 1.6, Co-60 on the other chip within
1.9, and Fig. 73's 300 K points within 1.5 but one. Where it fails, it fails in two ways. (1) The shape of the
excess current: the model's surface-recombination current bends over above 0.6 V (n 1.6 to 1.3) where Sutton's
stays near n = 1.8 to 1.9, so every comparison at a higher VBE or at a fixed JC = 1 uA/um^2 (VBE 0.70 V) comes out
low (R195, R197, R198). A midgap single-level trap at the spacer cannot hold n near 2 to 0.7 V in this device; a
trap distribution across the gap or a wider surface depletion along the spacer would. (2) Inverse mode: the STI
interface on the cell's edge (A25.3's modelling choice) gives a response linear in dose, while the measured one
saturates (D^0.3 to D^0.6); with the oxide charge shared with the spacer, k_Q cannot do both, and the fit leaves
inverse 1 Mrad a factor 2.1 low. Sutton's own explanation (accumulation of the base surface by the STI's trapped
charge) needs the STI's own oxide-charge rate, not a shared one. 77 K fails as the contract expected (no
temperature-dependent build-up). These are recorded as xfail(strict=True) with these numbers; no tolerance changed.
