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
