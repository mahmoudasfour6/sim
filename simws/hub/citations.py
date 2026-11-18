"""Citation registry and bibliography (interface specification: citations).

Adapters list citation keys in their manifest; every result envelope carries them, so a study's bibliography is
the union over the results it used:

    from simws.hub.citations import bibliography
    print(bibliography(results, fmt="bibtex"))

Entries record what was checked: 'checked': True only where the bibliographic data were confirmed against the
publisher record when the entry was added; unchecked entries are printed with a note so they are verified before
submission. DOIs are left out where they were not confirmed.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

REGISTRY: Dict[str, dict] = {
    "ngspice47": dict(type="manual", author="H. Vogt and M. Hendrix and P. Nenzi and D. Warning",
                      title="Ngspice User's Manual, Version 47", year="2026",
                      url="https://ngspice.sourceforge.io/docs.html", checked=False),
    "openvaf": dict(type="misc", author="P. Kuthe and A. Buermen and others",
                    title="OpenVAF-Reloaded: a Verilog-A compiler for compact models (OSDI 0.3 build osdi_0.3-31-gf47d557)",
                    year="2024", url="https://fides.fe.uni-lj.si/openvaf/", checked=False),
    "lucsanyi2022": dict(type="article",
                         author="D. Lucsanyi and R. Garcia Alia and K. Bilko and M. Cecchetto and S. Fiore and E. Pirovano",
                         title="G4SEE: A Geant4-Based Single Event Effect Simulation Toolkit and Its Validation Through "
                               "Monoenergetic Neutron Measurements",
                         journal="IEEE Transactions on Nuclear Science", volume="69", number="3", pages="273--281",
                         year="2022", doi="10.1109/TNS.2022.3149989", checked=True),
    # DOI as printed in the RustBCA README; volume, number and year from the JOSS record were not re-checked
    "drobny2021": dict(type="article", author="J. T. Drobny and D. Curreli",
                       title="RustBCA: A High-Performance Binary-Collision-Approximation Code for Ion-Material Interactions",
                       journal="Journal of Open Source Software", volume="6", number="64", pages="3298", year="2021",
                       doi="10.21105/joss.03298", checked=False),
    "geant4_2016": dict(type="article", author="J. Allison and others", title="Recent developments in Geant4",
                        journal="Nuclear Instruments and Methods in Physics Research A", volume="835",
                        pages="186--225", year="2016", doi="10.1016/j.nima.2016.06.125", checked=False),
    "mendenhall2005": dict(type="article", author="M. H. Mendenhall and R. A. Weller",
                           title="An algorithm for computing screened Coulomb scattering in Geant4",
                           journal="Nuclear Instruments and Methods in Physics Research B", volume="227",
                           number="3", pages="420--430", year="2005", doi="10.1016/j.nimb.2004.08.014",
                           checked=False),
    "montanari2024": dict(type="article", author="C. C. Montanari and others",
                          title="The IAEA electronic stopping power database: modernization, review, and analysis "
                                "of the existing experimental data",
                          journal="Nuclear Instruments and Methods in Physics Research B", volume="551",
                          pages="165336", year="2024", doi="10.1016/j.nimb.2024.165336", checked=False),
    "holmstrom2008": dict(type="article", author="E. Holmstrom and A. Kuronen and K. Nordlund",
                          title="Threshold defect production in silicon determined by density functional theory "
                                "molecular dynamics simulations",
                          journal="Physical Review B", volume="78", pages="045202", year="2008",
                          doi="10.1103/PhysRevB.78.045202", checked=True),
    "devsim": dict(type="article", author="J. E. Sanchez", title="DEVSIM: A TCAD Semiconductor Device Simulator",
                   journal="Journal of Open Source Software", volume="7", number="70", pages="3898", year="2022",
                   doi="10.21105/joss.03898", checked=False),
    "giannozzi2009": dict(type="article", author="P. Giannozzi and others",
                          title="QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials",
                          journal="J. Phys.: Condens. Matter", volume="21", pages="395502", year="2009",
                          doi="10.1088/0953-8984/21/39/395502", checked=False),
    "giannozzi2017": dict(type="article", author="P. Giannozzi and others",
                          title="Advanced capabilities for materials modelling with Quantum ESPRESSO",
                          journal="J. Phys.: Condens. Matter", volume="29", pages="465901", year="2017",
                          doi="10.1088/1361-648X/aa8f79", checked=False),
    "hamann2013": dict(type="article", author="D. R. Hamann", title="Optimized norm-conserving Vanderbilt pseudopotentials",
                       journal="Phys. Rev. B", volume="88", pages="085117", year="2013", doi="10.1103/PhysRevB.88.085117",
                       checked=False),
    "vansetten2018": dict(type="article", author="M. J. van Setten and others",
                          title="The PseudoDojo: Training and grading a 85 element optimized norm-conserving pseudopotential table",
                          journal="Comput. Phys. Commun.", volume="226", pages="39", year="2018",
                          doi="10.1016/j.cpc.2018.01.012", checked=False),
    "varshni1967": dict(type="article", author="Y. P. Varshni", title="Temperature dependence of the energy gap in semiconductors",
                        journal="Physica", volume="34", pages="149", year="1967", checked=False),
    "thurmond1975": dict(type="article", author="C. D. Thurmond", title="The standard thermodynamic functions for the formation of electrons and holes in Ge, Si, GaAs, and GaP",
                         journal="J. Electrochem. Soc.", volume="122", pages="1133", year="1975", checked=False),
    "altermatt2003": dict(type="article", author="P. P. Altermatt and others",
                          title="Reassessment of the intrinsic carrier density in crystalline silicon in view of band-gap narrowing",
                          journal="J. Appl. Phys.", volume="93", pages="1598", year="2003", checked=False),
    "altermatt2006a": dict(type="article", author="P. P. Altermatt and A. Schenk and G. Heiser",
                           title="A simulation model for the density of states and for incomplete ionization in crystalline silicon. I. Establishing the model in Si:P",
                           journal="J. Appl. Phys.", volume="100", pages="113714", year="2006", checked=False),
    "altermatt2006b": dict(type="article", author="P. P. Altermatt and A. Schenk and B. Schmithuesen and G. Heiser",
                           title="A simulation model for the density of states and for incomplete ionization in crystalline silicon. II. Investigation of Si:As and Si:B and usage in device simulation",
                           journal="J. Appl. Phys.", volume="100", pages="113715", year="2006", checked=False),
    "klaassen1992a": dict(type="article", author="D. B. M. Klaassen",
                          title="A unified mobility model for device simulation - I. Model equations and concentration dependence",
                          journal="Solid-State Electron.", volume="35", pages="953", year="1992", checked=False),
    "klaassen1992b": dict(type="article", author="D. B. M. Klaassen",
                          title="A unified mobility model for device simulation - II. Temperature dependence of carrier mobility and lifetime",
                          journal="Solid-State Electron.", volume="35", pages="961", year="1992", checked=False),
    "nilsson1973": dict(type="article", author="N. G. Nilsson",
                        title="An accurate approximation of the generalized Einstein relation for degenerate semiconductors",
                        journal="Phys. Status Solidi A", volume="19", pages="K75", year="1973", checked=False),
    "green1990": dict(type="article", author="M. A. Green",
                      title="Intrinsic concentration, effective densities of states, and effective mass in silicon",
                      journal="J. Appl. Phys.", volume="67", pages="2944", year="1990", checked=False),
    "masetti1983": dict(type="article", author="G. Masetti and M. Severi and S. Solmi",
                        title="Modeling of carrier mobility against carrier concentration in arsenic-, phosphorus-, and boron-doped silicon",
                        journal="IEEE Trans. Electron Devices", volume="30", pages="764", year="1983", checked=False),
    "rieger1993": dict(type="article", author="M. M. Rieger and P. Vogl",
                       title="Electronic-band parameters in strained Si1-xGex alloys on Si1-yGey substrates",
                       journal="Phys. Rev. B", volume="48", pages="14276", year="1993", checked=False),
    "mclean1987": dict(type="techreport", author="F. B. McLean and T. R. Oldham",
                       title="Basic mechanisms of radiation effects in electronic materials and devices",
                       institution="Harry Diamond Laboratories", number="HDL-TR-2129", year="1987", checked=False),
    "dozier1987": dict(type="article", author="C. M. Dozier and D. M. Fleetwood and D. B. Brown and P. W. Winokur",
                       title="An evaluation of low-energy X-ray and cobalt-60 irradiations of MOS transistors",
                       journal="IEEE Trans. Nucl. Sci.", volume="34", pages="1535", year="1987", checked=False),
    "esqueda2020": dict(type="misc", author="I. S. Esqueda and others",
                        title="Reduced interface-trap buildup model for total ionizing dose (as cited in sio2.yaml)",
                        note="arXiv:2009.12652", year="2020", checked=False),
    "petersen1992": dict(type="article", author="E. L. Petersen and others",
                         title="Rate prediction for single event effects - a critique",
                         journal="IEEE Trans. Nucl. Sci.", volume="39", pages="1577", year="1992", checked=False),
    "roache2002": dict(type="article", author="P. J. Roache",
                       title="Code verification by the method of manufactured solutions",
                       journal="J. Fluids Eng.", volume="124", pages="4", year="2002", checked=False),
    "roache1998": dict(type="book", author="P. J. Roache",
                       title="Verification and Validation in Computational Science and Engineering",
                       publisher="Hermosa", year="1998", checked=False),
    "eca2014": dict(type="article", author="L. Eca and M. Hoekstra",
                    title="A procedure for the estimation of the numerical uncertainty of CFD calculations based on grid refinement studies",
                    journal="J. Comput. Phys.", volume="262", pages="104", year="2014", checked=False),
    "robertson2004": dict(type="article", author="J. Robertson", title="High dielectric constant oxides",
                          journal="Eur. Phys. J. Appl. Phys.", volume="28", pages="265", year="2004", checked=False),
}


def keys_of(results: Iterable) -> List[str]:
    out: List[str] = []
    for r in results:
        for k in getattr(r, "citations", []) or []:
            if k not in out:
                out.append(k)
    return out


def unknown(keys: Iterable[str]) -> List[str]:
    return [k for k in keys if k not in REGISTRY]


def bibtex(key: str) -> str:
    e = dict(REGISTRY[key])
    t = e.pop("type")
    checked = e.pop("checked", False)
    fields = ",\n".join(f"  {k} = {{{v}}}" for k, v in e.items() if v)
    note = "" if checked else "% UNCHECKED: confirm against the publisher record before submission\n"
    return f"{note}@{t}{{{key},\n{fields}\n}}"


def text(key: str) -> str:
    e = REGISTRY[key]
    where = e.get("journal") or e.get("institution") or e.get("publisher") or e.get("note", "")
    vol = f" {e['volume']}" if e.get("volume") else ""
    pages = f", {e['pages']}" if e.get("pages") else ""
    doi = f", doi:{e['doi']}" if e.get("doi") else ""
    return f"{e['author']}, {e['title']}, {where}{vol}{pages} ({e['year']}){doi}" + ("" if e.get("checked") else " [unchecked]")


def bibliography(results_or_keys, fmt: str = "text") -> str:
    items = list(results_or_keys)
    keys = items if all(isinstance(x, str) for x in items) else keys_of(items)
    missing = unknown(keys)
    if missing:
        raise KeyError(f"citation keys not in the registry: {missing}")
    f = bibtex if fmt == "bibtex" else text
    return "\n\n".join(f(k) for k in keys) if fmt == "bibtex" else "\n".join(f"[{i + 1}] {f(k)}" for i, k in enumerate(keys))
