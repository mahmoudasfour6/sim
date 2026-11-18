// g4implant: ion ranges and displacement counts in a layered amorphous target with Geant4 (simws contracts A13,
// A14).
//
//     g4implant input.txt output.txt
//
// Nuclear collisions: TestEm7's G4ScreenedNuclearRecoil (M. H. Mendenhall and R. A. Weller, NIM B 227 (2005) 420),
// the only process that deflects the ion (no multiple scattering process, as TestEm7 below 100 MeV).
// Electronic stopping, one of:
//   ls         Lindhard-Scharff, a continuous loss n S_LS(E) with S_LS the formula RustBCA 3.0.0 uses
//              (src/material.rs): 1.212 A^2 eV Z1^(7/6) Z2 / (Z1^(2/3) + Z2^(2/3))^(3/2) sqrt(E[eV] / M1[u]),
//              summed over the elements of the material (Bragg's rule) and integrated exactly over each step;
//   empirical  G4ionIonisation with G4IonParametrisedLossModel (Geant4's ion stopping data and its scaling).
// No delta electrons (production cut 1 km).
//
// Input (one keyword per line, '#' comments):
//   seed <int>   n <int>   batches <int>
//   ion <Z> <A>   energy_keV <E>   tilt_deg <t>
//   screening zbl|mol   stopping ls|empirical
//   recoil_cutoff_eV_per_u <c>   physics_cutoff_eV <p>   recoils 0|1
//   bins <width_um> <count>                         (displacement histogram, used when recoils is 1)
//   layer <name> <thickness_um> <k>   then k lines: <symbol> <Z> <mass_amu> <atoms_per_cm3>
//   dedx_keV <E1> <E2> ...                          (electronic dE/dx of the ion in the first layer)
// G4ScreenedNuclearRecoil stops a moving atom below recoil_cutoff x (its mass in u), creates a recoil only above
// recoil_cutoff x (the target atom's mass in u), and ignores collisions that transfer less than physics_cutoff.
// Output: '# ' header lines, then one line per ion in launch order: <status> <x_um> <z_um>, status 0 stopped in
// the target, 1 left through the front, 2 through the back. With recoils 1, then one line per batch:
// 'D <batch> <ions> <recoils outside the target> <count bin 0> <count bin 1> ...' with the depth (x) at which each
// recoil was created (a displaced target atom: a vacancy).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "G4BaryonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4Element.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4GenericIon.hh"
#include "G4IonConstructor.hh"
#include "G4IonParametrisedLossModel.hh"
#include "G4IonTable.hh"
#include "G4LeptonConstructor.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MesonConstructor.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4ScreenedNuclearRecoil.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4Step.hh"
#include "G4VContinuousProcess.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "G4ionIonisation.hh"
#include "Randomize.hh"

struct Species { std::string sym; int Z; double mass; double n_cm3; };
struct LayerIn { std::string name; double t_um; std::vector<Species> sp; };
struct Input {
  long seed = 1; long n = 1000; int batches = 1; int Z = 5; int A = 11; double E_keV = 50; double tilt = 0;
  std::string screening = "zbl", stopping = "ls"; double recoil_cutoff = 0.1, physics_cutoff = 1.0;
  int recoils = 0; double bin_um = 0.005; int nbins = 0;
  std::vector<LayerIn> layers; std::vector<double> dedx_keV;
};

static void fail(const std::string& m) { std::cerr << "g4implant: " << m << "\n"; std::exit(2); }

static Input read_input(const char* path) {
  Input in; std::ifstream f(path); std::string line;
  if (!f) fail(std::string("cannot read ") + path);
  while (std::getline(f, line)) {
    auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
    std::istringstream s(line); std::string k; if (!(s >> k)) continue;
    if (k == "seed") s >> in.seed; else if (k == "n") s >> in.n; else if (k == "batches") s >> in.batches;
    else if (k == "ion") s >> in.Z >> in.A; else if (k == "energy_keV") s >> in.E_keV;
    else if (k == "tilt_deg") s >> in.tilt; else if (k == "screening") s >> in.screening;
    else if (k == "stopping") s >> in.stopping;
    else if (k == "recoil_cutoff_eV_per_u") s >> in.recoil_cutoff;
    else if (k == "physics_cutoff_eV") s >> in.physics_cutoff; else if (k == "recoils") s >> in.recoils;
    else if (k == "bins") s >> in.bin_um >> in.nbins;
    else if (k == "dedx_keV") { double e; while (s >> e) in.dedx_keV.push_back(e); }
    else if (k == "layer") {
      LayerIn L; int m = 0; s >> L.name >> L.t_um >> m;
      for (int i = 0; i < m; ++i) {
        if (!std::getline(f, line)) fail("layer " + L.name + ": missing species line");
        std::istringstream q(line); Species sp; q >> sp.sym >> sp.Z >> sp.mass >> sp.n_cm3; L.sp.push_back(sp);
      }
      in.layers.push_back(L);
    } else fail("unknown keyword '" + k + "'");
  }
  if (in.layers.empty()) fail("no layer");
  if (in.stopping != "ls" && in.stopping != "empirical") fail("stopping is ls or empirical");
  if (in.batches < 1 || in.batches > in.n) fail("need 1 <= batches <= n");
  if (in.recoils && in.nbins < 1) fail("recoils 1 needs 'bins <width_um> <count>'");
  return in;
}

static double g_total_um = 0;
static std::vector<G4Material*> g_mats;

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Input& in) : in_(in) {}
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    std::map<std::string, G4Element*> els;
    double T = 0; for (auto& L : in_.layers) T += L.t_um;
    g_total_um = T;
    const double margin = 1e-3 * um, wide = 1.0 * m;      // laterally 2 m: an infinite slab for these ranges
    auto* vac = nist->FindOrBuildMaterial("G4_Galactic");
    auto* wlog = new G4LogicalVolume(new G4Box("world", 0.5 * T * um + margin, wide, wide), vac, "world");
    auto* wphys = new G4PVPlacement(nullptr, {}, wlog, "world", nullptr, false, 0);
    double x0 = -0.5 * T * um;
    for (auto& L : in_.layers) {
      double rho = 0;                                       // g/cm3
      for (auto& sp : L.sp) rho += sp.n_cm3 * sp.mass * 1.66053906660e-24;
      auto* mat = new G4Material("simws_" + L.name, rho * g / cm3, (G4int)L.sp.size());
      for (auto& sp : L.sp) {
        auto key = sp.sym + "_" + std::to_string(sp.mass);
        if (!els.count(key)) els[key] = new G4Element("simws_" + sp.sym, sp.sym, sp.Z, sp.mass * g / mole);
        mat->AddElement(els[key], sp.n_cm3 * sp.mass * 1.66053906660e-24 / rho);
      }
      g_mats.push_back(mat);
      auto* lv = new G4LogicalVolume(new G4Box(L.name, 0.5 * L.t_um * um, wide, wide), mat, L.name);
      new G4PVPlacement(nullptr, G4ThreeVector(x0 + 0.5 * L.t_um * um, 0, 0), lv, L.name, wlog, false, 0);
      x0 += L.t_um * um;
    }
    return wphys;
  }
 private:
  const Input& in_;
};

// Lindhard-Scharff electronic stopping as a continuous process: dE/dx = -a sqrt(E), integrated exactly over a
// step (sqrt(E1) = sqrt(E0) - a L / 2), the step limited so that one step loses at most 5 % of the energy.
class LindhardScharff : public G4VContinuousProcess {
 public:
  LindhardScharff() : G4VContinuousProcess("lindhardScharff", fElectromagnetic) {}
  G4bool IsApplicable(const G4ParticleDefinition& p) override { return p.GetParticleType() == "nucleus"; }
  static double coefficient(double z1, double m1, const G4Material* mat) {   // a in MeV^(1/2) / mm
    const auto* els = mat->GetElementVector();
    const double* nat = mat->GetVecNbOfAtomsPerVolume();   // 1/mm^3
    double k = 0;
    for (size_t i = 0; i < els->size(); ++i) {
      const double z2 = (*els)[i]->GetZ();
      // eV m^2 per sqrt(eV/u); 1 eV m^2 = 1 MeV mm^2
      const double c = 1.212e-20 * std::pow(z1, 7.0 / 6.0) * z2 /
                       std::pow(std::pow(z1, 2.0 / 3.0) + std::pow(z2, 2.0 / 3.0), 1.5);
      k += nat[i] * c;                                      // MeV / mm per sqrt(eV/u)
    }
    // dE/dx [MeV/mm] = k sqrt(E[eV] / m1) = k sqrt(1e6 / m1) sqrt(E[MeV])
    return k * std::sqrt(1e6 / m1);
  }
  static double coefficient(const G4Track& tr) {
    return coefficient(tr.GetDefinition()->GetAtomicNumber(), tr.GetDefinition()->GetPDGMass() / amu_c2,
                       tr.GetMaterial());
  }
  G4double GetContinuousStepLimit(const G4Track& tr, G4double, G4double, G4double&) override {
    const double a = coefficient(tr);
    if (a <= 0) return DBL_MAX;
    const double e = tr.GetKineticEnergy() / MeV;
    return 2.0 * std::sqrt(e) * (1.0 - std::sqrt(0.95)) / a * mm;
  }
  G4VParticleChange* AlongStepDoIt(const G4Track& tr, const G4Step& st) override {
    aParticleChange.Initialize(tr);
    const double a = coefficient(tr);
    const double e0 = st.GetPreStepPoint()->GetKineticEnergy() / MeV;
    const double s = std::sqrt(e0) - 0.5 * a * st.GetStepLength() / mm;
    const double e1 = s > 0 ? s * s : 0.0;
    aParticleChange.ProposeEnergy(e1 * MeV);
    aParticleChange.ProposeLocalEnergyDeposit((e0 - e1) * MeV);
    if (e1 <= 0) aParticleChange.ProposeTrackStatus(fStopAndKill);
    return &aParticleChange;
  }
};

class IonPhysics : public G4VPhysicsConstructor {
 public:
  explicit IonPhysics(const Input& in) : G4VPhysicsConstructor("simws_ion"), in_(in) {}
  void ConstructParticle() override {
    G4BosonConstructor().ConstructParticle(); G4LeptonConstructor().ConstructParticle();
    G4MesonConstructor().ConstructParticle(); G4BaryonConstructor().ConstructParticle();
    G4IonConstructor().ConstructParticle();
  }
  void ConstructProcess() override {
    auto* ion = G4GenericIon::GenericIon();
    auto* pm = ion->GetProcessManager();
    if (in_.stopping == "empirical") {
      auto* ioni = new G4ionIonisation();
      ioni->SetEmModel(new G4IonParametrisedLossModel());
      pm->AddProcess(ioni, -1, 2, 2);
    } else {
      pm->AddContinuousProcess(new LindhardScharff());
    }
    auto* nr = new G4ScreenedNuclearRecoil("nuclearStopping", in_.screening, in_.recoils != 0,
                                           in_.recoil_cutoff * eV, in_.physics_cutoff * eV);
    nr->SetMaxEnergyForScattering(100. * MeV);
    pm->AddDiscreteProcess(nr);
  }
 private:
  const Input& in_;
};

class PhysicsList : public G4VModularPhysicsList {
 public:
  explicit PhysicsList(const Input& in) { RegisterPhysics(new IonPhysics(in)); SetDefaultCutValue(1 * km); }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Gun(const Input& in) : in_(in), gun_(new G4ParticleGun(1)) {}
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* ev) override {
    if (!def_) {
      def_ = G4IonTable::GetIonTable()->GetIon(in_.Z, in_.A, 0.0);
      gun_->SetParticleDefinition(def_);
      gun_->SetParticleCharge(in_.Z * eplus);
      gun_->SetParticleEnergy(in_.E_keV * keV);
      double t = in_.tilt * deg;
      gun_->SetParticleMomentumDirection(G4ThreeVector(std::cos(t), std::sin(t), 0));
      gun_->SetParticlePosition(G4ThreeVector(-0.5 * g_total_um * um - 0.5e-3 * um, 0, 0));
    }
    gun_->GeneratePrimaryVertex(ev);
  }
 private:
  const Input& in_; G4ParticleGun* gun_; G4ParticleDefinition* def_ = nullptr;
};

static FILE* g_out = nullptr;
static int g_exit = 0;                                   // how the current primary left: 0 not, 1 front, 2 back

// A moving atom that enters the vacuum world has left the target: it is stopped there, so no process ever acts on
// the world's material (G4ScreenedNuclearRecoil would collide with the hydrogen of G4_Galactic and fail).
class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* st) override {
    auto* pv = st->GetPostStepPoint()->GetPhysicalVolume();
    if (pv == nullptr || pv->GetName() != "world") return;
    G4Track* tr = st->GetTrack();
    tr->SetTrackStatus(fStopAndKill);
    if (tr->GetParentID() == 0) {
      double x = st->GetPostStepPoint()->GetPosition().x() / um + 0.5 * g_total_um;
      g_exit = x < 0.5 * g_total_um ? 1 : 2;
    }
  }
};

class Tracking : public G4UserTrackingAction {
 public:
  explicit Tracking(const Input& in)
      : in_(in), hist_(in.batches, std::vector<long>(in.nbins > 0 ? in.nbins : 1, 0)), outside_(in.batches, 0),
        ions_(in.batches, 0) {}
  int batch() const {
    long id = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    return (int)(id * in_.batches / in_.n);
  }
  void PreUserTrackingAction(const G4Track* tr) override {
    if (tr->GetParentID() == 0) { ions_[batch()]++; g_exit = 0; return; }
    if (!in_.recoils) return;
    double x = tr->GetVertexPosition().x() / um + 0.5 * g_total_um;
    int b = batch();
    if (x < 0 || x > g_total_um) { outside_[b]++; return; }
    long i = (long)std::floor(x / in_.bin_um);
    if (i >= in_.nbins) i = in_.nbins - 1;                   // the last bin takes anything at or beyond its end
    hist_[b][i]++;
  }
  void PostUserTrackingAction(const G4Track* tr) override {
    if (tr->GetParentID() != 0) return;
    auto p = tr->GetPosition();
    double x = p.x() / um + 0.5 * g_total_um, z = p.z() / um;
    int st = g_exit != 0 ? g_exit : (x < 0 ? 1 : (x > g_total_um ? 2 : 0));
    std::fprintf(g_out, "%d %.10e %.10e\n", st, x, z);
  }
  void write() const {
    if (!in_.recoils) return;
    for (int b = 0; b < in_.batches; ++b) {
      std::fprintf(g_out, "D %d %ld %ld", b, ions_[b], outside_[b]);
      for (long c : hist_[b]) std::fprintf(g_out, " %ld", c);
      std::fprintf(g_out, "\n");
    }
  }
 private:
  const Input& in_;
  std::vector<std::vector<long>> hist_;
  std::vector<long> outside_, ions_;
};

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: g4implant input.txt output.txt\n"; return 2; }
  Input in = read_input(argv[1]);
  g_out = std::fopen(argv[2], "w");
  if (!g_out) fail(std::string("cannot write ") + argv[2]);
  CLHEP::HepRandom::setTheSeed(in.seed);
  auto* rm = new G4RunManager();
  auto* par = G4EmParameters::Instance();
  par->SetDefaults();
  par->SetMinEnergy(10 * eV);
  par->SetLowestMuHadEnergy(0.1 * eV);              // G4ScreenedNuclearRecoil decides where a moving atom stops
  par->SetStepFunctionIons(0.1, 1 * nm);
  par->SetFluo(false);
  rm->SetUserInitialization(new Detector(in));
  rm->SetUserInitialization(new PhysicsList(in));
  rm->SetUserAction(new Gun(in));
  auto* tracking = new Tracking(in);
  rm->SetUserAction(tracking);
  rm->SetUserAction(new Stepping());
  rm->Initialize();
  rm->BeamOn(0);                                    // builds the physics tables (no event, no random number)
  std::fprintf(g_out, "# g4implant 1 geant4 %s\n", G4Version.c_str());
  for (auto* m : g_mats)
    std::fprintf(g_out, "# material %s %.10g g/cm3\n", m->GetName().c_str(), m->GetDensity() / (g / cm3));
  {
    auto* def = G4IonTable::GetIonTable()->GetIon(in.Z, in.A, 0.0);
    std::fprintf(g_out, "# ion %d %d mass_u %.10g\n", in.Z, in.A, def->GetPDGMass() / amu_c2);
  }
  if (!in.dedx_keV.empty()) {
    G4EmCalculator calc;
    auto* def = G4IonTable::GetIonTable()->GetIon(in.Z, in.A, 0.0);
    for (double e : in.dedx_keV) {
      double d = in.stopping == "empirical" ? calc.GetDEDX(e * keV, def, g_mats[0])
          : LindhardScharff::coefficient(in.Z, def->GetPDGMass() / amu_c2, g_mats[0]) * std::sqrt(e * keV / MeV) * MeV / mm;
      std::fprintf(g_out, "# dedx_e %.10g keV %.10e MeV_cm2_g\n", e,
                   d / (MeV / cm) / (g_mats[0]->GetDensity() / (g / cm3)));
    }
  }
  std::fflush(g_out);
  rm->BeamOn((G4int)in.n);
  tracking->write();
  std::fclose(g_out);
  delete rm;
  return 0;
}
