// g4track: energy deposition of an ion track in a slab, binned in (depth z, radius r from the track axis).
// Output CSV (per primary): z_lo_um,z_hi_um,r_lo_um,r_hi_um,edep_MeV ; header lines start with '#'.
//
//   g4track --Z 26 --A 56 --energy-MeV 1500 --material G4_Si --thickness-um 20 --rmax-um 2 --nr 40 --nz 40
//           --events 200 --cut-um 0.1 --seed 1 --out track.csv
//
// Physics: QBBC hadronic + G4EmStandardPhysics_option4 (most accurate low-energy EM). Serial run manager.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4IonTable.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4Version.hh"
#include "Randomize.hh"

struct Args {
  int Z = 26, A = 56, nr = 40, nz = 40, events = 100;
  long seed = 1;
  double energy_MeV = 1500, thickness_um = 20, rmax_um = 2, cut_um = 0.1;
  std::string material = "G4_Si", out = "track.csv";
};

static Args parse(int argc, char** argv) {
  Args a;
  std::map<std::string, std::string> kv;
  for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
  auto get = [&](const char* k) -> const std::string* { auto it = kv.find(k); return it == kv.end() ? nullptr : &it->second; };
  if (auto v = get("--Z")) a.Z = std::stoi(*v);
  if (auto v = get("--A")) a.A = std::stoi(*v);
  if (auto v = get("--energy-MeV")) a.energy_MeV = std::stod(*v);
  if (auto v = get("--material")) a.material = *v;
  if (auto v = get("--thickness-um")) a.thickness_um = std::stod(*v);
  if (auto v = get("--rmax-um")) a.rmax_um = std::stod(*v);
  if (auto v = get("--nr")) a.nr = std::stoi(*v);
  if (auto v = get("--nz")) a.nz = std::stoi(*v);
  if (auto v = get("--events")) a.events = std::stoi(*v);
  if (auto v = get("--cut-um")) a.cut_um = std::stod(*v);
  if (auto v = get("--seed")) a.seed = std::stol(*v);
  if (auto v = get("--out")) a.out = *v;
  return a;
}

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Args& a) : a_(a) {}
  G4VPhysicalVolume* Construct() override {
    auto nist = G4NistManager::Instance();
    auto vac = nist->FindOrBuildMaterial("G4_Galactic");
    auto mat = nist->FindOrBuildMaterial(a_.material);
    const double half_t = 0.5 * a_.thickness_um * um, half_l = 50 * a_.rmax_um * um + 50 * um;
    auto world_s = new G4Box("World", 2 * half_l, 2 * half_l, 4 * half_t + 10 * um);
    auto world_l = new G4LogicalVolume(world_s, vac, "World");
    auto world_p = new G4PVPlacement(nullptr, {}, world_l, "World", nullptr, false, 0);
    auto slab_s = new G4Box("Slab", half_l, half_l, half_t);
    slab_l_ = new G4LogicalVolume(slab_s, mat, "Slab");
    new G4PVPlacement(nullptr, {}, slab_l_, "Slab", world_l, false, 0);
    return world_p;
  }
  G4LogicalVolume* slab() const { return slab_l_; }
 private:
  Args a_;
  G4LogicalVolume* slab_l_ = nullptr;
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Gun(const Args& a) : a_(a), gun_(new G4ParticleGun(1)) {}
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* ev) override {
    if (!ion_) {
      ion_ = G4IonTable::GetIonTable()->GetIon(a_.Z, a_.A, 0.0);
      gun_->SetParticleDefinition(ion_);
      gun_->SetParticleEnergy(a_.energy_MeV * MeV);
      gun_->SetParticleMomentumDirection({0, 0, 1});
      gun_->SetParticlePosition({0, 0, -0.5 * a_.thickness_um * um - 1 * um});
    }
    gun_->GeneratePrimaryVertex(ev);
  }
 private:
  Args a_;
  G4ParticleGun* gun_;
  G4ParticleDefinition* ion_ = nullptr;
};

class Scorer : public G4UserSteppingAction {
 public:
  Scorer(const Args& a, const Detector* d) : a_(a), det_(d), h_((a.nr + 1) * a.nz, 0.0) {}
  void UserSteppingAction(const G4Step* s) override {
    const double e = s->GetTotalEnergyDeposit();
    if (e <= 0 || s->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume() != det_->slab()) return;
    total_ += e;
    // deposit at a uniformly sampled point along the step (unbiased for steps crossing bin edges)
    const G4ThreeVector p = s->GetPreStepPoint()->GetPosition() +
                            G4UniformRand() * (s->GetPostStepPoint()->GetPosition() - s->GetPreStepPoint()->GetPosition());
    const double z = (p.z() + 0.5 * a_.thickness_um * um) / um, r = std::hypot(p.x(), p.y()) / um;
    int iz = static_cast<int>(z / a_.thickness_um * a_.nz);
    if (iz < 0 || iz >= a_.nz) return;
    int ir = r >= a_.rmax_um ? a_.nr : static_cast<int>(r / a_.rmax_um * a_.nr);   // last bin = r >= rmax
    h_[iz * (a_.nr + 1) + ir] += e;
  }
  const std::vector<double>& hist() const { return h_; }
  double total() const { return total_; }
 private:
  Args a_;
  const Detector* det_;
  std::vector<double> h_;
  double total_ = 0;
};

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  G4Random::setTheSeed(a.seed);
  auto rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
  auto det = new Detector(a);
  rm->SetUserInitialization(det);
  G4PhysListFactory factory;
  G4VModularPhysicsList* phys = factory.GetReferencePhysList("QBBC");
  phys->ReplacePhysics(new G4EmStandardPhysics_option4());
  phys->SetDefaultCutValue(a.cut_um * um);
  phys->SetVerboseLevel(0);
  rm->SetUserInitialization(phys);
  rm->SetUserAction(new Gun(a));
  auto scorer = new Scorer(a, det);
  rm->SetUserAction(scorer);
  rm->Initialize();
  rm->BeamOn(a.events);

  G4EmCalculator calc;
  auto ion = G4IonTable::GetIonTable()->GetIon(a.Z, a.A, 0.0);
  auto mat = G4NistManager::Instance()->FindOrBuildMaterial(a.material);
  const double dedx = calc.ComputeTotalDEDX(a.energy_MeV * MeV, ion, mat);           // MeV/mm internally
  const double rho = mat->GetDensity() / (g / cm3);

  std::ofstream f(a.out);
  f.precision(10);
  f << "# g4track Geant4 " << G4Version << "\n"
    << "# ion_Z=" << a.Z << " ion_A=" << a.A << " energy_MeV=" << a.energy_MeV << " material=" << a.material
    << " density_g_cm3=" << rho << " thickness_um=" << a.thickness_um << " rmax_um=" << a.rmax_um
    << " nr=" << a.nr << " nz=" << a.nz << " events=" << a.events << " cut_um=" << a.cut_um << " seed=" << a.seed << "\n"
    << "# entrance_dEdx_total_MeV_per_um=" << dedx / (MeV / um)
    << " entrance_LET_MeV_cm2_per_mg=" << (dedx / (MeV / cm)) / (rho * 1000.0) << "\n"
    << "# total_edep_MeV_per_event=" << scorer->total() / MeV / a.events << "\n"
    << "z_lo_um,z_hi_um,r_lo_um,r_hi_um,edep_MeV\n";
  const double dz = a.thickness_um / a.nz, dr = a.rmax_um / a.nr;
  for (int iz = 0; iz < a.nz; ++iz)
    for (int ir = 0; ir <= a.nr; ++ir) {
      const double r_hi = ir == a.nr ? -1.0 : (ir + 1) * dr;   // -1 marks the r >= rmax overflow bin
      f << iz * dz << "," << (iz + 1) * dz << "," << ir * dr << "," << r_hi << ","
        << scorer->hist()[iz * (a.nr + 1) + ir] / MeV / a.events << "\n";
    }
  delete rm;
  std::cout << "g4track: wrote " << a.out << " (" << a.events << " events, entrance LET "
            << (dedx / (MeV / cm)) / (rho * 1000.0) << " MeV cm2/mg)\n";
  return 0;
}
