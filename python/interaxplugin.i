/* Part of the OpenMM InteraxFF plugin. Copyright (c) 2026 the Authors. MIT license, see LICENSE. */

%module interaxplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_string.i"

/*
 * The following lines are needed to handle std::vector.
 */

%include "std_vector.i"
namespace std {
  %template(vectori) vector<int>;
};

%{
#include "OpenMMInterax.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Convert C++ exceptions to Python exceptions.
 */
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

/*
 * The reference parameters of the get...Parameters() methods are output values.  Marking them as such
 * will cause swig to return a tuple.
 */
%apply double& OUTPUT {double& m1, double& m2, double& m3, double& alpha, double& width, double& amplitude, double& k, double& l};
%apply bool& OUTPUT {bool& isAlchemical};
%apply int& OUTPUT {int& particle1, int& particle2};

/*
 * Docstrings: the signature of every method, and a description of every class (the C++ headers document each method).
 */
%feature("autodoc", "0");
%feature("docstring") InteraxPlugin::InteraxPairForce "The abstract base class of the InteraxFF pair forces: the
cutoff, switching function, exclusions and alchemical scaling that InteraxDispersionForce, InteraxExchangeForce and
InteraxShortRangeForce have in common. Lengths are in nm and energies in kJ/mol.";
%feature("docstring") InteraxPlugin::InteraxDispersionForce "Tang-Toennies damped C6/C8/C10 pair dispersion, with the
coefficients combined from the XDM moments m1, m2, m3 (kJ/mol nm^3, nm^5, nm^7), the polarizability alpha (nm^3) and
the Slater width (nm) of each particle, and an optional long range correction.";
%feature("docstring") InteraxPlugin::InteraxExchangeForce "Exchange repulsion E = sum_ij A_i A_j S(r; s_i, s_j), with S
the overlap integral (nm^-3) of the normalized Slater densities of widths s (nm), and amplitudes A in
sqrt(kJ/mol nm^3).";
%feature("docstring") InteraxPlugin::InteraxShortRangeForce "Short-range correction
E = sum_ij [(K_i + K_j)/2 St + (L_i + L_j)/2 St^2], with St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j) the normalized Slater
overlap, K and L in kJ/mol and widths s in nm.";

namespace InteraxPlugin {

const char* getPluginVersion();
const char* getPluginGitCommit();

class InteraxPairForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2
    };
    virtual int getNumParticles() const = 0;
    virtual bool getParticleIsAlchemical(int index) const = 0;
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    bool getUseSwitchingFunction() const;
    void setUseSwitchingFunction(bool use);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    int getNumExclusions() const;
    int addExclusion(int particle1, int particle2);
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    void setExclusionParticles(int index, int particle1, int particle2);
    const std::string& getAlchemicalParameterName() const;
    void setAlchemicalParameterName(const std::string& name);
    double getSoftcoreRadius() const;
    void setSoftcoreRadius(double radius);
    bool usesPeriodicBoundaryConditions() const;
private:
    InteraxPairForce();
};

class InteraxDispersionForce : public InteraxPairForce {
public:
    InteraxDispersionForce();
    int getNumParticles() const;
    bool getParticleIsAlchemical(int index) const;
    int addParticle(double m1, double m2, double m3, double alpha, double width, bool isAlchemical=false);
    void getParticleParameters(int index, double& m1, double& m2, double& m3, double& alpha, double& width, bool& isAlchemical) const;
    void setParticleParameters(int index, double m1, double m2, double m3, double alpha, double width, bool isAlchemical);
    double getDampingScale() const;
    void setDampingScale(double scale);
    bool getUseLongRangeCorrection() const;
    void setUseLongRangeCorrection(bool use);
    void updateParametersInContext(OpenMM::Context& context);
};

class InteraxExchangeForce : public InteraxPairForce {
public:
    enum OverlapMethod {
        Exact = 0,
        InteraxOrder2 = 2
    };
    InteraxExchangeForce();
    int getNumParticles() const;
    bool getParticleIsAlchemical(int index) const;
    int addParticle(double amplitude, double width, bool isAlchemical=false);
    void getParticleParameters(int index, double& amplitude, double& width, bool& isAlchemical) const;
    void setParticleParameters(int index, double amplitude, double width, bool isAlchemical);
    OverlapMethod getOverlapMethod() const;
    void setOverlapMethod(OverlapMethod method);
    void updateParametersInContext(OpenMM::Context& context);
};

class InteraxShortRangeForce : public InteraxPairForce {
public:
    enum OverlapMethod {
        Exact = 0,
        InteraxOrder2 = 2
    };
    InteraxShortRangeForce();
    int getNumParticles() const;
    bool getParticleIsAlchemical(int index) const;
    int addParticle(double k, double l, double width, bool isAlchemical=false);
    void getParticleParameters(int index, double& k, double& l, double& width, bool& isAlchemical) const;
    void setParticleParameters(int index, double k, double l, double width, bool isAlchemical);
    OverlapMethod getOverlapMethod() const;
    void setOverlapMethod(OverlapMethod method);
    void updateParametersInContext(OpenMM::Context& context);
};

} // namespace InteraxPlugin

/*
 * Add methods for casting a Force to one of the InteraxFF forces.  System.getForce() returns a plain Force:
 * cast() recovers the InteraxFF class, and isinstance() tests whether a Force is an instance of it.
 */
%define CAST_HELPERS(CLASS)
%extend InteraxPlugin::CLASS {
    static InteraxPlugin::CLASS& cast(OpenMM::Force& force) {
        return dynamic_cast<InteraxPlugin::CLASS&>(force);
    }
    static bool isinstance(OpenMM::Force& force) {
        return (dynamic_cast<InteraxPlugin::CLASS*>(&force) != NULL);
    }
}
%enddef
CAST_HELPERS(InteraxDispersionForce)
CAST_HELPERS(InteraxExchangeForce)
CAST_HELPERS(InteraxShortRangeForce)

/*
 * The version of the plugin, as in other OpenMM Python packages.
 */
%pythoncode %{
__version__ = getPluginVersion()
%}
