%module interaxplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_string.i"
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
 * Output arguments are returned as tuples.
 */
%apply double& OUTPUT {double& m1, double& m2, double& m3, double& alpha, double& width, double& amplitude, double& k, double& l, double& frequency};
%apply bool& OUTPUT {bool& isAlchemical};
%apply int& OUTPUT {int& particle1, int& particle2};
/* getMoleculeParameters returns the particles of the molecule as a tuple. */
%typemap(in, numinputs=0) std::vector<int>& particles (std::vector<int> temp) {
    $1 = &temp;
}
%typemap(argout) std::vector<int>& particles {
    PyObject* list = PyTuple_New($1->size());
    for (size_t i = 0; i < $1->size(); i++)
        PyTuple_SET_ITEM(list, i, PyLong_FromLong((*$1)[i]));
    %append_output(list);
}

namespace InteraxPlugin {

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
    InteraxExchangeForce();
    int getNumParticles() const;
    bool getParticleIsAlchemical(int index) const;
    int addParticle(double amplitude, double width, bool isAlchemical=false);
    void getParticleParameters(int index, double& amplitude, double& width, bool& isAlchemical) const;
    void setParticleParameters(int index, double amplitude, double width, bool isAlchemical);
    void updateParametersInContext(OpenMM::Context& context);
};

class InteraxShortRangeForce : public InteraxPairForce {
public:
    InteraxShortRangeForce();
    int getNumParticles() const;
    bool getParticleIsAlchemical(int index) const;
    int addParticle(double k, double l, double width, bool isAlchemical=false);
    void getParticleParameters(int index, double& k, double& l, double& width, bool& isAlchemical) const;
    void setParticleParameters(int index, double k, double l, double width, bool isAlchemical);
    void updateParametersInContext(OpenMM::Context& context);
};

class InteraxThreeBodyForce : public OpenMM::Force {
public:
    InteraxThreeBodyForce();
    int getNumMolecules() const;
    int addMolecule(const std::vector<int>& particles, double alpha, double frequency, bool isAlchemical=false);
    void getMoleculeParameters(int index, std::vector<int>& particles, double& alpha, double& frequency, bool& isAlchemical) const;
    void setMoleculeParameters(int index, const std::vector<int>& particles, double alpha, double frequency, bool isAlchemical);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getScaleFactor() const;
    void setScaleFactor(double scale);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    const std::string& getAlchemicalParameterName() const;
    void setAlchemicalParameterName(const std::string& name);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;
};

}

/*
 * System.getForce() returns a plain Force: cast() recovers the InteraxFF class, isinstance() tests for it.
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
CAST_HELPERS(InteraxThreeBodyForce)
