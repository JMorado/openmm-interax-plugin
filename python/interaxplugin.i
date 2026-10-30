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

/*
 * Accept any Python sequence for a std::vector<double> argument, and return std::vector<double> values as lists.
 * (OpenMM's own typemaps for these need its numpy helpers.)
 */
%typemap(in) const std::vector<double>& (std::vector<double> temp) {
    if (!PySequence_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence of numbers");
        SWIG_fail;
    }
    Py_ssize_t n = PySequence_Size($input);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_GetItem($input, i);
        temp.push_back(PyFloat_AsDouble(item));
        Py_DECREF(item);
    }
    if (PyErr_Occurred())
        SWIG_fail;
    $1 = &temp;
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_DOUBLE_ARRAY) const std::vector<double>& {
    $1 = PySequence_Check($input) ? 1 : 0;
}
%typemap(out) std::vector<double> {
    const std::vector<double>& values = $1;
    $result = PyList_New(values.size());
    for (size_t i = 0; i < values.size(); i++)
        PyList_SET_ITEM($result, i, PyFloat_FromDouble(values[i]));
}
%typemap(out) const std::vector<double>& {
    $result = PyList_New($1->size());
    for (size_t i = 0; i < $1->size(); i++)
        PyList_SET_ITEM($result, i, PyFloat_FromDouble((*$1)[i]));
}

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
%apply double& OUTPUT {double& m1, double& m2, double& m3, double& alpha, double& width, double& amplitude, double& k, double& l, double& frequency};
%apply bool& OUTPUT {bool& isAlchemical};
%apply int& OUTPUT {int& particle1, int& particle2, int& molecule};
/*
 * getSiteParameters() returns the particles of the site as a tuple.
 */
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
        InteraxOrder2 = 2,
        InteraxOrder6 = 6
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
        InteraxOrder2 = 2,
        InteraxOrder6 = 6
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

class InteraxThreeBodyForce : public OpenMM::Force {
public:
    enum DampingMethod {
        NoDamping = 0,
        TangToennies = 1
    };
    InteraxThreeBodyForce();
    int getNumSites() const;
    int addSite(const std::vector<int>& particles, double alpha, double frequency, double width=0.0, int molecule=-1, bool isAlchemical=false);
    void getSiteParameters(int index, std::vector<int>& particles, double& alpha, double& frequency, double& width, int& molecule, bool& isAlchemical) const;
    void setSiteParameters(int index, const std::vector<int>& particles, double alpha, double frequency, double width, int molecule, bool isAlchemical);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getScaleFactor() const;
    void setScaleFactor(double scale);
    DampingMethod getDampingMethod() const;
    void setDampingMethod(DampingMethod method);
    double getDampingScale() const;
    void setDampingScale(double scale);
    double getCentroidDampingRadius() const;
    void setCentroidDampingRadius(double radius);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    const std::string& getAlchemicalParameterName() const;
    void setAlchemicalParameterName(const std::string& name);
    void updateParametersInContext(OpenMM::Context& context);
};

class InteraxMultipoleForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        PME = 1
    };
    enum ParticleAxisTypes { ZThenX = 0, Bisector = 1, ZBisect = 2, ThreeFold = 3, ZOnly = 4, NoAxisType = 5 };
    enum ChiralityMode { NoChirality = 0, AmoebaChirality = 1 };
    InteraxMultipoleForce();
    int getNumParticles() const;
    int getNumExceptions() const;
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    const std::vector<double>& getExtrapolationCoefficients() const;
    void setExtrapolationCoefficients(const std::vector<double>& coefficients);
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    int addParticle(double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                    double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                    bool isAlchemical=false);
    void setParticleParameters(int index, double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                               double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                               bool isAlchemical=false);
    int addException(int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale, double dipoleDipoleScale,
                     bool replace=false);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    bool getCoreDamping() const;
    void setCoreDamping(bool enabled);
    double getCoreDampingScale() const;
    void setCoreDampingScale(double eta);
    double getMutualDampingScale() const;
    void setMutualDampingScale(double kappa);
    bool getUseOwnKernel() const;
    void setUseOwnKernel(bool enabled);
    ChiralityMode getChiralityMode() const;
    void setChiralityMode(ChiralityMode mode);
    const std::string& getAlchemicalParameterName() const;
    void setAlchemicalParameterName(const std::string& name);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;
};

/*
 * Methods with vector or multiple output arguments are wrapped as methods that return flat lists: the particle
 * parameters as [charge, dipole (3), quadrupole (9), coreCharge, alpha, polarizability, axisType, z, x, y,
 * isAlchemical], the dipoles as [x0, y0, z0, x1, ...] (e nm), and the PME parameters as [alpha, nx, ny, nz].
 */
%extend InteraxMultipoleForce {
    std::vector<double> getParticleParametersList(int index) const {
        double charge, coreCharge, alpha, polarizability;
        int axisType, z, x, y;
        bool isAlchemical;
        std::vector<double> dipole, quadrupole;
        self->getParticleParameters(index, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, z, x, y, isAlchemical);
        std::vector<double> out = {charge};
        out.insert(out.end(), dipole.begin(), dipole.end());
        out.insert(out.end(), quadrupole.begin(), quadrupole.end());
        std::vector<double> rest = {coreCharge, alpha, polarizability, (double) axisType, (double) z, (double) x, (double) y, (double) isAlchemical};
        out.insert(out.end(), rest.begin(), rest.end());
        return out;
    }
    std::vector<double> getInducedDipolesList(OpenMM::Context& context) {
        std::vector<OpenMM::Vec3> d;
        self->getInducedDipoles(context, d);
        std::vector<double> out;
        for (auto& v : d) {
            out.push_back(v[0]);
            out.push_back(v[1]);
            out.push_back(v[2]);
        }
        return out;
    }
    std::vector<double> getLabFramePermanentDipolesList(OpenMM::Context& context) {
        std::vector<OpenMM::Vec3> d;
        self->getLabFramePermanentDipoles(context, d);
        std::vector<double> out;
        for (auto& v : d) {
            out.push_back(v[0]);
            out.push_back(v[1]);
            out.push_back(v[2]);
        }
        return out;
    }
    std::vector<double> getPMEParametersInContextList(const OpenMM::Context& context) const {
        double alpha;
        int nx, ny, nz;
        self->getPMEParametersInContext(context, alpha, nx, ny, nz);
        return {alpha, (double) nx, (double) ny, (double) nz};
    }
}

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
CAST_HELPERS(InteraxThreeBodyForce)
CAST_HELPERS(InteraxMultipoleForce)
