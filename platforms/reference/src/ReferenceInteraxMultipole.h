#ifndef OPENMM_REFERENCEINTERAXMULTIPOLE_H_
#define OPENMM_REFERENCEINTERAXMULTIPOLE_H_

/* Portions copyright (c) 2006-2022 Stanford University and Simbios.
 * Contributors: Pande Group
 * Portions copyright (c) 2026 the Authors.
 * Authors: Joao Morado
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "InteraxMultipoleForce.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <array>
#include <map>
#include <utility>
#include <vector>
#include <complex>

namespace InteraxPlugin {

/**
 * 4-dimensional double vector
 */
class InteraxDouble4 {
public:
    /**
     * Create an InteraxDouble4 whose elements are all 0.
     */
    InteraxDouble4() {
        data[0] = data[1] = data[2] = data[3] = 0.0;
    }
    /**
     * Create an InteraxDouble4 with specified x, y, z, w components.
     */
    InteraxDouble4(double x, double y, double z, double w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
    double operator[](int index) const {
        assert(index >= 0 && index < 4);
        return data[index];
    }
    double& operator[](int index) {
        assert(index >= 0 && index < 4);
        return data[index];
    }

private:
    double data[4];
};

/**
 * This class computes InteraxMultipoleForce on the Reference platform without a cutoff: the multipoles in the lab
 * frame, the induced dipoles, and the energy, forces and torques of the electrostatics and polarization.  The CPU
 * platform uses it too.
 */
class ReferenceInteraxMultipole {

public:

    /**
     * Constructor
     *
     */
    ReferenceInteraxMultipole(const InteraxMultipoleForce& force);

    /**
     * Get nonbonded method.
     *
     * @return nonbonded method
     */
    InteraxMultipoleForce::NonbondedMethod getNonbondedMethod() const;

    /**
     * Set the value of the alchemical parameter, which scales the charges, multipoles and polarizabilities
     * of the alchemical particles.
     *
     * @param lambda    the value of the alchemical parameter
     */
    void setLambda(double lambda) {
        _lambda = lambda;
    }

    /**
     * Set the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.
     *
     * @param coefficients    a vector whose mth entry specifies the coefficient for mu_m
     */
    void setExtrapolationCoefficients(const std::vector<double> &coefficients);

    /**
     * Calculate force and energy.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param forces                    add forces to this vector
     *
     * @return energy
     */
    double calculateForceAndEnergy(const std::vector<OpenMM::Vec3>& particlePositions,
                                   std::vector<OpenMM::Vec3>& forces);

    /**
     * Calculate particle induced dipoles.
     *
     * @param particlePositions            Cartesian coordinates of particles
     * @param[out] outputInducedDipoles    the induced dipole of each particle
     */
    void calculateInducedDipoles(const std::vector<OpenMM::Vec3>& particlePositions,
                                 std::vector<OpenMM::Vec3>& outputInducedDipoles);

    /**
     * Calculate particle permanent dipoles rotated in the lab frame.
     *
     * @param particlePositions                     Cartesian coordinates of particles
     * @param[out] outputRotatedPermanentDipoles    the permanent dipole of each particle in the lab frame
     */
    void calculateLabFramePermanentDipoles(const std::vector<OpenMM::Vec3>& particlePositions,
                                           std::vector<OpenMM::Vec3>& outputRotatedPermanentDipoles);

protected:

    enum QuadrupoleIndices { QXX, QXY, QXZ, QYY, QYZ, QZZ };

    /*
     * Particle parameters and coordinates
     */
    class MultipoleParticleData {
        public:
            int index, axisType, multipoleAtomX, multipoleAtomY, multipoleAtomZ;
            OpenMM::Vec3 position;
            OpenMM::Vec3 dipole, localDipole, qiDipole, qiInducedDipole;
            double quadrupole[6], localQuadrupole[6], qiQuadrupole[6];
            double coreCharge, valenceCharge, alpha, polarizability;
            // Whether the particle is alchemical, and its parameters before scaling by the alchemical
            // parameter (loadParticleData() sets the scaled ones)
            bool isAlchemical;
            OpenMM::Vec3 baseLocalDipole;
            double baseLocalQuadrupole[6], baseCoreCharge, baseValenceCharge, basePolarizability;
    };

    /**
     * Particle parameters transformed into fractional coordinates
     */
    class TransformedMultipole {
    public:
        double charge;
        OpenMM::Vec3 dipole;
        double quadrupole[6];
    };

    /**
     * Information defining an exception.
     */
    class Exception {
    public:
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
    };

    unsigned int _numParticles;
    InteraxMultipoleForce::NonbondedMethod _nonbondedMethod;
    double _electric, _cutoffDistance, _cutoffDistanceSquared;
    // Core damping of the polarizing fields, the scale of the exponents in the damping of the induced
    // dipole couplings, and the value of the alchemical parameter
    bool _coreDamping;
    double _coreDampingScale, _mutualDampingScale, _lambda;
    InteraxMultipoleForce::ChiralityMode _chiralityMode;
    std::map<std::pair<int, int>, Exception> exceptions;
    std::vector<MultipoleParticleData> particleData;
    std::vector<TransformedMultipole> _transformed;
    std::vector<OpenMM::Vec3> _fixedMultipoleField;
    std::vector<OpenMM::Vec3> _polarizingField; // The permanent field at each particle, before multiplying by the polarizability
    std::vector<OpenMM::Vec3> _inducedDipole;
    std::vector<OpenMM::Vec3> _inducedDipoleField;
    std::vector<std::vector<OpenMM::Vec3> > _ptDipoleD;

    int _maxPTOrder;
    std::vector<double> _extrapolationCoefficients;
    std::vector<double> _extPartCoefficients;

    /**
     * Load particle data.
     *
     * @param particlePositions   particle coordinates
     */
    void loadParticleData(const std::vector<OpenMM::Vec3>& particlePositions);

    /**
     * Calculate fixed multipole fields.
     */
    virtual void calculateFixedMultipoleField();

    /**
     * Compute the damping factors for the field due to the fixed multipole moments of a particle.
     *
     * @param particle     parameters for the particle
     * @param r            the distance from the particle at which to calculate the damping factors
     * @param fdamp3       outputs the damping factor for the r^-3 term
     * @param fdamp5       outputs the damping factor for the r^-5 term
     * @param fdamp7       outputs the damping factor for the r^-7 term
     */
    void computeDirectFieldDampingFactors(const MultipoleParticleData& particle, double r, double& fdamp3, double& fdamp5, double& fdamp7) const;
    /**
     * Compute the damping factors for the field at one particle due to the induced dipole of another particle.
     *
     * @param particleI    parameters for the first particle
     * @param particleJ    parameters for the second particle
     * @param r            the distance between the two particles
     * @param fdamp3       outputs the damping factor for the r^-3 term
     * @param fdamp5       outputs the damping factor for the r^-5 term
     */
    void computeMutualFieldDampingFactors(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ, double r, double& fdamp3, double& fdamp5) const;
    /**
     * Compute the damping factors for the overlap of two particles.
     *
     * @param particleI    parameters for the first particle
     * @param particleJ    parameters for the second particle
     * @param r            the distance between the two particles
     * @param fdampI1      outputs the damping factor for the r^-1 term for the valence charge of particle I
     * @param fdampI3      outputs the damping factor for the r^-3 term for the valence charge of particle I
     * @param fdampI5      outputs the damping factor for the r^-5 term for the valence charge of particle I
     * @param fdampI7      outputs the damping factor for the r^-7 term for the valence charge of particle I
     * @param fdampI9      outputs the damping factor for the r^-9 term for the valence charge of particle I
     * @param fdampJ1      outputs the damping factor for the r^-1 term for the valence charge of particle J
     * @param fdampJ3      outputs the damping factor for the r^-3 term for the valence charge of particle J
     * @param fdampJ5      outputs the damping factor for the r^-5 term for the valence charge of particle J
     * @param fdampJ7      outputs the damping factor for the r^-7 term for the valence charge of particle J
     * @param fdampJ9      outputs the damping factor for the r^-9 term for the valence charge of particle J
     * @param fdampIJ1     outputs the damping factor for the r^-1 term for the overlap between valence charges
     * @param fdampIJ3     outputs the damping factor for the r^-3 term for the overlap between valence charges
     * @param fdampIJ5     outputs the damping factor for the r^-5 term for the overlap between valence charges
     * @param fdampIJ7     outputs the damping factor for the r^-7 term for the overlap between valence charges
     * @param fdampIJ9     outputs the damping factor for the r^-9 term for the overlap between valence charges
     * @param fdampIJ11    outputs the damping factor for the r^-9 term for the overlap between valence charges
     */
    void computeOverlapDampingFactors(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ, double r,
            double& fdampI1, double& fdampI3, double& fdampI5, double& fdampI7, double& fdampI9,
            double& fdampJ1, double& fdampJ3, double& fdampJ5, double& fdampJ7, double& fdampJ9,
            double& fdampIJ1, double& fdampIJ3, double& fdampIJ5, double& fdampIJ7, double& fdampIJ9, double& fdampIJ11) const;
    /**
     * Compute the damping factors for the overlap of two Slater densities with different exponents (the
     * expressions used by computeOverlapDampingFactors() for unequal exponents, and a series in the difference of the
     * exponents when they are nearly equal, where those expressions lose precision).
     *
     * @param aI       the exponent of the first density
     * @param aJ       the exponent of the second density
     * @param r        the distance between the two particles
     * @param fdamp    outputs the damping factors for the r^-1, r^-3, r^-5, r^-7, r^-9 and r^-11 terms
     */
    static void computePairDampingFactors(double aI, double aJ, double r, double (&fdamp)[6]);
    /**
     * Check if multipoles at chiral site should be inverted.
     *
     * @param  particleI            particleI data
     * @param  axisType             axis type
     * @param  particleZ            z-axis particle to particleI
     * @param  particleX            x-axis particle to particleI
     * @param  particleY            y-axis particle to particleI
     *
     */
    void checkChiralCenterAtParticle(MultipoleParticleData& particleI, int axisType, MultipoleParticleData& particleZ,
                                     MultipoleParticleData& particleX, MultipoleParticleData& particleY);

    /**
     * Invert multipole moments (dipole[Y], quadrupole[XY] and quadrupole[YZ]) if chiral center inverted.
     */
    void checkChiral();

    /**
     * Whether a pair is an exception with all scale factors zero and the method is NoCutoff, so that the
     * pair contributes nothing and is skipped. This allows particles at the same position (bond multipole sites on
     * their host atom), whose pair terms would otherwise be evaluated at r = 0 and multiplied by zero.
     */
    bool isFullyExcluded(int i, int j) const;

    /**
     * Pairs closer than this (nm) are taken to be at the same position. Only fully excluded pairs may be;
     * with PME their real-space terms use the r = 0 limits (ReferenceInteraxMultipolePme::coincidentEwaldLimit()).
     */
    static constexpr double COINCIDENT_DISTANCE = 1e-6;
    /**
     * Apply rotation matrix to molecular dipole/quadrupoles to get corresponding lab frame values
     * for particle I.
     *
     * @param particleI    the particle
     * @param particleZ    the z particle of its frame
     * @param particleX    the x particle of its frame, or nullptr
     * @param particleY    the y particle of its frame, or nullptr
     * @param axisType     the axis type
     */
    void applyRotationMatrixToParticle(      MultipoleParticleData& particleI,
                                       const MultipoleParticleData* particleZ,
                                       const MultipoleParticleData* particleX,
                                             MultipoleParticleData* particleY, int axisType) const;

    /**
     * Forms the rotation matrix for the quasi-internal coordinate system,
     * which is the rotation matrix that describes the orientation of the
     * internuclear vector for a given pair (I,J) in lab frame.
     *
     * @param deltaR                the internuclear vector, corrected for periodic boundary conditions
     * @param r                     the distance between atoms I and J
     * @param rotationMatrix        the output rotation matrix for a 3-vector
     */
    void formQIRotationMatrix(const OpenMM::Vec3 &deltaR, double r, double (&rotationMatrix)[3][3]) const;

    /**
     * Rotate a vector from the lab frame to the quasi-internal coordinate system.
     *
     * @param v      the vector to rotate
     * @param mat    the rotation matrix computed by formQIRotationMatrix()
     * @return the rotated vector
     */
    OpenMM::Vec3 rotateVectorToQI(const OpenMM::Vec3 v, const double (&mat)[3][3]) const;

    /**
     * Rotate a vector from the quasi-internal coordinate system back to the lab frame.
     *
     * @param v      the vector to rotate
     * @param mat    the rotation matrix computed by formQIRotationMatrix()
     * @return the rotated vector
     */
    OpenMM::Vec3 rotateVectorFromQI(const OpenMM::Vec3 v, const double (&mat)[3][3]) const;

    /**
     * Rotate a quadrupole from the lab frame to the quasi-internal coordinate system.
     *
     * @param q      the quadrupole to rotate
     * @param r      the rotated quadrupole is stored into this
     * @param mat    the rotation matrix computed by formQIRotationMatrix()
     */
    void rotateQuadrupoleToQI(const double (&q)[6], double (&r)[6], const double (&mat)[3][3]) const;

    /**
     * Apply rotation matrix to molecular dipole/quadrupoles of all particles to get corresponding lab frame values.
     */
    void applyRotationMatrix();

    /**
     * Calculate electric field at particle I due fixed multipoles at particle J and vice versa
     * (field at particle J due fixed multipoles at particle I).
     *
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    virtual void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ);

    /**
     * Initialize induced dipoles
     */
    virtual void initializeInducedDipoles();

    /**
     * Calculate fields due induced dipoles at each site.
     *
     * @param particleI     positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ     positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    void calculateInducedDipolePairIxns(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ);

    /**
     * Calculate induced dipole fields.
     *
     * @param particleData    vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param optOrder        the order of the extrapolated perturbation series being computed
     */
    virtual void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData, int optOrder);
    /**
     * Calculated induced dipoles using extrapolated perturbation theory.
     *
     * @param particleData   vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void convergeInduceDipolesByExtrapolation(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Calculate induced dipoles.
     */
    virtual void calculateInducedDipoles();

    /**
     * Setup:
     *        if needed invert multipole moments at chiral centers
     *        rotate molecular multipole moments to lab frame
     *        setup scaling maps and
     *        calculate induced dipoles (see calculateInducedDipoles below)
     *
     * @param particlePositions         Cartesian coordinates of particles
     */
    void setup(const std::vector<OpenMM::Vec3>& particlePositions);

    /**
     * Calculate electrostatic interaction between particles I and K.
     *
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param r                 the distance between the two particles
     * @param force             the force to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueI           the torque to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueK           the torque to apply (in the quasi-internal frame) to particle K should be added to this
     */
    virtual double calculateElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                                 double r, OpenMM::Vec3& force, OpenMM::Vec3& torqueI, OpenMM::Vec3& torqueK) const;

    /**
     * Calculate electrostatic interactions involving induced dipoles on particles I and K.
     *
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param deltaR            the displacement between the two particles (in the lab frame)
     * @param r                 the distance between the two particles
     * @param force             the force to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueI           the torque to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueK           the torque to apply (in the quasi-internal frame) to particle K should be added to this
     * @param labForce          the force to apply (in the lab frame) to particle I should be added to this
     */
    virtual void calculateInducedDipolePairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                               OpenMM::Vec3 deltaR, double r, OpenMM::Vec3& force, OpenMM::Vec3& torqueI, OpenMM::Vec3& torqueK, OpenMM::Vec3& labForce) const;

    /**
     * Compute the field at particle I due to the fixed multipoles of particle J, without the Ewald
     * splitting and scaled by the exception scale.
     *
     * @param particleI    positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ    positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param deltaR       the position of particle J relative to particle I
     * @return the field, in the sign convention of _fixedMultipoleField (from which it is subtracted)
     */
    OpenMM::Vec3 calculateDirectPairField(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ, const OpenMM::Vec3& deltaR) const;

    /**
     * Compute the core damping factor f(r) = 1 - (1 + ar + (ar)^2/2) exp(-ar) of the field at a particle,
     * where a is the core damping scale times the exponent of the particle.
     *
     * @param particle     parameters for the polarized particle
     * @param r            the distance from the particle
     * @param f            outputs the damping factor
     * @param dfdr         outputs the derivative of the damping factor with respect to r
     */
    void computeCoreDampingFactor(const MultipoleParticleData& particle, double r, double& f, double& dfdr) const;

    /**
     * Apply core damping to the fixed multipole fields, by adding (f-1) times the field of each other
     * particle (within the cutoff) to the field at each particle.
     */
    void addCoreDampingFields();

    /**
     * Calculate the forces and torques that core damping of the fixed multipole fields adds to the
     * interaction of the induced dipoles on particles I and K (computed by calculateInducedDipolePairIxn() without it).
     *
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param r                 the distance between the two particles
     * @param force             the force to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueI           the torque to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueK           the torque to apply (in the quasi-internal frame) to particle K should be added to this
     */
    void calculateCoreDampingPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                     double r, OpenMM::Vec3& force, OpenMM::Vec3& torqueI, OpenMM::Vec3& torqueK) const;

    /**
     * Map particle torque to force.
     *
     * @param particleI               particle whose torque is to be mapped
     * @param particleU               particle1 of lab frame for particleI
     * @param particleV               particle2 of lab frame for particleI
     * @param particleW               particle3 of lab frame for particleI
     * @param axisType                axis type (Bisector/Z-then-X, ...)
     * @param torque                  torque on particle I
     * @param forces                  vector of particle forces to be updated
     */
    void mapTorqueToForceForParticle(const MultipoleParticleData& particleI,
                                     const MultipoleParticleData& particleU,
                                     const MultipoleParticleData& particleV,
                                           MultipoleParticleData* particleW,
                                     int axisType, const OpenMM::Vec3& torque, std::vector<OpenMM::Vec3>& forces);

    /**
     * Map torques to forces.
     *
     * @param torques                 output torques
     * @param forces                  output forces
     */
    void mapTorqueToForce(std::vector<OpenMM::Vec3>& torques,
                          std::vector<OpenMM::Vec3>& forces);

    /**
     * Calculate the forces and energy.
     *
     * @param torques    output torques
     * @param forces     output forces
     * @return energy
     */
    virtual double calculateInteractions(std::vector<OpenMM::Vec3>& torques,
                                         std::vector<OpenMM::Vec3>& forces);

    /**
     * Normalize a Vec3.
     *
     * @param vectorToNormalize    vector to normalize
     * @return norm of vector on input
     */
    double normalizeVec3(OpenMM::Vec3& vectorToNormalize) const;

    /**
     * Initialize vector of Vec3 (size=numParticles).
     *
     * @param vectorToInitialize    vector to initialize
     */
    void initializeVec3Vector(std::vector<OpenMM::Vec3>& vectorToInitialize) const;

    /**
     * Apply periodic boundary conditions to difference in positions.
     *
     * @param deltaR    difference in particle positions; modified on output after applying PBC
     */
    virtual void getPeriodicDelta(OpenMM::Vec3& deltaR) const {};
};

/**
 * This class computes InteraxMultipoleForce on the Reference platform with PME: the real-space interactions within the
 * cutoff, and the reciprocal-space and self terms of the multipoles and induced dipoles.
 */
class ReferenceInteraxMultipolePme : public ReferenceInteraxMultipole {

public:

    /**
     * Constructor
     *
     */
    ReferenceInteraxMultipolePme(const InteraxMultipoleForce& force, const OpenMM::System& system);

    /**
     * Destructor
     *
     */
    ~ReferenceInteraxMultipolePme();

    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     *
     */
    double getCutoffDistance() const;

    /**
     * Get alpha used in Ewald summation.
     *
     * @return alpha
     *
     */
    double getAlphaEwald() const;

    /**
     * Get PME grid dimensions.
     *
     * @param pmeGridDimensions    contains PME grid dimensions upon return
     */
    void getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const;

    /**
     * Set periodic box size.
     *
     * @param vectors    the vectors defining the periodic box
     */
     void setPeriodicBoxSize(OpenMM::Vec3* vectors);

private:

    static const int AMOEBA_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;

    OpenMM::Vec3 _recipBoxVectors[3];
    OpenMM::Vec3 _periodicBoxVectors[3];

    int _pmeGridDimensions[3];

    std::vector<std::complex<double> > _pmeGrid;

    std::vector<double> _pmeBsplineModuli[3];
    std::vector<InteraxDouble4> _thetai[3];
    std::vector<std::array<int, 3> > _iGrid;
    std::vector<double> _phi;
    std::vector<double> _phidp;
    std::vector<std::vector<double> > optPhi;

    /**
     * Resize PME arrays.
     *
     */
    void resizePmeArrays();

    /**
     * Zero Pme grid.
     */
    void initializePmeGrid();

    /**
     * Modify input vector of differences in particle positions for periodic boundary conditions.
     *
     * @param deltaR    input vector of difference in particle positions; on output adjusted for
     *                  periodic boundary conditions
     */
    void getPeriodicDelta(OpenMM::Vec3& deltaR) const;

    /**
     * Initialize B-spline moduli.
     *
     */
    void initializeBSplineModuli();

    /**
     * Calculate direct-space field at site I due fixed multipoles at site J and vice versa.
     *
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ);

    /**
     * Calculate fixed multipole fields.
     */
    void calculateFixedMultipoleField();

    /**
     * This is called from computeAmoebaBsplines().  It calculates the spline coefficients for a single atom along a single axis.
     *
     * @param thetai    output spline coefficients
     * @param w         offset from grid point
     */
    void computeBSplinePoint(std::vector<InteraxDouble4>& thetai, double w);

    /**
     * Compute bspline coefficients.
     *
     * @param particleData   vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void computeAmoebaBsplines(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Transform multipoles from cartesian coordinates to fractional coordinates.
     */
    void transformMultipolesToFractionalCoordinates(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Transform potential from fractional coordinates to cartesian coordinates.
     */
    void transformPotentialToCartesianCoordinates(const std::vector<double>& fphi, std::vector<double>& cphi) const;

    /**
     * Spread fixed multipoles onto PME grid.
     *
     * @param particleData vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void spreadFixedMultipolesOntoGrid(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Perform reciprocal convolution.
     *
     */
    void performAmoebaReciprocalConvolution();

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     *
     */
    void computeFixedPotentialFromGrid(void);

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     *
     */
    void computeInducedPotentialFromGrid();

    /**
     * Calculate reciprocal space energy and force due to fixed multipoles.
     *
     * @param particleData    vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces          upon return updated vector of forces
     * @param torques         upon return updated vector of torques
     *
     * @return energy
     */
    double computeReciprocalSpaceFixedMultipoleForceAndEnergy(const std::vector<MultipoleParticleData>& particleData,
                                                              std::vector<OpenMM::Vec3>& forces, std::vector<OpenMM::Vec3>& torques) const;

    /**
     * Set reciprocal space fixed multipole fields.
     *
     */
    void recordFixedMultipoleField();

    /**
     * Compute the potential due to the reciprocal space PME calculation for induced dipoles.
     */
    void calculateReciprocalSpaceInducedDipoleField();

    /**
     * Calculate direct space field at particleI due to induced dipole at particle J and vice versa for
     * inducedDipole and inducedDipolePolar.
     *
     * @param particleI    positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ    positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    void calculateDirectInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                              const MultipoleParticleData& particleJ);

    /**
     * Initialize induced dipoles
     */
    void initializeInducedDipoles();

    /**
     * Spread induced dipoles onto grid.
     *
     * @param inputInducedDipole      induced dipole value
     */
    void spreadInducedDipolesOnGrid(const std::vector<OpenMM::Vec3>& inputInducedDipole);

    /**
     * Calculate induced dipole fields.
     *
     * @param particleData    vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param optOrder        the order of the extrapolated perturbation series being computed
     */
    void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData, int optOrder);

    /**
     * Set reciprocal space induced dipole fields.
     *
     * @param field       reciprocal space output induced dipole field value at each site
     *
     */
    void recordInducedDipoleField(std::vector<OpenMM::Vec3>& field);

    /**
     * The r = 0 limit erf_k(0) = 2^(k+1) alpha^(2k+1) / ((2k+1) sqrt(pi)) of the reciprocal-space radial
     * function of order k, for fully excluded pairs at the same position.
     */
    double coincidentEwaldLimit(int k) const;

    /**
     * Compute Pme self energy.
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     */
    double calculatePmeSelfEnergy(const std::vector<MultipoleParticleData>& particleData) const;

    /**
     * Compute the self torques.
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param torques                 vector of torques
     */
    void calculatePmeSelfTorque(const std::vector<MultipoleParticleData>& particleData, std::vector<OpenMM::Vec3>& torques) const;

    /**
     * Calculate direct space electrostatic interaction between particles I and K.
     *
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param r                 the distance between the two particles
     * @param force             the force to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueI           the torque to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueK           the torque to apply (in the quasi-internal frame) to particle K should be added to this
     */
    double calculateElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                         double r, OpenMM::Vec3& force, OpenMM::Vec3& torqueI, OpenMM::Vec3& torqueK) const;

    /**
     * Calculate electrostatic interactions involving induced dipoles on particles I and K.
     *
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param deltaR            the displacement between the two particles (in the lab frame)
     * @param r                 the distance between the two particles
     * @param force             the force to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueI           the torque to apply (in the quasi-internal frame) to particle I should be added to this
     * @param torqueK           the torque to apply (in the quasi-internal frame) to particle K should be added to this
     * @param labForce          the force to apply (in the lab frame) to particle I should be added to this
     */
    void calculateInducedDipolePairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                       OpenMM::Vec3 deltaR, double r, OpenMM::Vec3& force, OpenMM::Vec3& torqueI, OpenMM::Vec3& torqueK, OpenMM::Vec3& labForce) const;

    /**
     * Calculate reciprocal space force/torque for dipole interaction.
     *
     * @param particleData      vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces            vector of particle forces to be updated
     * @param torques           vector of particle torques to be updated
     */
     void computeReciprocalSpaceInducedDipoleForce(const std::vector<MultipoleParticleData>& particleData,
                                                   std::vector<OpenMM::Vec3>& forces, std::vector<OpenMM::Vec3>& torques) const;

    /**
     * Calculate the forces and energy.
     *
     * @param torques    output torques
     * @param forces     output forces
     * @return energy
     */
    double calculateInteractions(std::vector<OpenMM::Vec3>& torques,
                                 std::vector<OpenMM::Vec3>& forces);

};

} // namespace InteraxPlugin

#endif /*OPENMM_REFERENCEINTERAXMULTIPOLE_H_*/
