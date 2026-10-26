/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org) and its example plugin.            *
 *                                                                            *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Joao Morado                                                       *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "InteraxMultipoleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <memory>
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

void testSerialization() {
    // Create a Force.

    InteraxMultipoleForce force;
    force.setForceGroup(3);
    force.setName("custom name");
    force.setNonbondedMethod(InteraxMultipoleForce::PME);
    force.setCutoffDistance(0.9);
    force.setEwaldErrorTolerance(2e-5);
    force.setPMEParameters(3.5, 24, 28, 30);
    force.setExtrapolationCoefficients({0.0, 0.1, 0.4, 0.5});
    force.setCoreDamping(true);
    force.setCoreDampingScale(0.7);
    force.setMutualDampingScale(0.8);
    force.setUseOwnKernel(true);
    force.setAlchemicalParameterName("lambda_test");
    double qs = 0.0001;
    force.addParticle(-0.38, {0.0, 0.001, 0.003}, {qs*0.7, qs*0.1, qs*0.2, qs*0.1, -qs*0.6, qs*0.3, qs*0.2, qs*0.3, -qs*0.1}, 6.0, 47.075, 0.000795,
            InteraxMultipoleForce::Bisector, 1, 2, -1, true);
    force.addParticle(0.19, {0.0, 0.0, -0.01}, {qs*0.04, 0.0, 0.0, 0.0, qs*0.02, 0.0, 0.0, 0.0, -qs*0.06}, 1.0, 47.909, 0.000341,
            InteraxMultipoleForce::ZThenX, 0, 2, -1, false);
    force.addParticle(0.19, {0.0, 0.0, -0.01}, {qs*0.04, 0.0, 0.0, 0.0, qs*0.02, 0.0, 0.0, 0.0, -qs*0.06}, 1.0, 47.909, 0.0,
            InteraxMultipoleForce::ThreeFold, 0, 1, 3, false);
    force.addParticle(0.0, {0.0, 0.0, 0.0}, vector<double>(9, 0.0), 0.0, 40.0, 0.001, InteraxMultipoleForce::ZOnly, 0, -1, -1, false);
    force.addException(0, 1, 0.0, 0.0, 0.2);
    force.addException(0, 2, 0.1, 0.3, 0.4);
    force.addException(3, 1, 1.0, 0.5, 0.0);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<InteraxMultipoleForce>(&force, "Force", buffer);
    unique_ptr<InteraxMultipoleForce> copy(XmlSerializer::deserialize<InteraxMultipoleForce>(buffer));

    // Compare the two forces to see if they are identical.

    InteraxMultipoleForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), force2.getEwaldErrorTolerance());
    double alpha1, alpha2;
    int nx1, ny1, nz1, nx2, ny2, nz2;
    force.getPMEParameters(alpha1, nx1, ny1, nz1);
    force2.getPMEParameters(alpha2, nx2, ny2, nz2);
    ASSERT_EQUAL(alpha1, alpha2);
    ASSERT_EQUAL(nx1, nx2);
    ASSERT_EQUAL(ny1, ny2);
    ASSERT_EQUAL(nz1, nz2);
    ASSERT_EQUAL_CONTAINERS(force.getExtrapolationCoefficients(), force2.getExtrapolationCoefficients());
    ASSERT_EQUAL(force.getCoreDamping(), force2.getCoreDamping());
    ASSERT_EQUAL(force.getCoreDampingScale(), force2.getCoreDampingScale());
    ASSERT_EQUAL(force.getMutualDampingScale(), force2.getMutualDampingScale());
    ASSERT_EQUAL(force.getUseOwnKernel(), force2.getUseOwnKernel());
    ASSERT_EQUAL(force.getAlchemicalParameterName(), force2.getAlchemicalParameterName());
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge1, coreCharge1, alpha1, polarizability1, charge2, coreCharge2, alpha2, polarizability2;
        int axisType1, atomZ1, atomX1, atomY1, axisType2, atomZ2, atomX2, atomY2;
        bool alchemical1, alchemical2;
        vector<double> dipole1, quadrupole1, dipole2, quadrupole2;
        force.getParticleParameters(i, charge1, dipole1, quadrupole1, coreCharge1, alpha1, polarizability1, axisType1, atomZ1, atomX1, atomY1, alchemical1);
        force2.getParticleParameters(i, charge2, dipole2, quadrupole2, coreCharge2, alpha2, polarizability2, axisType2, atomZ2, atomX2, atomY2, alchemical2);
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL_CONTAINERS(dipole1, dipole2);
        ASSERT_EQUAL_CONTAINERS(quadrupole1, quadrupole2);
        ASSERT_EQUAL(coreCharge1, coreCharge2);
        ASSERT_EQUAL(alpha1, alpha2);
        ASSERT_EQUAL(polarizability1, polarizability2);
        ASSERT_EQUAL(axisType1, axisType2);
        ASSERT_EQUAL(atomZ1, atomZ2);
        ASSERT_EQUAL(atomX1, atomX2);
        ASSERT_EQUAL(atomY1, atomY2);
        ASSERT_EQUAL(alchemical1, alchemical2);
    }
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int a1, a2, b1, b2;
        double mm1, dm1, dd1, mm2, dm2, dd2;
        force.getExceptionParameters(i, a1, b1, mm1, dm1, dd1);
        force2.getExceptionParameters(i, a2, b2, mm2, dm2, dd2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(mm1, mm2);
        ASSERT_EQUAL(dm1, dm2);
        ASSERT_EQUAL(dd1, dd2);
    }

    // The deserialized force still knows its exceptions: adding one again for the same pair fails.

    bool failed = false;
    try {
        force2.addException(1, 3, 0.0, 0.0, 0.0);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);

    // Serializing the copy gives the same XML.

    stringstream buffer2;
    XmlSerializer::serialize<InteraxMultipoleForce>(copy.get(), "Force", buffer2);
    stringstream buffer3;
    XmlSerializer::serialize<InteraxMultipoleForce>(&force, "Force", buffer3);
    ASSERT_EQUAL(buffer3.str(), buffer2.str());
}

void testDefaults() {
    // A force with default settings keeps them.

    InteraxMultipoleForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxMultipoleForce>(&force, "Force", buffer);
    unique_ptr<InteraxMultipoleForce> copy(XmlSerializer::deserialize<InteraxMultipoleForce>(buffer));
    ASSERT_EQUAL(InteraxMultipoleForce::NoCutoff, copy->getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), copy->getCutoffDistance());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), copy->getEwaldErrorTolerance());
    ASSERT_EQUAL_CONTAINERS(force.getExtrapolationCoefficients(), copy->getExtrapolationCoefficients());
    ASSERT_EQUAL(false, copy->getCoreDamping());
    ASSERT_EQUAL(1.0, copy->getCoreDampingScale());
    ASSERT_EQUAL(1.0, copy->getMutualDampingScale());
    ASSERT_EQUAL(false, copy->getUseOwnKernel());
    ASSERT_EQUAL(string("lambda_electrostatics"), copy->getAlchemicalParameterName());
}

void testUnsupportedVersion() {
    InteraxMultipoleForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxMultipoleForce>(&force, "Force", buffer);
    string xml = buffer.str();
    xml.replace(xml.find("version=\"1\""), 11, "version=\"2\"");
    stringstream modified(xml);
    bool failed = false;
    try {
        delete XmlSerializer::deserialize<InteraxMultipoleForce>(modified);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

int main() {
    try {
        testSerialization();
        testDefaults();
        testUnsupportedVersion();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
