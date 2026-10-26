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

#include "InteraxThreeBodyForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <memory>
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

void compareForces(const InteraxThreeBodyForce& force, const InteraxThreeBodyForce& force2) {
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getScaleFactor(), force2.getScaleFactor());
    ASSERT_EQUAL(force.getDampingMethod(), force2.getDampingMethod());
    ASSERT_EQUAL(force.getDampingScale(), force2.getDampingScale());
    ASSERT_EQUAL(force.getCentroidDampingRadius(), force2.getCentroidDampingRadius());
    ASSERT_EQUAL(force.usesPeriodicBoundaryConditions(), force2.usesPeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getAlchemicalParameterName(), force2.getAlchemicalParameterName());
    ASSERT_EQUAL(force.getNumSites(), force2.getNumSites());
    for (int i = 0; i < force.getNumSites(); i++) {
        vector<int> particles1, particles2;
        double alpha1, frequency1, width1, alpha2, frequency2, width2;
        int molecule1, molecule2;
        bool alchemical1, alchemical2;
        force.getSiteParameters(i, particles1, alpha1, frequency1, width1, molecule1, alchemical1);
        force2.getSiteParameters(i, particles2, alpha2, frequency2, width2, molecule2, alchemical2);
        ASSERT_EQUAL_CONTAINERS(particles1, particles2);
        ASSERT_EQUAL(alpha1, alpha2);
        ASSERT_EQUAL(frequency1, frequency2);
        ASSERT_EQUAL(width1, width2);
        ASSERT_EQUAL(molecule1, molecule2);
        ASSERT_EQUAL(alchemical1, alchemical2);
    }
}

void testSerialization(double centroidDampingRadius) {
    // Create a Force.

    InteraxThreeBodyForce force;
    force.setForceGroup(3);
    force.setName("custom name");
    force.setCutoffDistance(1.1);
    force.setScaleFactor(0.8);
    force.setDampingMethod(InteraxThreeBodyForce::TangToennies);
    force.setDampingScale(0.7);
    force.setCentroidDampingRadius(centroidDampingRadius);
    force.setUsesPeriodicBoundaryConditions(true);
    force.setAlchemicalParameterName("lambda_test");
    force.addSite({0, 1, 2}, 0.004087456746801944, 911.0681141195369, 0.03, 0, false);
    force.addSite({3}, 0.0016246535729804365, 1262.664320774599, 0.025, -1, true);
    force.addSite({4, 6}, 0.008817709933796472, 747.5256543931184, 0.035, 2, false);

    // Serialize and then deserialize it, and compare the two forces to see if they are identical.

    stringstream buffer;
    XmlSerializer::serialize<InteraxThreeBodyForce>(&force, "Force", buffer);
    unique_ptr<InteraxThreeBodyForce> copy(XmlSerializer::deserialize<InteraxThreeBodyForce>(buffer));
    compareForces(force, *copy);

    // Without centroid damping version 2 is written (readable by older builds), with it version 3.

    string expectedVersion = (centroidDampingRadius == 0 ? "version=\"2\"" : "version=\"3\"");
    ASSERT(buffer.str().find(expectedVersion) != string::npos);

    // Serializing the copy gives the same XML.

    stringstream buffer2;
    XmlSerializer::serialize<InteraxThreeBodyForce>(copy.get(), "Force", buffer2);
    ASSERT_EQUAL(buffer.str(), buffer2.str());
}

void testVersion1() {
    // Version 1 had one site per molecule (the centroid of its particles), without damping.

    string xml = "<?xml version=\"1.0\" ?>\n"
        "<Force alchemicalParameter=\"lambda_sterics\" cutoff=\"1.2\" forceGroup=\"3\" name=\"CustomCentroidBondForce\" scaleFactor=\"1.5\" "
        "type=\"InteraxThreeBodyForce\" usesPeriodic=\"1\" version=\"1\">\n"
        "<Molecules>\n"
        "<Molecule alchemical=\"0\" alpha=\".004087456746801944\" frequency=\"911.0681141195369\"><Particle index=\"0\"/><Particle index=\"1\"/></Molecule>\n"
        "<Molecule alchemical=\"1\" alpha=\".0016246535729804365\" frequency=\"1262.664320774599\"><Particle index=\"2\"/></Molecule>\n"
        "</Molecules>\n"
        "</Force>\n";
    stringstream buffer(xml);
    unique_ptr<InteraxThreeBodyForce> force(XmlSerializer::deserialize<InteraxThreeBodyForce>(buffer));
    InteraxThreeBodyForce expected;
    expected.setForceGroup(3);
    expected.setName("CustomCentroidBondForce");
    expected.setCutoffDistance(1.2);
    expected.setScaleFactor(1.5);
    expected.setUsesPeriodicBoundaryConditions(true);
    expected.addSite({0, 1}, 0.004087456746801944, 911.0681141195369, 0.0, -1, false);
    expected.addSite({2}, 0.0016246535729804365, 1262.664320774599, 0.0, -1, true);
    compareForces(expected, *force);
}

void testUnsupportedVersion() {
    InteraxThreeBodyForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxThreeBodyForce>(&force, "Force", buffer);
    string xml = buffer.str();
    xml.replace(xml.find("version=\"2\""), 11, "version=\"4\"");
    stringstream modified(xml);
    bool failed = false;
    try {
        delete XmlSerializer::deserialize<InteraxThreeBodyForce>(modified);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

int main() {
    try {
        testSerialization(0.0);
        testSerialization(0.25);
        testVersion1();
        testUnsupportedVersion();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
