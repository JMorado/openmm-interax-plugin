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

#include "InteraxDispersionForce.h"
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

    InteraxDispersionForce force;
    force.setForceGroup(3);
    force.setName("custom name");
    force.setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
    force.setCutoffDistance(1.1);
    force.setUseSwitchingFunction(true);
    force.setSwitchingDistance(0.95);
    force.setAlchemicalParameterName("lambda_test");
    force.setSoftcoreRadius(0.12);
    force.setDampingScale(0.7155241370201111);
    force.setUseLongRangeCorrection(true);
    force.addParticle(2.0412491330659432, 0.07991696226413432, 0.003917538949072952, 0.0020393931939312773, 0.026844933830306183);
    force.addParticle(0.47385034770767653, 0.0062664158577877644, 0.00015411387998271836, 0.00023498508736915282, 0.020351567997950105, true);
    force.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, false);
    force.addExclusion(0, 1);
    force.addExclusion(2, 1);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<InteraxDispersionForce>(&force, "Force", buffer);
    unique_ptr<InteraxDispersionForce> copy(XmlSerializer::deserialize<InteraxDispersionForce>(buffer));

    // Compare the two forces to see if they are identical.

    InteraxDispersionForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getUseSwitchingFunction(), force2.getUseSwitchingFunction());
    ASSERT_EQUAL(force.getSwitchingDistance(), force2.getSwitchingDistance());
    ASSERT_EQUAL(force.getAlchemicalParameterName(), force2.getAlchemicalParameterName());
    ASSERT_EQUAL(force.getSoftcoreRadius(), force2.getSoftcoreRadius());
    ASSERT_EQUAL(force.getDampingScale(), force2.getDampingScale());
    ASSERT_EQUAL(force.getUseLongRangeCorrection(), force2.getUseLongRangeCorrection());
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double m1a, m2a, m3a, alphaa, widtha, m1b, m2b, m3b, alphab, widthb;
        bool alchemicala, alchemicalb;
        force.getParticleParameters(i, m1a, m2a, m3a, alphaa, widtha, alchemicala);
        force2.getParticleParameters(i, m1b, m2b, m3b, alphab, widthb, alchemicalb);
        ASSERT_EQUAL(m1a, m1b);
        ASSERT_EQUAL(m2a, m2b);
        ASSERT_EQUAL(m3a, m3b);
        ASSERT_EQUAL(alphaa, alphab);
        ASSERT_EQUAL(widtha, widthb);
        ASSERT_EQUAL(alchemicala, alchemicalb);
    }
    ASSERT_EQUAL(force.getNumExclusions(), force2.getNumExclusions());
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int a1, a2, b1, b2;
        force.getExclusionParticles(i, a1, b1);
        force2.getExclusionParticles(i, a2, b2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
    }

    // Serializing the copy gives the same XML.

    stringstream buffer2;
    XmlSerializer::serialize<InteraxDispersionForce>(&force2, "Force", buffer2);
    ASSERT_EQUAL(buffer.str(), buffer2.str());
}

void testDefaults() {
    // A force with default settings keeps them.

    InteraxDispersionForce force;
    force.addParticle(1.0, 0.1, 0.01, 0.001, 0.02);
    stringstream buffer;
    XmlSerializer::serialize<InteraxDispersionForce>(&force, "Force", buffer);
    unique_ptr<InteraxDispersionForce> copy(XmlSerializer::deserialize<InteraxDispersionForce>(buffer));
    ASSERT_EQUAL(InteraxPairForce::NoCutoff, copy->getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), copy->getCutoffDistance());
    ASSERT_EQUAL(false, copy->getUseSwitchingFunction());
    ASSERT_EQUAL(string("lambda_sterics"), copy->getAlchemicalParameterName());
    ASSERT_EQUAL(0.0, copy->getSoftcoreRadius());
    ASSERT_EQUAL(1.0, copy->getDampingScale());
    ASSERT_EQUAL(false, copy->getUseLongRangeCorrection());
}

void testUnsupportedVersion() {
    InteraxDispersionForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxDispersionForce>(&force, "Force", buffer);
    string xml = buffer.str();
    xml.replace(xml.find("version=\"1\""), 11, "version=\"2\"");
    stringstream modified(xml);
    bool failed = false;
    try {
        delete XmlSerializer::deserialize<InteraxDispersionForce>(modified);
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
