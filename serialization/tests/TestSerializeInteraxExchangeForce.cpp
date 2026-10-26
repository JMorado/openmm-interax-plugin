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

#include "InteraxExchangeForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <memory>
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

void testSerialization(InteraxExchangeForce::OverlapMethod method) {
    // Create a Force.

    InteraxExchangeForce force;
    force.setForceGroup(3);
    force.setName("custom name");
    force.setNonbondedMethod(InteraxPairForce::CutoffNonPeriodic);
    force.setCutoffDistance(1.1);
    force.setUseSwitchingFunction(true);
    force.setSwitchingDistance(0.95);
    force.setAlchemicalParameterName("lambda_test");
    force.setSoftcoreRadius(0.12);
    force.setOverlapMethod(method);
    force.addParticle(10.326424719450545, 0.0248302193400194);
    force.addParticle(1.0254403205662432, 0.019763296864870033, true);
    force.addParticle(0.0, 0.0, false);
    force.addExclusion(0, 1);
    force.addExclusion(2, 1);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<InteraxExchangeForce>(&force, "Force", buffer);
    unique_ptr<InteraxExchangeForce> copy(XmlSerializer::deserialize<InteraxExchangeForce>(buffer));

    // Compare the two forces to see if they are identical.

    InteraxExchangeForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getUseSwitchingFunction(), force2.getUseSwitchingFunction());
    ASSERT_EQUAL(force.getSwitchingDistance(), force2.getSwitchingDistance());
    ASSERT_EQUAL(force.getAlchemicalParameterName(), force2.getAlchemicalParameterName());
    ASSERT_EQUAL(force.getSoftcoreRadius(), force2.getSoftcoreRadius());
    ASSERT_EQUAL(force.getOverlapMethod(), force2.getOverlapMethod());
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double amplitudea, widtha, amplitudeb, widthb;
        bool alchemicala, alchemicalb;
        force.getParticleParameters(i, amplitudea, widtha, alchemicala);
        force2.getParticleParameters(i, amplitudeb, widthb, alchemicalb);
        ASSERT_EQUAL(amplitudea, amplitudeb);
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
    XmlSerializer::serialize<InteraxExchangeForce>(&force2, "Force", buffer2);
    ASSERT_EQUAL(buffer.str(), buffer2.str());
}

void testDefaults() {
    // A force with default settings keeps them.

    InteraxExchangeForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxExchangeForce>(&force, "Force", buffer);
    unique_ptr<InteraxExchangeForce> copy(XmlSerializer::deserialize<InteraxExchangeForce>(buffer));
    ASSERT_EQUAL(InteraxPairForce::NoCutoff, copy->getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), copy->getCutoffDistance());
    ASSERT_EQUAL(false, copy->getUseSwitchingFunction());
    ASSERT_EQUAL(string("lambda_sterics"), copy->getAlchemicalParameterName());
    ASSERT_EQUAL(0.0, copy->getSoftcoreRadius());
    ASSERT_EQUAL(InteraxExchangeForce::Exact, copy->getOverlapMethod());
}

void testUnsupportedVersion() {
    InteraxExchangeForce force;
    stringstream buffer;
    XmlSerializer::serialize<InteraxExchangeForce>(&force, "Force", buffer);
    string xml = buffer.str();
    xml.replace(xml.find("version=\"1\""), 11, "version=\"2\"");
    stringstream modified(xml);
    bool failed = false;
    try {
        delete XmlSerializer::deserialize<InteraxExchangeForce>(modified);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

int main() {
    try {
        testSerialization(InteraxExchangeForce::Exact);
        testSerialization(InteraxExchangeForce::InteraxOrder2);
        testSerialization(InteraxExchangeForce::InteraxOrder6);
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
