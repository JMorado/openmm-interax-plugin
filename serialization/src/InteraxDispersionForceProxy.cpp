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

#include "InteraxDispersionForceProxy.h"
#include "InteraxDispersionForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxDispersionForceProxy::InteraxDispersionForceProxy() : SerializationProxy("InteraxDispersionForce") {
}

void InteraxDispersionForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const InteraxDispersionForce& force = *reinterpret_cast<const InteraxDispersionForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setBoolProperty("useSwitchingFunction", force.getUseSwitchingFunction());
    node.setDoubleProperty("switchingDistance", force.getSwitchingDistance());
    node.setStringProperty("alchemicalParameter", force.getAlchemicalParameterName());
    node.setDoubleProperty("dampingScale", force.getDampingScale());
    node.setBoolProperty("useLongRangeCorrection", force.getUseLongRangeCorrection());
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double m1, m2, m3, alpha, width;
        bool isAlchemical;
        force.getParticleParameters(i, m1, m2, m3, alpha, width, isAlchemical);
        particles.createChildNode("Particle").setDoubleProperty("m1", m1).setDoubleProperty("m2", m2).setDoubleProperty("m3", m3).setDoubleProperty("alpha", alpha).setDoubleProperty("width", width).setBoolProperty("alchemical", isAlchemical);
    }
    SerializationNode& exclusions = node.createChildNode("Exclusions");
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusions.createChildNode("Exclusion").setIntProperty("p1", particle1).setIntProperty("p2", particle2);
    }
}

void* InteraxDispersionForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    InteraxDispersionForce* force = new InteraxDispersionForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((InteraxPairForce::NonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setUseSwitchingFunction(node.getBoolProperty("useSwitchingFunction"));
        force->setSwitchingDistance(node.getDoubleProperty("switchingDistance"));
        force->setAlchemicalParameterName(node.getStringProperty("alchemicalParameter"));
        force->setDampingScale(node.getDoubleProperty("dampingScale"));
        force->setUseLongRangeCorrection(node.getBoolProperty("useLongRangeCorrection"));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (auto& particle : particles.getChildren())
            force->addParticle(particle.getDoubleProperty("m1"), particle.getDoubleProperty("m2"), particle.getDoubleProperty("m3"), particle.getDoubleProperty("alpha"), particle.getDoubleProperty("width"), particle.getBoolProperty("alchemical"));
        const SerializationNode& exclusions = node.getChildNode("Exclusions");
        for (auto& exclusion : exclusions.getChildren())
            force->addExclusion(exclusion.getIntProperty("p1"), exclusion.getIntProperty("p2"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
