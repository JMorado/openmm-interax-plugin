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

#include "InteraxThreeBodyForceProxy.h"
#include "InteraxThreeBodyForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxThreeBodyForceProxy::InteraxThreeBodyForceProxy() : SerializationProxy("InteraxThreeBodyForce") {
}

void InteraxThreeBodyForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const InteraxThreeBodyForce& force = *reinterpret_cast<const InteraxThreeBodyForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setDoubleProperty("scaleFactor", force.getScaleFactor());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setStringProperty("alchemicalParameter", force.getAlchemicalParameterName());
    SerializationNode& molecules = node.createChildNode("Molecules");
    for (int i = 0; i < force.getNumMolecules(); i++) {
        vector<int> particles;
        double alpha, frequency;
        bool isAlchemical;
        force.getMoleculeParameters(i, particles, alpha, frequency, isAlchemical);
        SerializationNode& molecule = molecules.createChildNode("Molecule");
        molecule.setDoubleProperty("alpha", alpha).setDoubleProperty("frequency", frequency).setBoolProperty("alchemical", isAlchemical);
        for (int particle : particles)
            molecule.createChildNode("Particle").setIntProperty("index", particle);
    }
}

void* InteraxThreeBodyForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    InteraxThreeBodyForce* force = new InteraxThreeBodyForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setScaleFactor(node.getDoubleProperty("scaleFactor"));
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
        force->setAlchemicalParameterName(node.getStringProperty("alchemicalParameter"));
        const SerializationNode& molecules = node.getChildNode("Molecules");
        for (auto& molecule : molecules.getChildren()) {
            vector<int> particles;
            for (auto& particle : molecule.getChildren())
                particles.push_back(particle.getIntProperty("index"));
            force->addMolecule(particles, molecule.getDoubleProperty("alpha"), molecule.getDoubleProperty("frequency"), molecule.getBoolProperty("alchemical"));
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
