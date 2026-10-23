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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/InteraxPairForceImpl.h"
#include "internal/InteraxPairTerms.h"
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

namespace {

/**
 * Compute the nodes and weights of the n-point Gauss-Legendre quadrature rule on the interval (0, 1).
 */
void computeGaussLegendreRule(int n, vector<double>& nodes, vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);
    for (int i = 0; i < n; i++) {
        // Newton iteration for the i'th root of P_n on (-1, 1), starting from the Chebyshev estimate.

        double x = cos(M_PI*(i+0.75)/(n+0.5));
        double derivative = 0;
        for (int iteration = 0; iteration < 100; iteration++) {
            double p0 = 1, p1 = x;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2*k-1)*x*p1-(k-1)*p0)/k;
                p0 = p1;
                p1 = p2;
            }
            derivative = n*(x*p1-p0)/(x*x-1);
            double step = p1/derivative;
            x -= step;
            if (fabs(step) < 1e-15)
                break;
        }
        nodes[i] = 0.5*(1-x);
        weights[i] = 1/((1-x*x)*derivative*derivative);
    }
}

typedef tuple<double, double, double, double, double, bool> ParticleClass;

double pairEnergy(double r, const ParticleClass& class1, const ParticleClass& class2, double dampingScale) {
    double energy, dEdR;
    InteraxPairTerms::dispersionPair(r, get<0>(class1), get<1>(class1), get<2>(class1), get<3>(class1), get<4>(class1),
            get<0>(class2), get<1>(class2), get<2>(class2), get<3>(class2), get<4>(class2), dampingScale, energy, dEdR);
    return energy;
}

} // namespace

void InteraxDispersionForceImpl::calcLongRangeCorrection(const InteraxDispersionForce& force, double& coefficient, double& alchemicalCoefficient) {
    coefficient = 0.0;
    alchemicalCoefficient = 0.0;
    if (force.getNonbondedMethod() != InteraxPairForce::CutoffPeriodic || !force.getUseLongRangeCorrection())
        return;

    // Group the particles into classes with identical parameters.

    map<ParticleClass, int> classCounts;
    for (int i = 0; i < force.getNumParticles(); i++) {
        double m1, m2, m3, alpha, width;
        bool isAlchemical;
        force.getParticleParameters(i, m1, m2, m3, alpha, width, isAlchemical);
        classCounts[make_tuple(m1, m2, m3, alpha, width, isAlchemical)]++;
    }
    vector<pair<ParticleClass, int> > classes(classCounts.begin(), classCounts.end());

    // For each pair of classes, integrate r^2 E(r) from the cutoff to infinity (with the change of variables
    // u = r_cutoff/r) and r^2 (1 - S(r)) E(r) over the switching interval.

    const int numPoints = 32;
    vector<double> nodes, weights;
    computeGaussLegendreRule(numPoints, nodes, weights);
    double cutoff = force.getCutoffDistance();
    bool useSwitch = force.getUseSwitchingFunction();
    double switchDistance = (useSwitch ? force.getSwitchingDistance() : cutoff);
    double switchWidth = cutoff-switchDistance;
    double dampingScale = force.getDampingScale();
    double sum = 0.0, alchemicalSum = 0.0;
    for (int i = 0; i < classes.size(); i++) {
        for (int j = i; j < classes.size(); j++) {
            const ParticleClass& class1 = classes[i].first;
            const ParticleClass& class2 = classes[j].first;
            if (get<3>(class1)*get<3>(class2) == 0)
                continue;
            double integral = 0.0;
            for (int k = 0; k < numPoints; k++) {
                double u = nodes[k];
                integral += weights[k]*pairEnergy(cutoff/u, class1, class2, dampingScale)*cutoff*cutoff*cutoff/(u*u*u*u);
                if (useSwitch) {
                    double x = nodes[k];
                    double r = switchDistance+x*switchWidth;
                    double oneMinusSwitch = x*x*x*(10+x*(-15+x*6));
                    integral += weights[k]*switchWidth*oneMinusSwitch*pairEnergy(r, class1, class2, dampingScale)*r*r;
                }
            }
            double count1 = classes[i].second, count2 = classes[j].second;
            double count = (i == j ? 0.5*count1*(count1+1) : count1*count2);
            // Every pair with an alchemical particle, those within the alchemical molecules included, is scaled:
            // the correction is then that of the full system at lambda = 1 and that of the rest at lambda = 0.
            if (get<5>(class1) || get<5>(class2))
                alchemicalSum += count*integral;
            else
                sum += count*integral;
        }
    }

    // Normalize as CustomNonbondedForce does: 2 pi N^2 times the average over the N(N+1)/2 pairs.

    double numParticles = force.getNumParticles();
    double normalization = 2*M_PI*numParticles*numParticles/(0.5*numParticles*(numParticles+1));
    coefficient = normalization*sum;
    alchemicalCoefficient = normalization*alchemicalSum;
}
