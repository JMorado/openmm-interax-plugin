#ifndef OPENMM_INTERAXPAIRTERMS_H_
#define OPENMM_INTERAXPAIRTERMS_H_

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

/**
 * The pair energies of the InteraxFF dispersion, exchange and short-range forces and their radial derivatives,
 * shared by the Reference and CPU platforms and by the long range correction. See docs/SPEC.md for the formulas.
 *
 * Every function is a template over the value type: double (Reference), float, or the OpenMM SIMD vectors fvec4
 * and fvec8 (CPU, one lane per pair), which live in the global namespace. Branches are written with blendIf(), which picks between two values for a
 * scalar condition and blends the lanes for a vector mask; the vector versions evaluate both alternatives.
 *
 * @private
 */

#include "openmm/internal/vectorize.h"
#ifdef __AVX__
#include "openmm/internal/vectorizeAvx.h"
#endif
#include <cmath>

namespace InteraxPlugin {

/** The scalar type of a value type: float for the single precision types, double for double. */
template <class T> struct ScalarOf { typedef float type; };
template <> struct ScalarOf<double> { typedef double type; };

/** Choose a where the condition holds and b elsewhere. */
inline double blendIf(bool condition, double a, double b) {
    return condition ? a : b;
}
inline float blendIf(bool condition, float a, float b) {
    return condition ? a : b;
}
inline fvec4 blendIf(fvec4 mask, fvec4 a, fvec4 b) {
    return ::blend(b, a, mask);
}
#ifdef __AVX__
inline fvec8 blendIf(fvec8 mask, fvec8 a, fvec8 b) {
    return ::blend(b, a, mask);
}
#endif

} // namespace InteraxPlugin

#ifdef __AVX__
/** OpenMM's AVX vectors have no exponential: evaluate it on each half with the SSE version. */
static inline fvec8 exp(fvec8 v) {
    return fvec8(_mm256_insertf128_ps(_mm256_castps128_ps256(exp(v.lowerVec()).val), exp(v.upperVec()).val, 1));
}
#endif

namespace InteraxPlugin {

namespace InteraxPairTerms {

/**
 * Coefficients of the Taylor expansion of the Slater overlap integral about the smaller width l, in t = |b - a|/l
 * and x = r/l (interax.geometry.OVERLAP_TAYLOR_COEFFS, divided by their denominators):
 *
 *   S     = exp(-x)/(pi l^3) sum_n t^n sum_k VALUE[n][k] x^k
 *   dS/dr = exp(-x)/(pi l^4) sum_n t^n sum_k DERIV[n][k] x^k,   DERIV[n][k] = (k+1) VALUE[n][k+1] - VALUE[n][k]
 */
static const int TAYLOR_ORDER = 6;
static const int TAYLOR_TERMS = 9;
static const double OVERLAP_TAYLOR_VALUE[TAYLOR_ORDER+1][TAYLOR_TERMS] = {
    {0.015625, 0.015625, 0.005208333333333333},
    {-0.0234375, -0.0234375, -0.005208333333333333, 0.0026041666666666665},
    {0.0234375, 0.0234375, 0.0013020833333333333, -0.006510416666666667, 0.00078125000000000004},
    {-0.01953125, -0.01953125, 0.00390625, 0.010416666666666666, -0.0028645833333333331, 0.00017361111111111112},
    {0.0146484375, 0.0146484375, -0.0087890625, -0.013671875, 0.0064453124999999997, -0.00082465277777777778, 3.1001984126984125e-05},
    {-0.01025390625, -0.01025390625, 0.0126953125, 0.01611328125, -0.011523437500000001, 0.0023220486111111111, -0.00017981150793650795, 4.6502976190476195e-06},
    {0.0068359375, 0.0068359375, -0.015543619791666666, -0.017822265625, 0.018017578124999999, -0.0050564236111111114, 0.0006029885912698413, -3.1777033730158732e-05, 6.0281635802469135e-07}
};
static const double OVERLAP_TAYLOR_DERIV[TAYLOR_ORDER+1][TAYLOR_TERMS] = {
    {0, -0.005208333333333333, -0.005208333333333333},
    {0, 0.013020833333333334, 0.013020833333333334, -0.0026041666666666665},
    {0, -0.020833333333333332, -0.020833333333333332, 0.0096354166666666671, -0.00078125000000000004},
    {0, 0.02734375, 0.02734375, -0.021874999999999999, 0.0037326388888888891, -0.00017361111111111112},
    {0, -0.0322265625, -0.0322265625, 0.039453124999999999, -0.01056857638888889, 0.0010106646825396826, -3.1001984126984125e-05},
    {0, 0.03564453125, 0.03564453125, -0.062207031250000003, 0.023133680555555557, -0.0034009176587301588, 0.00021236359126984127, -4.6502976190476195e-06},
    {0, -0.037923177083333336, -0.037923177083333336, 0.089892578124999997, -0.043299696180555555, 0.0086743551587301591, -0.00082542782738095238, 3.6599564594356259e-05, -6.0281635802469135e-07}
};

/**
 * Where the Taylor expansion replaces the closed form: t < TaylorMaxT and t*x < TaylorMaxTX. The closed form loses
 * relative precision ~ eps/t^2 to cancellation, and the order-6 expansion degrades as (t*x)^7, so the switch depends
 * on the precision: with these values the integral is accurate to ~1e-6 relative in single precision and ~1e-10 in
 * double precision over the whole range of distances (docs/SPEC.md).
 */
template <class Scalar> struct OverlapSwitch {
    static Scalar maxT() { return 0.2f; }
    static Scalar maxTX() { return 1.6f; }
};
template <> struct OverlapSwitch<double> {
    static double maxT() { return 0.02; }
    static double maxTX() { return 0.3; }
};

/**
 * Compute the overlap integral S(r; a, b) of two normalized Slater densities exp(-r/s)/(8 pi s^3) of widths a and b
 * at distance r, and its derivative with respect to r.
 */
template <class T>
inline void slaterOverlap(T r, T a, T b, T& overlap, T& dOverlapdR) {
    using std::exp;
    using std::min;
    using std::max;
    typedef typename ScalarOf<T>::type Scalar;
    const Scalar pi = Scalar(M_PI);

    // Closed form (MEDFF SI Eq. 32): S = F/(8 pi r).

    T aa = a*a, bb = b*b;
    T d = bb-aa;
    T invD2 = 1/(d*d);
    T expA = exp(-r/a), expB = exp(-r/b);
    T c = Scalar(4)*aa*bb*invD2/d;
    T f = c*(expA-expB) + r*invD2*(a*expA+b*expB);
    T dfdr = c*(expB/b-expA/a) + invD2*(a*expA+b*expB-r*(expA+expB));
    T invR = 1/r;
    T closedOverlap = f*invR/(Scalar(8)*pi);
    T closedDeriv = (r*dfdr-f)*invR*invR/(Scalar(8)*pi);

    // Taylor expansion about the smaller width, for nearly equal widths where the closed form cancels.

    T l = min(a, b);
    T invL = 1/l;
    T t = (max(a, b)-l)*invL;
    T x = r*invL;
    T sumValue = Scalar(0), sumDeriv = Scalar(0);
    for (int n = TAYLOR_ORDER; n >= 0; n--) {
        T value = Scalar(0), deriv = Scalar(0);
        for (int k = TAYLOR_TERMS-1; k >= 0; k--) {
            value = value*x + Scalar(OVERLAP_TAYLOR_VALUE[n][k]);
            deriv = deriv*x + Scalar(OVERLAP_TAYLOR_DERIV[n][k]);
        }
        sumValue = sumValue*t + value;
        sumDeriv = sumDeriv*t + deriv;
    }
    T prefactor = exp(-x)*invL*invL*invL/pi;
    T taylorOverlap = prefactor*sumValue;
    T taylorDeriv = prefactor*invL*sumDeriv;

    auto useTaylor = (t < OverlapSwitch<Scalar>::maxT()) & (t*x < OverlapSwitch<Scalar>::maxTX());
    overlap = blendIf(useTaylor, taylorOverlap, closedOverlap);
    dOverlapdR = blendIf(useTaylor, taylorDeriv, closedDeriv);
}

/**
 * Compute the Tang-Toennies damping functions f_n(x) = 1 - exp(-x) sum_{k=0}^{n} x^k/k! for n = 6, 8, 10 and their
 * derivatives f_n'(x) = exp(-x) x^n/n!. For x < (n+1)/2 the closed form cancels, and f_n is evaluated instead from
 * the series exp(-x) sum_{k>n} x^k/k!, whose terms then fall off faster than 2^-k.
 */
template <class T>
inline void tangToenniesDamping(T x, T damping[3], T dDampingdX[3]) {
    using std::exp;
    typedef typename ScalarOf<T>::type Scalar;
    const int maxTerm = 40;
    T expX = exp(-x);
    T term = Scalar(1);          // x^k/k!
    T partialSum = Scalar(1);    // sum_{j<=k} x^j/j!
    T terms[11], partialSums[11];
    terms[0] = term;
    partialSums[0] = partialSum;
    for (int k = 1; k <= 10; k++) {
        term = term*x*Scalar(1.0/k);
        partialSum = partialSum+term;
        terms[k] = term;
        partialSums[k] = partialSum;
    }
    T tail = Scalar(0);          // sum_{10<k<=maxTerm} x^k/k!
    for (int k = 11; k <= maxTerm; k++) {
        term = term*x*Scalar(1.0/k);
        tail = tail+term;
    }
    for (int i = 0; i < 3; i++) {
        int n = 6+2*i;
        T series = tail;
        for (int k = n+1; k <= 10; k++)
            series = series+terms[k];
        damping[i] = blendIf(x < Scalar(0.5*(n+1)), expX*series, 1-expX*partialSums[n]);
        dDampingdX[i] = expX*terms[n];
    }
}

/**
 * Compute the dispersion energy of a pair and its derivative with respect to r.
 *
 * @param r                  the distance
 * @param m1i, m2i, m3i      the XDM moments of the first particle
 * @param alphai, widthi     the dispersion polarizability and Slater width of the first particle
 * @param m1j, m2j, m3j      the XDM moments of the second particle
 * @param alphaj, widthj     the dispersion polarizability and Slater width of the second particle
 * @param dampingScale       the scale beta of the damping argument
 * @param[out] energy        the energy of the pair
 * @param[out] dEdR          the derivative of the energy with respect to r
 */
template <class T>
inline void dispersionPair(T r, T m1i, T m2i, T m3i, T alphai, T widthi, T m1j, T m2j, T m3j, T alphaj, T widthj,
        T dampingScale, T& energy, T& dEdR) {
    typedef typename ScalarOf<T>::type Scalar;
    T prefactor = alphai*alphaj/(alphai*m1j+alphaj*m1i);
    T c[3];
    c[0] = prefactor*m1i*m1j;
    c[1] = Scalar(1.5)*prefactor*(m1i*m2j+m2i*m1j);
    c[2] = prefactor*(Scalar(2)*(m1i*m3j+m3i*m1j)+Scalar(4.2)*m2i*m2j);
    T dxdr = Scalar(2)*dampingScale/(widthi+widthj);
    T damping[3], dDampingdX[3];
    tangToenniesDamping(dxdr*r, damping, dDampingdX);
    T invR = 1/r;
    T invR2 = invR*invR;
    T invRn = invR2*invR2*invR2;    // r^-6
    energy = Scalar(0);
    dEdR = Scalar(0);
    for (int i = 0; i < 3; i++) {
        Scalar n = Scalar(6+2*i);
        energy = energy - damping[i]*c[i]*invRn;
        dEdR = dEdR - c[i]*invRn*(dDampingdX[i]*dxdr - n*damping[i]*invR);
        invRn = invRn*invR2;
    }
}

/**
 * Compute the exchange repulsion energy A_i A_j S(r; s_i, s_j) of a pair and its derivative with respect to r.
 */
template <class T>
inline void exchangePair(T r, T amplitudei, T widthi, T amplitudej, T widthj, T& energy, T& dEdR) {
    T overlap, dOverlapdR;
    slaterOverlap(r, widthi, widthj, overlap, dOverlapdR);
    T amplitude = amplitudei*amplitudej;
    energy = amplitude*overlap;
    dEdR = amplitude*dOverlapdR;
}

/**
 * Compute the short-range correction energy (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 of a pair, with
 * St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j), and its derivative with respect to r.
 */
template <class T>
inline void shortRangePair(T r, T ki, T li, T widthi, T kj, T lj, T widthj, T& energy, T& dEdR) {
    typedef typename ScalarOf<T>::type Scalar;
    T overlap, dOverlapdR;
    slaterOverlap(r, widthi, widthj, overlap, dOverlapdR);
    T widthSum = widthi+widthj;
    T normalization = Scalar(8*M_PI)*widthSum*widthSum*widthSum;
    T normalizedOverlap = normalization*overlap;
    T k = Scalar(0.5)*(ki+kj);
    T l = Scalar(0.5)*(li+lj);
    energy = (k+l*normalizedOverlap)*normalizedOverlap;
    dEdR = (k+Scalar(2)*l*normalizedOverlap)*normalization*dOverlapdR;
}

} // namespace InteraxPairTerms

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXPAIRTERMS_H_*/
