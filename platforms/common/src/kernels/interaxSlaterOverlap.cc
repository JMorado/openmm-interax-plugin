// Overlap integral S(r; a, b) of two normalized Slater densities exp(-r/s)/(8 pi s^3) of widths OVERLAP_WIDTH1 and
// OVERLAP_WIDTH2 at distance r, and its derivative with respect to r (docs/SPEC.md). This snippet is inserted into
// the pair interaction code with the placeholders replaced. Nearly equal widths use the Taylor expansion of order
// TAYLOR_ORDER (InteraxPairTerms::SlaterTaylorScheme) about the width l, the smaller width or, for the InteraxFF
// schemes, that of the particle with the lower index: where the widths differ by less than TAYLOR_MAX_DELTA, or
// where t = (other width - l)/l and x = r/l satisfy |t| < TAYLOR_MAX_T and |t| x < TAYLOR_MAX_TX.
real OVERLAP, OVERLAP_DERIV;
{
#if TAYLOR_ABOUT_SMALLER_WIDTH
    const real overlapA = OVERLAP_WIDTH1, overlapB = OVERLAP_WIDTH2;
    const real overlapL = min(overlapA, overlapB);
#else
    const bool overlapFirst = (ATOM_INDEX[atom1] < ATOM_INDEX[atom2]);
    const real overlapA = (overlapFirst ? OVERLAP_WIDTH1 : OVERLAP_WIDTH2), overlapB = (overlapFirst ? OVERLAP_WIDTH2 : OVERLAP_WIDTH1);
    const real overlapL = overlapA;
#endif
    const real overlapInvL = 1/overlapL;
    const real overlapDelta = overlapA+overlapB-2*overlapL;
    const real overlapT = overlapDelta*overlapInvL;
    const real overlapX = r*overlapInvL;
    if (fabs(overlapDelta) < TAYLOR_MAX_DELTA || (fabs(overlapT) < TAYLOR_MAX_T && fabs(overlapT)*overlapX < TAYLOR_MAX_TX)) {
        // S = exp(-x)/(pi l^3) sum_n t^n sum_k VALUE[n][k] x^k,  dS/dr = exp(-x)/(pi l^4) sum_n t^n sum_k DERIV[n][k] x^k
        const real VALUE[7][9] = {
        {0.015625, 0.015625, 0.005208333333333333, 0, 0, 0, 0, 0, 0},
        {-0.0234375, -0.0234375, -0.005208333333333333, 0.0026041666666666665, 0, 0, 0, 0, 0},
        {0.0234375, 0.0234375, 0.0013020833333333333, -0.006510416666666667, 0.00078125000000000004, 0, 0, 0, 0},
        {-0.01953125, -0.01953125, 0.00390625, 0.010416666666666666, -0.0028645833333333331, 0.00017361111111111112, 0, 0, 0},
        {0.0146484375, 0.0146484375, -0.0087890625, -0.013671875, 0.0064453124999999997, -0.00082465277777777778, 3.1001984126984125e-05, 0, 0},
        {-0.01025390625, -0.01025390625, 0.0126953125, 0.01611328125, -0.011523437500000001, 0.0023220486111111111, -0.00017981150793650795, 4.6502976190476195e-06, 0},
        {0.0068359375, 0.0068359375, -0.015543619791666666, -0.017822265625, 0.018017578124999999, -0.0050564236111111114, 0.0006029885912698413, -3.1777033730158732e-05, 6.0281635802469135e-07}
        };
        const real DERIV[7][9] = {
        {0, -0.005208333333333333, -0.005208333333333333, 0, 0, 0, 0, 0, 0},
        {0, 0.013020833333333334, 0.013020833333333334, -0.0026041666666666665, 0, 0, 0, 0, 0},
        {0, -0.020833333333333332, -0.020833333333333332, 0.0096354166666666671, -0.00078125000000000004, 0, 0, 0, 0},
        {0, 0.02734375, 0.02734375, -0.021874999999999999, 0.0037326388888888891, -0.00017361111111111112, 0, 0, 0},
        {0, -0.0322265625, -0.0322265625, 0.039453124999999999, -0.01056857638888889, 0.0010106646825396826, -3.1001984126984125e-05, 0, 0},
        {0, 0.03564453125, 0.03564453125, -0.062207031250000003, 0.023133680555555557, -0.0034009176587301588, 0.00021236359126984127, -4.6502976190476195e-06, 0},
        {0, -0.037923177083333336, -0.037923177083333336, 0.089892578124999997, -0.043299696180555555, 0.0086743551587301591, -0.00082542782738095238, 3.6599564594356259e-05, -6.0281635802469135e-07}
        };
        real sumValue = 0, sumDeriv = 0;
        #pragma unroll
        for (int n = TAYLOR_ORDER; n >= 0; n--) {
            real value = 0, deriv = 0;
            #pragma unroll
            for (int k = 8; k >= 0; k--) {
                value = value*overlapX + VALUE[n][k];
                deriv = deriv*overlapX + DERIV[n][k];
            }
            sumValue = sumValue*overlapT + value;
            sumDeriv = sumDeriv*overlapT + deriv;
        }
        const real prefactor = EXP(-overlapX)*overlapInvL*overlapInvL*overlapInvL*(real) 0.31830988618379067;
        OVERLAP = prefactor*sumValue;
        OVERLAP_DERIV = prefactor*overlapInvL*sumDeriv;
    }
    else {
        // Closed form (MEDFF SI Eq. 32): S = F/(8 pi r).
        const real aa = overlapA*overlapA, bb = overlapB*overlapB;
        const real d = bb-aa;
        const real invD2 = 1/(d*d);
        const real expA = EXP(-r/overlapA), expB = EXP(-r/overlapB);
        const real c = 4*aa*bb*invD2/d;
        const real f = c*(expA-expB) + r*invD2*(overlapA*expA+overlapB*expB);
        const real dfdr = c*(expB/overlapB-expA/overlapA) + invD2*(overlapA*expA+overlapB*expB-r*(expA+expB));
        OVERLAP = f*invR*(real) 0.039788735772973836;
        OVERLAP_DERIV = (r*dfdr-f)*invR*invR*(real) 0.039788735772973836;
    }
}
