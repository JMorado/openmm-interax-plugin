    // End of the code for one InteraxFF pair force: apply the switching function and the alchemical scaling to
    // pairEnergy and pairDerivative (dE/dr), and accumulate them.
#if USE_SWITCH
    if (r > SWITCH_CUTOFF) {
        real t = (r-SWITCH_CUTOFF)*INV_SWITCH_WIDTH;
        real switchValue = 1+t*t*t*(-10+t*(15-t*6));
        real switchDeriv = t*t*(-30+t*(60-t*30))*INV_SWITCH_WIDTH;
        pairDerivative = pairDerivative*switchValue + pairEnergy*switchDeriv;
        pairEnergy *= switchValue;
    }
#endif
    real scale = (IS_ALCHEMICAL1 != IS_ALCHEMICAL2 ? (real) LAMBDA : (real) 1);
    tempEnergy += scale*pairEnergy;
    dEdR -= scale*pairDerivative*invR;
}
