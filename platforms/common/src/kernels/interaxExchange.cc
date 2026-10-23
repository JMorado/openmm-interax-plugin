    // Exchange repulsion A_i A_j S(r; s_i, s_j). PARAMS = (amplitude, width, isAlchemical, 0).
    {
        SLATER_OVERLAP
        pairEnergy = PARAMS1.x*PARAMS2.x*exchangeOverlap;
        pairDerivative = PARAMS1.x*PARAMS2.x*exchangeOverlapDeriv;
    }
