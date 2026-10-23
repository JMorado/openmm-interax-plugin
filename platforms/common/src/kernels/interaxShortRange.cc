    // Short-range correction (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 with St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j).
    // PARAMS = (k, l, width, isAlchemical).
    {
        SLATER_OVERLAP
        const real widthSum = PARAMS1.z+PARAMS2.z;
        const real normalization = (real) 25.132741228718345*widthSum*widthSum*widthSum;
        const real normalizedOverlap = normalization*shortRangeOverlap;
        const real k = 0.5f*(PARAMS1.x+PARAMS2.x);
        const real l = 0.5f*(PARAMS1.y+PARAMS2.y);
        pairEnergy = (k+l*normalizedOverlap)*normalizedOverlap;
        pairDerivative = (k+2*l*normalizedOverlap)*normalization*shortRangeOverlapDeriv;
    }
