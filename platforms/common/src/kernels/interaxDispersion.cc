    // Dispersion: C6, C8 and C10 combined from the XDM moments and polarizabilities, with Tang-Toennies damping.
    // PARAMS = (m1, m2, m3, alpha), EXTRA = (width, isAlchemical).
    {
        const real4 p1 = PARAMS1, p2 = PARAMS2;
        // floor: zero dispersion (not 0/0) for a pair without polarizabilities (two virtual sites)
        const real prefactor = p1.w*p2.w/max(p1.w*p2.x+p2.w*p1.x, (real) 1e-30f);
        const real c6 = prefactor*p1.x*p2.x;
        const real c8 = 1.5f*prefactor*(p1.x*p2.y+p1.y*p2.x);
        const real c10 = prefactor*(2*(p1.x*p2.z+p1.z*p2.x)+(real) 4.2*p1.y*p2.y);
        const real dxdr = 2*DAMPING_SCALE/(EXTRA1.x+EXTRA2.x);
        const real x = dxdr*r;

        // Terms x^k/k! for k <= 10, and the damping functions f_n = 1 - exp(-x) sum_{k<=n} x^k/k!. Where x < (n+1)/2
        // that difference cancels, and f_n = exp(-x) sum_{k>n} x^k/k! is used instead.
        const real expX = EXP(-x);
        real terms[11];
        terms[0] = 1;
        #pragma unroll
        for (int k = 1; k <= 10; k++)
            terms[k] = terms[k-1]*x*(real) (1.0/k);
        real sum6 = 0;
        #pragma unroll
        for (int k = 0; k <= 6; k++)
            sum6 += terms[k];
        const real sum8 = sum6+terms[7]+terms[8];
        const real sum10 = sum8+terms[9]+terms[10];
        real f6 = 1-expX*sum6, f8 = 1-expX*sum8, f10 = 1-expX*sum10;
        if (x < 5.5f) {
            real term = terms[10], tail10 = 0;
            for (int k = 11; k <= 40; k++) {
                term *= x*(real) (1.0/k);
                tail10 += term;
            }
            const real tail8 = tail10+terms[9]+terms[10];
            const real tail6 = tail8+terms[7]+terms[8];
            f10 = expX*tail10;
            if (x < 4.5f)
                f8 = expX*tail8;
            if (x < 3.5f)
                f6 = expX*tail6;
        }
        const real invR2 = invR*invR;
        const real invR6 = invR2*invR2*invR2;
        const real e6 = c6*invR6, e8 = c8*invR6*invR2, e10 = c10*invR6*invR2*invR2;
        pairEnergy = -(f6*e6 + f8*e8 + f10*e10);
        pairDerivative = -(e6*(expX*terms[6]*dxdr - 6*f6*invR) + e8*(expX*terms[8]*dxdr - 8*f8*invR) + e10*(expX*terms[10]*dxdr - 10*f10*invR));
    }
