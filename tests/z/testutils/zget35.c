/**
 * @file zget35.c
 * @brief ZGET35 tests ZTRSYL, a routine for solving the Sylvester matrix
 *        equation op(A)*X + ISGN*X*op(B) = scale*C.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

#define LDT 10
#define NCASE 9

/**
 * ZGET35 tests ZTRSYL, a routine for solving the Sylvester matrix
 * equation
 *
 *    op(A)*X + ISGN*X*op(B) = scale*C,
 *
 * A and B are assumed to be in Schur canonical form, op() represents an
 * optional transpose, and ISGN can be -1 or +1.  Scale is an output
 * less than or equal to 1, chosen to avoid overflow in X.
 *
 * The test code verifies that the following residual is order 1:
 *
 *    norm(op(A)*X + ISGN*X*op(B) - scale*C) /
 *        (EPS*max(norm(A),norm(B))*norm(X))
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples where INFO is nonzero.
 * @param[out]    knt     Total number of examples tested.
 */
void zget35(f64* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f64 ZERO  = 0.0;
    const f64 ONE   = 1.0;
    const f64 TWO   = 2.0;
    const f64 LARGE = 1.0e6;
    const c128 CONE = CMPLX(1.0, 0.0);

    static const INT mdim[NCASE] = {1, 1, 4, 4, 4, 4, 4, 4, 6};
    static const INT ndim[NCASE] = {1, 3, 4, 4, 4, 4, 4, 3, 5};

    /* Test case 1: M=1, N=1 */
    static const c128 atmp1[1] = {CMPLX(2.0, 0.0)};
    static const c128 btmp1[1] = {CMPLX(2.0, 0.0)};
    static const c128 ctmp1[1] = {CMPLX(1.0, 1.0)};

    /* Test case 2: M=1, N=3 */
    static const c128 atmp2[1] = {CMPLX(1.0, 1.0)};
    static const c128 btmp2[3*3] = {
        /* col 0 */  CMPLX(1.0, 1.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 1 */  CMPLX(1.0, 1.0),  CMPLX(1.5, 1.5),  CMPLX(0.0, 0.0),
        /* col 2 */  CMPLX(1.0, 1.0),  CMPLX(2.0, 1.0),  CMPLX(2.0, 2.0)
    };
    static const c128 ctmp2[1*3] = {
        CMPLX(2.0, 1.0),  CMPLX(2.0, 1.0),  CMPLX(9.0, 0.0)
    };

    /* Test case 3: M=4, N=4 (all zeros) */
    static const c128 atmp3[4*4] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };
    static const c128 btmp3[4*4] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };
    static const c128 ctmp3[4*4] = {
        /* col 0 */  CMPLX(1.0, 0.0),  CMPLX(2.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 7.0),
        /* col 1 */  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 2 */  CMPLX(2.0, 0.0),  CMPLX(8.0, 9.0),  CMPLX(0.0, 0.0),  CMPLX(2.0, 0.0),
        /* col 3 */  CMPLX(1.0, 3.0),  CMPLX(2.0, 2.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0)
    };

    /* Test case 4: M=4, N=4 (all zeros) */
    static const c128 atmp4[4*4] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };
    static const c128 btmp4[4*4] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };
    static const c128 ctmp4[4*4] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };

    /* Test case 5: M=4, N=4 (Identity, Identity, Identity) */
    static const c128 atmp5[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };
    static const c128 btmp5[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };
    static const c128 ctmp5[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };

    /* Test case 6: M=4, N=4 (Identity, -Identity, Identity) */
    static const c128 atmp6[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };
    static const c128 btmp6[4*4] = {
       CMPLX(-1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0),CMPLX(-1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),CMPLX(-1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),CMPLX(-1.0, 0.0)
    };
    static const c128 ctmp6[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };

    /* Test case 7: M=4, N=4 */
    static const c128 atmp7[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 1.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 1.0), CMPLX(0.0, 1.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 1.0), CMPLX(1.0, 0.0)
    };
    static const c128 btmp7[4*4] = {
       CMPLX(-1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 1.0),CMPLX(-1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 1.0),CMPLX(-1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 1.0),CMPLX(-1.0, 0.0)
    };
    static const c128 ctmp7[4*4] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };

    /* Test case 8: M=4, N=3 */
    static const c128 atmp8[4*4] = {
        CMPLX(0.0621, 0.7054), CMPLX(0.0, 0.0),        CMPLX(0.0, 0.0),        CMPLX(0.0, 0.0),
        CMPLX(0.1062, 0.0503), CMPLX(0.2640, 0.5782),  CMPLX(0.0, 0.0),        CMPLX(0.0, 0.0),
        CMPLX(0.6553, 0.5876), CMPLX(0.9700, 0.7256),  CMPLX(0.0380, 0.2849),  CMPLX(0.0, 0.0),
        CMPLX(0.2560, 0.8642), CMPLX(0.5598, 0.1943),  CMPLX(0.9166, 0.0580),  CMPLX(0.1402, 0.6908)
    };
    static const c128 btmp8[3*3] = {
        CMPLX(0.6769, 0.6219), CMPLX(0.0, 0.0),        CMPLX(0.0, 0.0),
        CMPLX(0.5965, 0.0505), CMPLX(0.0726, 0.7195),  CMPLX(0.0, 0.0),
        CMPLX(0.7361, 0.5069), CMPLX(0.2531, 0.9764),  CMPLX(0.3481, 0.5602)
    };
    static const c128 ctmp8[4*3] = {
        CMPLX(0.9110, 0.7001), CMPLX(0.0728, 0.5887), CMPLX(0.1729, 0.6041), CMPLX(0.3785, 0.7924),
        CMPLX(0.1821, 0.5406), CMPLX(0.3271, 0.5647), CMPLX(0.9368, 0.3514), CMPLX(0.6588, 0.8646),
        CMPLX(0.8879, 0.5813), CMPLX(0.3793, 0.1667), CMPLX(0.8149, 0.3535), CMPLX(0.1353, 0.8362)
    };

    /* Test case 9: M=6, N=5 */
    static const c128 atmp9[6*6] = {
        /* col 0 */
         CMPLX(3.0, 5.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 1 */
         CMPLX(3.0, 22.0), CMPLX(-3.0, 5.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 2 */
         CMPLX(2.0, 3.0),  CMPLX(3.0, 2.0),  CMPLX(3.0, 2.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 3 */
         CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),CMPLX(-33.0, 2.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 4 */
         CMPLX(3.0, 3.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),CMPLX(-22.0, 3.0),  CMPLX(0.0, 0.0),
        /* col 5 */
       CMPLX(311.0, 2.0), CMPLX(11.0, 2.0),  CMPLX(1.0, -2.0),  CMPLX(1.0, 2.0),  CMPLX(1.0, 2.0),  CMPLX(2.0, -3.0)
    };
    static const c128 btmp9[5*5] = {
        /* col 0 */
         CMPLX(9.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 1 */
         CMPLX(2.0, 0.0),CMPLX(-19.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 2 */
       CMPLX(-12.0, 0.0), CMPLX(12.0, 0.0), CMPLX(98.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 3 */
         CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0), CMPLX(11.0, 0.0), CMPLX(13.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 4 */
         CMPLX(3.0, 0.0),  CMPLX(3.0, 0.0),  CMPLX(3.0, 0.0), CMPLX(11.0, 0.0), CMPLX(13.0, 0.0)
    };
    static const c128 ctmp9[6*5] = {
        /* col 0 */
         CMPLX(3.0, -5.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 1 */
         CMPLX(3.0, 22.0),CMPLX(-3.0, 5.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 2 */
         CMPLX(2.0, 31.0), CMPLX(33.0, 22.0),CMPLX(-3.0, 2.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 3 */
         CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),CMPLX(-33.0, 2.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),
        /* col 4 */
         CMPLX(3.0, 3.0), CMPLX(-2.0, 3.0),  CMPLX(2.0, -3.0),  CMPLX(2.0, 3.0),CMPLX(-22.0, 3.0),  CMPLX(0.0, -2.0)
    };

    static const c128* atmp_all[NCASE] = {
        atmp1, atmp2, atmp3, atmp4, atmp5, atmp6, atmp7, atmp8, atmp9
    };
    static const c128* btmp_all[NCASE] = {
        btmp1, btmp2, btmp3, btmp4, btmp5, btmp6, btmp7, btmp8, btmp9
    };
    static const c128* ctmp_all[NCASE] = {
        ctmp1, ctmp2, ctmp3, ctmp4, ctmp5, ctmp6, ctmp7, ctmp8, ctmp9
    };

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    f64 vm1[3], vm2[3];
    vm1[0] = sqrt(smlnum);
    vm1[1] = ONE;
    vm1[2] = LARGE;
    vm2[0] = ONE;
    vm2[1] = ONE + TWO * eps;
    vm2[2] = TWO;

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    c128 a[LDT * LDT], b[LDT * LDT], c[LDT * LDT];
    c128 csav[LDT * LDT];
    f64 dum[1];
    f64 scale, tnrm, xnrm, res1, res;
    c128 rmul;
    INT m, n, info;

    for (INT ic = 0; ic < NCASE; ic++) {
        m = mdim[ic];
        n = ndim[ic];
        const c128* atmp = atmp_all[ic];
        const c128* btmp = btmp_all[ic];
        const c128* ctmp = ctmp_all[ic];

        for (INT imla = 0; imla < 3; imla++) {
            for (INT imlad = 0; imlad < 3; imlad++) {
                for (INT imlb = 0; imlb < 3; imlb++) {
                    for (INT imlc = 0; imlc < 3; imlc++) {
                        for (INT itrana = 0; itrana < 2; itrana++) {
                            for (INT itranb = 0; itranb < 2; itranb++) {
                                for (INT isgn = -1; isgn <= 1; isgn += 2) {
                                    const char* trana = (itrana == 0) ? "N" : "C";
                                    const char* tranb = (itranb == 0) ? "N" : "C";
                                    tnrm = ZERO;
                                    for (INT i = 0; i < m; i++) {
                                        for (INT j = 0; j < m; j++) {
                                            a[i + j * LDT] = atmp[i + j * m] * vm1[imla];
                                            tnrm = fmax(tnrm, cabs(a[i + j * LDT]));
                                        }
                                        a[i + i * LDT] = a[i + i * LDT] * vm2[imlad];
                                        tnrm = fmax(tnrm, cabs(a[i + i * LDT]));
                                    }
                                    for (INT i = 0; i < n; i++) {
                                        for (INT j = 0; j < n; j++) {
                                            b[i + j * LDT] = btmp[i + j * n] * vm1[imlb];
                                            tnrm = fmax(tnrm, cabs(b[i + j * LDT]));
                                        }
                                    }
                                    if (tnrm == ZERO)
                                        tnrm = ONE;
                                    for (INT i = 0; i < m; i++) {
                                        for (INT j = 0; j < n; j++) {
                                            c[i + j * LDT] = ctmp[i + j * m] * vm1[imlc];
                                            csav[i + j * LDT] = c[i + j * LDT];
                                        }
                                    }
                                    (*knt)++;
                                    ztrsyl(trana, tranb, isgn, m, n, a,
                                           LDT, b, LDT, c, LDT, &scale,
                                           &info);
                                    if (info != 0)
                                        (*ninfo)++;
                                    xnrm = zlange("M", m, n, c, LDT, dum);
                                    rmul = CONE;
                                    if (xnrm > ONE && tnrm > ONE) {
                                        if (xnrm > bignum / tnrm) {
                                            rmul = CMPLX(fmax(xnrm, tnrm), 0.0);
                                            rmul = CONE / rmul;
                                        }
                                    }
                                    {
                                        c128 beta1 = -scale * rmul;
                                        cblas_zgemm(CblasColMajor,
                                                    itrana == 0 ? CblasNoTrans : CblasConjTrans,
                                                    CblasNoTrans,
                                                    m, n, m, &rmul,
                                                    a, LDT, c, LDT, &beta1,
                                                    csav, LDT);
                                    }
                                    {
                                        c128 alpha2 = (f64)isgn * rmul;
                                        cblas_zgemm(CblasColMajor,
                                                    CblasNoTrans,
                                                    itranb == 0 ? CblasNoTrans : CblasConjTrans,
                                                    m, n, n, &alpha2,
                                                    c, LDT, b, LDT, &CONE,
                                                    csav, LDT);
                                    }
                                    res1 = zlange("M", m, n, csav, LDT, dum);
                                    res = res1 / fmax(smlnum,
                                              fmax(smlnum * xnrm,
                                                   ((cabs(rmul) * tnrm) * eps) * xnrm));
                                    if (res > *rmax) {
                                        *lmax = *knt;
                                        *rmax = res;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
