/**
 * @file cget35.c
 * @brief CGET35 tests CTRSYL, a routine for solving the Sylvester matrix
 *        equation op(A)*X + ISGN*X*op(B) = scale*C.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

#define LDT 10
#define NCASE 9

/**
 * CGET35 tests CTRSYL, a routine for solving the Sylvester matrix
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
void cget35(f32* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f32 ZERO  = 0.0f;
    const f32 ONE   = 1.0f;
    const f32 TWO   = 2.0f;
    const f32 LARGE = 1.0e6f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    static const INT mdim[NCASE] = {1, 1, 4, 4, 4, 4, 4, 4, 6};
    static const INT ndim[NCASE] = {1, 3, 4, 4, 4, 4, 4, 3, 5};

    /* Test case 1: M=1, N=1 */
    static const c64 atmp1[1] = {CMPLXF(2.0f, 0.0f)};
    static const c64 btmp1[1] = {CMPLXF(2.0f, 0.0f)};
    static const c64 ctmp1[1] = {CMPLXF(1.0f, 1.0f)};

    /* Test case 2: M=1, N=3 */
    static const c64 atmp2[1] = {CMPLXF(1.0f, 1.0f)};
    static const c64 btmp2[3*3] = {
        /* col 0 */  CMPLXF(1.0f, 1.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 1 */  CMPLXF(1.0f, 1.0f),  CMPLXF(1.5f, 1.5f),  CMPLXF(0.0f, 0.0f),
        /* col 2 */  CMPLXF(1.0f, 1.0f),  CMPLXF(2.0f, 1.0f),  CMPLXF(2.0f, 2.0f)
    };
    static const c64 ctmp2[1*3] = {
        CMPLXF(2.0f, 1.0f),  CMPLXF(2.0f, 1.0f),  CMPLXF(9.0f, 0.0f)
    };

    /* Test case 3: M=4, N=4 (all zeros) */
    static const c64 atmp3[4*4] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };
    static const c64 btmp3[4*4] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };
    static const c64 ctmp3[4*4] = {
        /* col 0 */  CMPLXF(1.0f, 0.0f),  CMPLXF(2.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 7.0f),
        /* col 1 */  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 2 */  CMPLXF(2.0f, 0.0f),  CMPLXF(8.0f, 9.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(2.0f, 0.0f),
        /* col 3 */  CMPLXF(1.0f, 3.0f),  CMPLXF(2.0f, 2.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f)
    };

    /* Test case 4: M=4, N=4 (all zeros) */
    static const c64 atmp4[4*4] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };
    static const c64 btmp4[4*4] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };
    static const c64 ctmp4[4*4] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };

    /* Test case 5: M=4, N=4 (Identity, Identity, Identity) */
    static const c64 atmp5[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };
    static const c64 btmp5[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };
    static const c64 ctmp5[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };

    /* Test case 6: M=4, N=4 (Identity, -Identity, Identity) */
    static const c64 atmp6[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };
    static const c64 btmp6[4*4] = {
       CMPLXF(-1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f),CMPLXF(-1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),CMPLXF(-1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),CMPLXF(-1.0f, 0.0f)
    };
    static const c64 ctmp6[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };

    /* Test case 7: M=4, N=4 */
    static const c64 atmp7[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 1.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 1.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 1.0f), CMPLXF(1.0f, 0.0f)
    };
    static const c64 btmp7[4*4] = {
       CMPLXF(-1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 1.0f),CMPLXF(-1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 1.0f),CMPLXF(-1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 1.0f),CMPLXF(-1.0f, 0.0f)
    };
    static const c64 ctmp7[4*4] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };

    /* Test case 8: M=4, N=3 */
    static const c64 atmp8[4*4] = {
        CMPLXF(0.0621f, 0.7054f), CMPLXF(0.0f, 0.0f),        CMPLXF(0.0f, 0.0f),        CMPLXF(0.0f, 0.0f),
        CMPLXF(0.1062f, 0.0503f), CMPLXF(0.2640f, 0.5782f),  CMPLXF(0.0f, 0.0f),        CMPLXF(0.0f, 0.0f),
        CMPLXF(0.6553f, 0.5876f), CMPLXF(0.9700f, 0.7256f),  CMPLXF(0.0380f, 0.2849f),  CMPLXF(0.0f, 0.0f),
        CMPLXF(0.2560f, 0.8642f), CMPLXF(0.5598f, 0.1943f),  CMPLXF(0.9166f, 0.0580f),  CMPLXF(0.1402f, 0.6908f)
    };
    static const c64 btmp8[3*3] = {
        CMPLXF(0.6769f, 0.6219f), CMPLXF(0.0f, 0.0f),        CMPLXF(0.0f, 0.0f),
        CMPLXF(0.5965f, 0.0505f), CMPLXF(0.0726f, 0.7195f),  CMPLXF(0.0f, 0.0f),
        CMPLXF(0.7361f, 0.5069f), CMPLXF(0.2531f, 0.9764f),  CMPLXF(0.3481f, 0.5602f)
    };
    static const c64 ctmp8[4*3] = {
        CMPLXF(0.9110f, 0.7001f), CMPLXF(0.0728f, 0.5887f), CMPLXF(0.1729f, 0.6041f), CMPLXF(0.3785f, 0.7924f),
        CMPLXF(0.1821f, 0.5406f), CMPLXF(0.3271f, 0.5647f), CMPLXF(0.9368f, 0.3514f), CMPLXF(0.6588f, 0.8646f),
        CMPLXF(0.8879f, 0.5813f), CMPLXF(0.3793f, 0.1667f), CMPLXF(0.8149f, 0.3535f), CMPLXF(0.1353f, 0.8362f)
    };

    /* Test case 9: M=6, N=5 */
    static const c64 atmp9[6*6] = {
        /* col 0 */
         CMPLXF(3.0f, 5.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 1 */
         CMPLXF(3.0f, 22.0f), CMPLXF(-3.0f, 5.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 2 */
         CMPLXF(2.0f, 3.0f),  CMPLXF(3.0f, 2.0f),  CMPLXF(3.0f, 2.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 3 */
         CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),CMPLXF(-33.0f, 2.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 4 */
         CMPLXF(3.0f, 3.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),CMPLXF(-22.0f, 3.0f),  CMPLXF(0.0f, 0.0f),
        /* col 5 */
       CMPLXF(311.0f, 2.0f), CMPLXF(11.0f, 2.0f),  CMPLXF(1.0f, -2.0f),  CMPLXF(1.0f, 2.0f),  CMPLXF(1.0f, 2.0f),  CMPLXF(2.0f, -3.0f)
    };
    static const c64 btmp9[5*5] = {
        /* col 0 */
         CMPLXF(9.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 1 */
         CMPLXF(2.0f, 0.0f),CMPLXF(-19.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 2 */
       CMPLXF(-12.0f, 0.0f), CMPLXF(12.0f, 0.0f), CMPLXF(98.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 3 */
         CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f), CMPLXF(11.0f, 0.0f), CMPLXF(13.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 4 */
         CMPLXF(3.0f, 0.0f),  CMPLXF(3.0f, 0.0f),  CMPLXF(3.0f, 0.0f), CMPLXF(11.0f, 0.0f), CMPLXF(13.0f, 0.0f)
    };
    static const c64 ctmp9[6*5] = {
        /* col 0 */
         CMPLXF(3.0f, -5.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 1 */
         CMPLXF(3.0f, 22.0f),CMPLXF(-3.0f, 5.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 2 */
         CMPLXF(2.0f, 31.0f), CMPLXF(33.0f, 22.0f),CMPLXF(-3.0f, 2.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 3 */
         CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),CMPLXF(-33.0f, 2.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),
        /* col 4 */
         CMPLXF(3.0f, 3.0f), CMPLXF(-2.0f, 3.0f),  CMPLXF(2.0f, -3.0f),  CMPLXF(2.0f, 3.0f),CMPLXF(-22.0f, 3.0f),  CMPLXF(0.0f, -2.0f)
    };

    static const c64* atmp_all[NCASE] = {
        atmp1, atmp2, atmp3, atmp4, atmp5, atmp6, atmp7, atmp8, atmp9
    };
    static const c64* btmp_all[NCASE] = {
        btmp1, btmp2, btmp3, btmp4, btmp5, btmp6, btmp7, btmp8, btmp9
    };
    static const c64* ctmp_all[NCASE] = {
        ctmp1, ctmp2, ctmp3, ctmp4, ctmp5, ctmp6, ctmp7, ctmp8, ctmp9
    };

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    f32 vm1[3], vm2[3];
    vm1[0] = sqrtf(smlnum);
    vm1[1] = ONE;
    vm1[2] = LARGE;
    vm2[0] = ONE;
    vm2[1] = ONE + TWO * eps;
    vm2[2] = TWO;

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    c64 a[LDT * LDT], b[LDT * LDT], c[LDT * LDT];
    c64 csav[LDT * LDT];
    f32 dum[1];
    f32 scale, tnrm, xnrm, res1, res;
    c64 rmul;
    INT m, n, info;

    for (INT ic = 0; ic < NCASE; ic++) {
        m = mdim[ic];
        n = ndim[ic];
        const c64* atmp = atmp_all[ic];
        const c64* btmp = btmp_all[ic];
        const c64* ctmp = ctmp_all[ic];

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
                                            tnrm = fmaxf(tnrm, cabsf(a[i + j * LDT]));
                                        }
                                        a[i + i * LDT] = a[i + i * LDT] * vm2[imlad];
                                        tnrm = fmaxf(tnrm, cabsf(a[i + i * LDT]));
                                    }
                                    for (INT i = 0; i < n; i++) {
                                        for (INT j = 0; j < n; j++) {
                                            b[i + j * LDT] = btmp[i + j * n] * vm1[imlb];
                                            tnrm = fmaxf(tnrm, cabsf(b[i + j * LDT]));
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
                                    ctrsyl(trana, tranb, isgn, m, n, a,
                                           LDT, b, LDT, c, LDT, &scale,
                                           &info);
                                    if (info != 0)
                                        (*ninfo)++;
                                    xnrm = clange("M", m, n, c, LDT, dum);
                                    rmul = CONE;
                                    if (xnrm > ONE && tnrm > ONE) {
                                        if (xnrm > bignum / tnrm) {
                                            rmul = CMPLXF(fmaxf(xnrm, tnrm), 0.0f);
                                            rmul = CONE / rmul;
                                        }
                                    }
                                    {
                                        c64 beta1 = -scale * rmul;
                                        cblas_cgemm(CblasColMajor,
                                                    itrana == 0 ? CblasNoTrans : CblasConjTrans,
                                                    CblasNoTrans,
                                                    m, n, m, &rmul,
                                                    a, LDT, c, LDT, &beta1,
                                                    csav, LDT);
                                    }
                                    {
                                        c64 alpha2 = (f32)isgn * rmul;
                                        cblas_cgemm(CblasColMajor,
                                                    CblasNoTrans,
                                                    itranb == 0 ? CblasNoTrans : CblasConjTrans,
                                                    m, n, n, &alpha2,
                                                    c, LDT, b, LDT, &CONE,
                                                    csav, LDT);
                                    }
                                    res1 = clange("M", m, n, csav, LDT, dum);
                                    res = res1 / fmaxf(smlnum,
                                              fmaxf(smlnum * xnrm,
                                                   ((cabsf(rmul) * tnrm) * eps) * xnrm));
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
