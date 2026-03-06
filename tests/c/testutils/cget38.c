/**
 * @file cget38.c
 * @brief CGET38 tests CTRSEN, a routine for estimating condition numbers of a
 *        cluster of eigenvalues and/or its associated right invariant subspace.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define LDT 20
#define LWORK (2 * LDT * (10 + LDT))
#define NCASES38 20

typedef struct {
    INT n;
    INT ndim;
    INT isrt;
    INT iselec[LDT];  /* 1-based indices from Fortran */
    f32 sin_val;
    f32 sepin;
} zget38_meta_t;

static const zget38_meta_t zget38_meta[NCASES38] = {
    /* Case 0 */ {1, 1, 0, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 0.0f},
    /* Case 1 */ {1, 1, 0, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.0f},
    /* Case 2 */ {5, 3, 0, {2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 2.9582e-31f},
    /* Case 3 */ {5, 3, 0, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.1921e-07f},
    /* Case 4 */ {5, 2, 0, {2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.0f},
    /* Case 5 */ {6, 3, 1, {3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.0124e-36f, 3.2099e-36f},
    /* Case 6 */ {6, 3, 0, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.0124e-36f, 3.2099e-36f},
    /* Case 7 */ {4, 2, 0, {3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.6350e-01f, 3.3122e-01f},
    /* Case 8 */ {4, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 8.4053e-01f, 7.4754e-01f},
    /* Case 9 */ {3, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.9550e-01f, 2.0464e+01f},
    /* Case 10 */ {4, 2, 0, {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.3333e-01f, 1.2569e-01f},
    /* Case 11 */ {4, 3, 0, {1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 8.2843e-01f},
    /* Case 12 */ {4, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.8985e-01f, 4.1447e+00f},
    /* Case 13 */ {5, 2, 1, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.1088e-01f, 4.6912e+00f},
    /* Case 14 */ {3, 2, 0, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 2.2361e-01f, 1.0f},
    /* Case 15 */ {4, 2, 1, {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 7.2803e-05f, 1.1947e-04f},
    /* Case 16 */ {7, 4, 0, {1, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.7241e-01f, 5.2080e-01f},
    /* Case 17 */ {5, 3, 1, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 4.5989e+00f},
    /* Case 18 */ {8, 4, 1, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.5269e-12f, 2.9360e-11f},
    /* Case 19 */ {3, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 2.0f},
};

/* Matrix data stored as flat (re, im) pairs in row-major order */
static const f32 zget38_data[] = {
    /* Case 0: N=1 */
    0.0f, 0.0f,

    /* Case 1: N=1 */
    1.0f, 0.0f,

    /* Case 2: N=5, 5x5 zero */
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,

    /* Case 3: N=5, 5x5 identity */
    1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f,

    /* Case 4: N=5, diag(1,2,3,4,5) */
    1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 2.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 3.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 4.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 5.0f,0.0f,

    /* Case 5: N=6, upper bidiagonal (0,i) + superdiag 1 */
    0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f,

    /* Case 6: N=6, lower bidiagonal (0,i) + subdiag 1 */
    0.0f,1.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    1.0f,0.0f, 0.0f,1.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f,

    /* Case 7: N=4 */
    9.4480e-01f,1.0f, 6.7670e-01f,1.0f, 6.9080e-01f,1.0f, 5.9650e-01f,1.0f,
    5.8760e-01f,1.0f, 8.6420e-01f,1.0f, 6.7690e-01f,1.0f, 7.2600e-02f,1.0f,
    7.2560e-01f,1.0f, 1.9430e-01f,1.0f, 9.6870e-01f,1.0f, 2.8310e-01f,1.0f,
    2.8490e-01f,1.0f, 5.8000e-02f,1.0f, 4.8450e-01f,1.0f, 7.3610e-01f,1.0f,

    /* Case 8: N=4 */
    2.1130e-01f,9.9330e-01f, 8.0960e-01f,4.2370e-01f, 4.8320e-01f,1.1670e-01f, 6.5380e-01f,4.9430e-01f,
    8.2400e-02f,8.3600e-01f, 8.4740e-01f,2.6130e-01f, 6.1350e-01f,6.2500e-01f, 4.8990e-01f,3.6500e-02f,
    7.5990e-01f,7.4690e-01f, 4.5240e-01f,2.4030e-01f, 2.7490e-01f,5.5100e-01f, 7.7410e-01f,2.2600e-01f,
    8.7000e-03f,3.7800e-02f, 8.0750e-01f,3.4050e-01f, 8.8070e-01f,3.5500e-01f, 9.6260e-01f,8.1590e-01f,

    /* Case 9: N=3 */
    1.0f,2.0f, 3.0f,4.0f, 21.0f,22.0f,
    43.0f,44.0f, 13.0f,14.0f, 15.0f,16.0f,
    5.0f,6.0f, 7.0f,8.0f, 25.0f,26.0f,

    /* Case 10: N=4 */
    5.0f,9.0f, 5.0f,5.0f, -6.0f,-6.0f, -7.0f,-7.0f,
    3.0f,3.0f, 6.0f,10.0f, -5.0f,-5.0f, -6.0f,-6.0f,
    2.0f,2.0f, 3.0f,3.0f, -1.0f,3.0f, -5.0f,-5.0f,
    1.0f,1.0f, 2.0f,2.0f, -3.0f,-3.0f, 0.0f,4.0f,

    /* Case 11: N=4 */
    3.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,2.0f,
    1.0f,0.0f, 3.0f,0.0f, 0.0f,-2.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,2.0f, 1.0f,0.0f, 1.0f,0.0f,
    0.0f,-2.0f, 0.0f,0.0f, 1.0f,0.0f, 1.0f,0.0f,

    /* Case 12: N=4 */
    7.0f,0.0f, 3.0f,0.0f, 1.0f,2.0f, -1.0f,2.0f,
    3.0f,0.0f, 7.0f,0.0f, 1.0f,-2.0f, -1.0f,-2.0f,
    1.0f,-2.0f, 1.0f,2.0f, 7.0f,0.0f, -3.0f,0.0f,
    -1.0f,-2.0f, -2.0f,2.0f, -3.0f,0.0f, 7.0f,0.0f,

    /* Case 13: N=5 */
    1.0f,2.0f, 3.0f,4.0f, 21.0f,22.0f, 23.0f,24.0f, 41.0f,42.0f,
    43.0f,44.0f, 13.0f,14.0f, 15.0f,16.0f, 33.0f,34.0f, 35.0f,36.0f,
    5.0f,6.0f, 7.0f,8.0f, 25.0f,26.0f, 27.0f,28.0f, 45.0f,46.0f,
    47.0f,48.0f, 17.0f,18.0f, 19.0f,20.0f, 37.0f,38.0f, 39.0f,40.0f,
    9.0f,10.0f, 11.0f,12.0f, 29.0f,30.0f, 31.0f,32.0f, 49.0f,50.0f,

    /* Case 14: N=3 */
    1.0f,1.0f, -1.0f,-1.0f, 2.0f,2.0f,
    0.0f,0.0f, 0.0f,1.0f, 2.0f,0.0f,
    0.0f,0.0f, -1.0f,0.0f, 3.0f,1.0f,

    /* Case 15: N=4 */
    -4.0f,-2.0f, -5.0f,-6.0f, -2.0f,-6.0f, 0.0f,-2.0f,
    1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f,

    /* Case 16: N=7 */
    2.0f,4.0f, 1.0f,1.0f, 6.0f,2.0f, 3.0f,3.0f, 5.0f,5.0f, 2.0f,6.0f, 1.0f,1.0f,
    1.0f,2.0f, 1.0f,3.0f, 3.0f,1.0f, 5.0f,-4.0f, 1.0f,1.0f, 7.0f,2.0f, 2.0f,3.0f,
    0.0f,0.0f, 3.0f,-2.0f, 1.0f,1.0f, 6.0f,3.0f, 2.0f,1.0f, 1.0f,4.0f, 2.0f,1.0f,
    0.0f,0.0f, 0.0f,0.0f, 2.0f,3.0f, 3.0f,1.0f, 1.0f,2.0f, 2.0f,2.0f, 3.0f,1.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 2.0f,-1.0f, 2.0f,2.0f, 3.0f,1.0f, 1.0f,3.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,-1.0f, 2.0f,1.0f, 2.0f,2.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 2.0f,-2.0f, 1.0f,1.0f,

    /* Case 17: N=5, Hermitian-like */
    0.0f,5.0f, 1.0f,2.0f, 2.0f,3.0f, -3.0f,6.0f, 6.0f,0.0f,
    -1.0f,2.0f, 0.0f,6.0f, 4.0f,5.0f, -3.0f,-2.0f, 5.0f,0.0f,
    -2.0f,3.0f, -4.0f,5.0f, 0.0f,7.0f, 3.0f,0.0f, 2.0f,0.0f,
    3.0f,6.0f, 3.0f,-2.0f, -3.0f,0.0f, 0.0f,-5.0f, 2.0f,1.0f,
    -6.0f,0.0f, -5.0f,0.0f, -2.0f,0.0f, -2.0f,1.0f, 0.0f,2.0f,

    /* Case 18: N=8 */
    0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f,
    0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,2.0f, 2.0f,0.0f, 0.0f,2.0f, 2.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 0.0f,3.0f, 3.0f,0.0f, 0.0f,3.0f, 3.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,1.0f, 0.0f,4.0f, 4.0f,0.0f, 0.0f,4.0f, 4.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.95f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.95f, 1.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.95f, 1.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.95f,

    /* Case 19: N=3 */
    2.0f,0.0f, 0.0f,-1.0f, 0.0f,0.0f,
    0.0f,1.0f, 2.0f,0.0f, 0.0f,0.0f,
    0.0f,0.0f, 0.0f,0.0f, 3.0f,0.0f,
};

static void rowpairs_to_colmajor_c128(const f32* pairs, c64* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = CMPLXF(pairs[2 * (i * n + j)],
                                      pairs[2 * (i * n + j) + 1]);
}

void cget38(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 EPSIN = 5.9605e-8f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    eps = fmaxf(eps, EPSIN);
    rmax[0] = ZERO;
    rmax[1] = ZERO;
    rmax[2] = ZERO;
    lmax[0] = 0;
    lmax[1] = 0;
    lmax[2] = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    ninfo[2] = 0;

    f32 val[3];
    val[0] = sqrtf(smlnum);
    val[1] = ONE;
    val[2] = sqrtf(sqrtf(bignum));

    INT select[LDT], ipnt[LDT];
    f32 rwork[LDT], wsrt[LDT], result[2];
    c64 q[LDT * LDT], qsav[LDT * LDT], qtmp[LDT * LDT];
    c64 t[LDT * LDT], tmp[LDT * LDT], tsav[LDT * LDT];
    c64 tsav1[LDT * LDT], ttmp[LDT * LDT];
    c64 w[LDT], wtmp[LDT];
    c64 work[LWORK];

    const f32* dptr = zget38_data;

    for (INT ic = 0; ic < NCASES38; ic++) {
        INT n = zget38_meta[ic].n;
        INT ndim = zget38_meta[ic].ndim;
        INT isrt = zget38_meta[ic].isrt;
        f32 sin_val = zget38_meta[ic].sin_val;
        f32 sepin = zget38_meta[ic].sepin;

        rowpairs_to_colmajor_c128(dptr, tmp, n, LDT);
        dptr += 2 * n * n;

        f32 tnrm = clange("M", n, n, tmp, LDT, rwork);

        for (INT iscl = 0; iscl < 3; iscl++) {

            (*knt)++;
            clacpy("F", n, n, tmp, LDT, t, LDT);
            f32 vmul = val[iscl];
            for (INT i = 0; i < n; i++)
                cblas_csscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;
            clacpy("F", n, n, t, LDT, tsav, LDT);

            INT info;
            cgehrd(n, 0, n - 1, t, LDT, work, &work[n], LWORK - n, &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0]++;
                continue;
            }

            clacpy("L", n, n, t, LDT, q, LDT);
            cunghr(n, 0, n - 1, q, LDT, work, &work[n], LWORK - n, &info);

            for (INT j = 0; j < n - 2; j++)
                for (INT i = j + 2; i < n; i++)
                    t[i + j * LDT] = CZERO;

            chseqr("S", "V", n, 0, n - 1, t, LDT, w, q, LDT,
                   work, LWORK, &info);
            if (info != 0) {
                lmax[1] = *knt;
                ninfo[1]++;
                continue;
            }

            for (INT i = 0; i < n; i++) {
                ipnt[i] = i;
                select[i] = 0;
            }
            if (isrt == 0) {
                for (INT i = 0; i < n; i++)
                    wsrt[i] = crealf(w[i]);
            } else {
                for (INT i = 0; i < n; i++)
                    wsrt[i] = cimagf(w[i]);
            }
            for (INT i = 0; i < n - 1; i++) {
                INT kmin = i;
                f32 vmin = wsrt[i];
                for (INT j = i + 1; j < n; j++) {
                    if (wsrt[j] < vmin) {
                        kmin = j;
                        vmin = wsrt[j];
                    }
                }
                wsrt[kmin] = wsrt[i];
                wsrt[i] = vmin;
                INT itmp = ipnt[i];
                ipnt[i] = ipnt[kmin];
                ipnt[kmin] = itmp;
            }
            for (INT i = 0; i < ndim; i++)
                select[ipnt[zget38_meta[ic].iselec[i] - 1]] = 1;

            clacpy("F", n, n, q, LDT, qsav, LDT);
            clacpy("F", n, n, t, LDT, tsav1, LDT);
            INT m;
            f32 s, sep;
            ctrsen("B", "V", select, n, t, LDT, q, LDT, wtmp,
                   &m, &s, &sep, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            f32 septmp = sep / vmul;
            f32 stmp = s;

            chst01(n, 0, n - 1, tsav, LDT, t, LDT, q, LDT, work, LWORK,
                   rwork, result);
            f32 vmax = fmaxf(result[0], result[1]);
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            f32 v = fmaxf(TWO * (f32)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            f32 tol, tolin;
            if (v > septmp)
                tol = ONE;
            else
                tol = v / septmp;
            if (v > sepin)
                tolin = ONE;
            else
                tolin = v / sepin;
            tol = fmaxf(tol, smlnum / eps);
            tolin = fmaxf(tolin, smlnum / eps);
            if (eps * (sin_val - tolin) > stmp + tol)
                vmax = ONE / eps;
            else if (sin_val - tolin > stmp + tol)
                vmax = (sin_val - tolin) / (stmp + tol);
            else if (sin_val + tolin < eps * (stmp - tol))
                vmax = ONE / eps;
            else if (sin_val + tolin < stmp - tol)
                vmax = (stmp - tol) / (sin_val + tolin);
            else
                vmax = ONE;
            if (vmax > rmax[1]) {
                rmax[1] = vmax;
                if (ninfo[1] == 0)
                    lmax[1] = *knt;
            }

            if (v > septmp * stmp)
                tol = septmp;
            else
                tol = v / stmp;
            if (v > sepin * sin_val)
                tolin = sepin;
            else
                tolin = v / sin_val;
            tol = fmaxf(tol, smlnum / eps);
            tolin = fmaxf(tolin, smlnum / eps);
            if (eps * (sepin - tolin) > septmp + tol)
                vmax = ONE / eps;
            else if (sepin - tolin > septmp + tol)
                vmax = (sepin - tolin) / (septmp + tol);
            else if (sepin + tolin < eps * (septmp - tol))
                vmax = ONE / eps;
            else if (sepin + tolin < septmp - tol)
                vmax = (septmp - tol) / (sepin + tolin);
            else
                vmax = ONE;
            if (vmax > rmax[1]) {
                rmax[1] = vmax;
                if (ninfo[1] == 0)
                    lmax[1] = *knt;
            }

            if (sin_val <= (f32)(2 * n) * eps && stmp <= (f32)(2 * n) * eps)
                vmax = ONE;
            else if (eps * sin_val > stmp)
                vmax = ONE / eps;
            else if (sin_val > stmp)
                vmax = sin_val / stmp;
            else if (sin_val < eps * stmp)
                vmax = ONE / eps;
            else if (sin_val < stmp)
                vmax = stmp / sin_val;
            else
                vmax = ONE;
            if (vmax > rmax[2]) {
                rmax[2] = vmax;
                if (ninfo[2] == 0)
                    lmax[2] = *knt;
            }

            if (sepin <= v && septmp <= v)
                vmax = ONE;
            else if (eps * sepin > septmp)
                vmax = ONE / eps;
            else if (sepin > septmp)
                vmax = sepin / septmp;
            else if (sepin < eps * septmp)
                vmax = ONE / eps;
            else if (sepin < septmp)
                vmax = septmp / sepin;
            else
                vmax = ONE;
            if (vmax > rmax[2]) {
                rmax[2] = vmax;
                if (ninfo[2] == 0)
                    lmax[2] = *knt;
            }

            vmax = ZERO;
            clacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            clacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ctrsen("E", "V", select, n, ttmp, LDT, qtmp, LDT, wtmp,
                   &m, &stmp, &septmp, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (s != stmp)
                vmax = ONE / eps;
            if (-ONE != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (crealf(ttmp[i + j * LDT]) != crealf(t[i + j * LDT]) ||
                        cimagf(ttmp[i + j * LDT]) != cimagf(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (crealf(qtmp[i + j * LDT]) != crealf(q[i + j * LDT]) ||
                        cimagf(qtmp[i + j * LDT]) != cimagf(q[i + j * LDT]))
                        vmax = ONE / eps;
                }

            clacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            clacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ctrsen("V", "V", select, n, ttmp, LDT, qtmp, LDT, wtmp,
                   &m, &stmp, &septmp, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (-ONE != stmp)
                vmax = ONE / eps;
            if (sep != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (crealf(ttmp[i + j * LDT]) != crealf(t[i + j * LDT]) ||
                        cimagf(ttmp[i + j * LDT]) != cimagf(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (crealf(qtmp[i + j * LDT]) != crealf(q[i + j * LDT]) ||
                        cimagf(qtmp[i + j * LDT]) != cimagf(q[i + j * LDT]))
                        vmax = ONE / eps;
                }

            clacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            clacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ctrsen("E", "N", select, n, ttmp, LDT, qtmp, LDT, wtmp,
                   &m, &stmp, &septmp, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (s != stmp)
                vmax = ONE / eps;
            if (-ONE != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (crealf(ttmp[i + j * LDT]) != crealf(t[i + j * LDT]) ||
                        cimagf(ttmp[i + j * LDT]) != cimagf(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (crealf(qtmp[i + j * LDT]) != crealf(qsav[i + j * LDT]) ||
                        cimagf(qtmp[i + j * LDT]) != cimagf(qsav[i + j * LDT]))
                        vmax = ONE / eps;
                }

            clacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            clacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ctrsen("V", "N", select, n, ttmp, LDT, qtmp, LDT, wtmp,
                   &m, &stmp, &septmp, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (-ONE != stmp)
                vmax = ONE / eps;
            if (sep != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (crealf(ttmp[i + j * LDT]) != crealf(t[i + j * LDT]) ||
                        cimagf(ttmp[i + j * LDT]) != cimagf(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (crealf(qtmp[i + j * LDT]) != crealf(qsav[i + j * LDT]) ||
                        cimagf(qtmp[i + j * LDT]) != cimagf(qsav[i + j * LDT]))
                        vmax = ONE / eps;
                }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }
        }
    }
}
