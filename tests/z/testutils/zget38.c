/**
 * @file zget38.c
 * @brief ZGET38 tests ZTRSEN, a routine for estimating condition numbers of a
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
    f64 sin_val;
    f64 sepin;
} zget38_meta_t;

static const zget38_meta_t zget38_meta[NCASES38] = {
    /* Case 0 */ {1, 1, 0, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 0.0},
    /* Case 1 */ {1, 1, 0, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.0},
    /* Case 2 */ {5, 3, 0, {2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 2.9582e-31},
    /* Case 3 */ {5, 3, 0, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.1921e-07},
    /* Case 4 */ {5, 2, 0, {2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.0},
    /* Case 5 */ {6, 3, 1, {3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.0124e-36, 3.2099e-36},
    /* Case 6 */ {6, 3, 0, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.0124e-36, 3.2099e-36},
    /* Case 7 */ {4, 2, 0, {3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.6350e-01, 3.3122e-01},
    /* Case 8 */ {4, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 8.4053e-01, 7.4754e-01},
    /* Case 9 */ {3, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.9550e-01, 2.0464e+01},
    /* Case 10 */ {4, 2, 0, {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.3333e-01, 1.2569e-01},
    /* Case 11 */ {4, 3, 0, {1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 8.2843e-01},
    /* Case 12 */ {4, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.8985e-01, 4.1447e+00},
    /* Case 13 */ {5, 2, 1, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.1088e-01, 4.6912e+00},
    /* Case 14 */ {3, 2, 0, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 2.2361e-01, 1.0},
    /* Case 15 */ {4, 2, 1, {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 7.2803e-05, 1.1947e-04},
    /* Case 16 */ {7, 4, 0, {1, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.7241e-01, 5.2080e-01},
    /* Case 17 */ {5, 3, 1, {1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 4.5989e+00},
    /* Case 18 */ {8, 4, 1, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.5269e-12, 2.9360e-11},
    /* Case 19 */ {3, 2, 0, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 2.0},
};

/* Matrix data stored as flat (re, im) pairs in row-major order */
static const f64 zget38_data[] = {
    /* Case 0: N=1 */
    0.0, 0.0,

    /* Case 1: N=1 */
    1.0, 0.0,

    /* Case 2: N=5, 5x5 zero */
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,

    /* Case 3: N=5, 5x5 identity */
    1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,0.0,

    /* Case 4: N=5, diag(1,2,3,4,5) */
    1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 2.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 3.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 4.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 5.0,0.0,

    /* Case 5: N=6, upper bidiagonal (0,i) + superdiag 1 */
    0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,1.0, 1.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,1.0,

    /* Case 6: N=6, lower bidiagonal (0,i) + subdiag 1 */
    0.0,1.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    1.0,0.0, 0.0,1.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 1.0,0.0, 0.0,1.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,1.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,1.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,1.0,

    /* Case 7: N=4 */
    9.4480e-01,1.0, 6.7670e-01,1.0, 6.9080e-01,1.0, 5.9650e-01,1.0,
    5.8760e-01,1.0, 8.6420e-01,1.0, 6.7690e-01,1.0, 7.2600e-02,1.0,
    7.2560e-01,1.0, 1.9430e-01,1.0, 9.6870e-01,1.0, 2.8310e-01,1.0,
    2.8490e-01,1.0, 5.8000e-02,1.0, 4.8450e-01,1.0, 7.3610e-01,1.0,

    /* Case 8: N=4 */
    2.1130e-01,9.9330e-01, 8.0960e-01,4.2370e-01, 4.8320e-01,1.1670e-01, 6.5380e-01,4.9430e-01,
    8.2400e-02,8.3600e-01, 8.4740e-01,2.6130e-01, 6.1350e-01,6.2500e-01, 4.8990e-01,3.6500e-02,
    7.5990e-01,7.4690e-01, 4.5240e-01,2.4030e-01, 2.7490e-01,5.5100e-01, 7.7410e-01,2.2600e-01,
    8.7000e-03,3.7800e-02, 8.0750e-01,3.4050e-01, 8.8070e-01,3.5500e-01, 9.6260e-01,8.1590e-01,

    /* Case 9: N=3 */
    1.0,2.0, 3.0,4.0, 21.0,22.0,
    43.0,44.0, 13.0,14.0, 15.0,16.0,
    5.0,6.0, 7.0,8.0, 25.0,26.0,

    /* Case 10: N=4 */
    5.0,9.0, 5.0,5.0, -6.0,-6.0, -7.0,-7.0,
    3.0,3.0, 6.0,10.0, -5.0,-5.0, -6.0,-6.0,
    2.0,2.0, 3.0,3.0, -1.0,3.0, -5.0,-5.0,
    1.0,1.0, 2.0,2.0, -3.0,-3.0, 0.0,4.0,

    /* Case 11: N=4 */
    3.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,2.0,
    1.0,0.0, 3.0,0.0, 0.0,-2.0, 0.0,0.0,
    0.0,0.0, 0.0,2.0, 1.0,0.0, 1.0,0.0,
    0.0,-2.0, 0.0,0.0, 1.0,0.0, 1.0,0.0,

    /* Case 12: N=4 */
    7.0,0.0, 3.0,0.0, 1.0,2.0, -1.0,2.0,
    3.0,0.0, 7.0,0.0, 1.0,-2.0, -1.0,-2.0,
    1.0,-2.0, 1.0,2.0, 7.0,0.0, -3.0,0.0,
    -1.0,-2.0, -2.0,2.0, -3.0,0.0, 7.0,0.0,

    /* Case 13: N=5 */
    1.0,2.0, 3.0,4.0, 21.0,22.0, 23.0,24.0, 41.0,42.0,
    43.0,44.0, 13.0,14.0, 15.0,16.0, 33.0,34.0, 35.0,36.0,
    5.0,6.0, 7.0,8.0, 25.0,26.0, 27.0,28.0, 45.0,46.0,
    47.0,48.0, 17.0,18.0, 19.0,20.0, 37.0,38.0, 39.0,40.0,
    9.0,10.0, 11.0,12.0, 29.0,30.0, 31.0,32.0, 49.0,50.0,

    /* Case 14: N=3 */
    1.0,1.0, -1.0,-1.0, 2.0,2.0,
    0.0,0.0, 0.0,1.0, 2.0,0.0,
    0.0,0.0, -1.0,0.0, 3.0,1.0,

    /* Case 15: N=4 */
    -4.0,-2.0, -5.0,-6.0, -2.0,-6.0, 0.0,-2.0,
    1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 1.0,0.0, 0.0,0.0,

    /* Case 16: N=7 */
    2.0,4.0, 1.0,1.0, 6.0,2.0, 3.0,3.0, 5.0,5.0, 2.0,6.0, 1.0,1.0,
    1.0,2.0, 1.0,3.0, 3.0,1.0, 5.0,-4.0, 1.0,1.0, 7.0,2.0, 2.0,3.0,
    0.0,0.0, 3.0,-2.0, 1.0,1.0, 6.0,3.0, 2.0,1.0, 1.0,4.0, 2.0,1.0,
    0.0,0.0, 0.0,0.0, 2.0,3.0, 3.0,1.0, 1.0,2.0, 2.0,2.0, 3.0,1.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 2.0,-1.0, 2.0,2.0, 3.0,1.0, 1.0,3.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,-1.0, 2.0,1.0, 2.0,2.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 2.0,-2.0, 1.0,1.0,

    /* Case 17: N=5, Hermitian-like */
    0.0,5.0, 1.0,2.0, 2.0,3.0, -3.0,6.0, 6.0,0.0,
    -1.0,2.0, 0.0,6.0, 4.0,5.0, -3.0,-2.0, 5.0,0.0,
    -2.0,3.0, -4.0,5.0, 0.0,7.0, 3.0,0.0, 2.0,0.0,
    3.0,6.0, 3.0,-2.0, -3.0,0.0, 0.0,-5.0, 2.0,1.0,
    -6.0,0.0, -5.0,0.0, -2.0,0.0, -2.0,1.0, 0.0,2.0,

    /* Case 18: N=8 */
    0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,1.0, 1.0,0.0,
    0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,2.0, 2.0,0.0, 0.0,2.0, 2.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,1.0, 1.0,0.0, 0.0,3.0, 3.0,0.0, 0.0,3.0, 3.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,1.0, 0.0,4.0, 4.0,0.0, 0.0,4.0, 4.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.95, 1.0,0.0, 0.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.95, 1.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.95, 1.0,0.0,
    0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.95,

    /* Case 19: N=3 */
    2.0,0.0, 0.0,-1.0, 0.0,0.0,
    0.0,1.0, 2.0,0.0, 0.0,0.0,
    0.0,0.0, 0.0,0.0, 3.0,0.0,
};

static void rowpairs_to_colmajor_c128(const f64* pairs, c128* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c128));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = CMPLX(pairs[2 * (i * n + j)],
                                      pairs[2 * (i * n + j) + 1]);
}

void zget38(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 EPSIN = 5.9605e-8;
    const c128 CZERO = CMPLX(0.0, 0.0);

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    eps = fmax(eps, EPSIN);
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

    f64 val[3];
    val[0] = sqrt(smlnum);
    val[1] = ONE;
    val[2] = sqrt(sqrt(bignum));

    INT select[LDT], ipnt[LDT];
    f64 rwork[LDT], wsrt[LDT], result[2];
    c128 q[LDT * LDT], qsav[LDT * LDT], qtmp[LDT * LDT];
    c128 t[LDT * LDT], tmp[LDT * LDT], tsav[LDT * LDT];
    c128 tsav1[LDT * LDT], ttmp[LDT * LDT];
    c128 w[LDT], wtmp[LDT];
    c128 work[LWORK];

    const f64* dptr = zget38_data;

    for (INT ic = 0; ic < NCASES38; ic++) {
        INT n = zget38_meta[ic].n;
        INT ndim = zget38_meta[ic].ndim;
        INT isrt = zget38_meta[ic].isrt;
        f64 sin_val = zget38_meta[ic].sin_val;
        f64 sepin = zget38_meta[ic].sepin;

        rowpairs_to_colmajor_c128(dptr, tmp, n, LDT);
        dptr += 2 * n * n;

        f64 tnrm = zlange("M", n, n, tmp, LDT, rwork);

        for (INT iscl = 0; iscl < 3; iscl++) {

            (*knt)++;
            zlacpy("F", n, n, tmp, LDT, t, LDT);
            f64 vmul = val[iscl];
            for (INT i = 0; i < n; i++)
                cblas_zdscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;
            zlacpy("F", n, n, t, LDT, tsav, LDT);

            INT info;
            zgehrd(n, 0, n - 1, t, LDT, work, &work[n], LWORK - n, &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0]++;
                continue;
            }

            zlacpy("L", n, n, t, LDT, q, LDT);
            zunghr(n, 0, n - 1, q, LDT, work, &work[n], LWORK - n, &info);

            for (INT j = 0; j < n - 2; j++)
                for (INT i = j + 2; i < n; i++)
                    t[i + j * LDT] = CZERO;

            zhseqr("S", "V", n, 0, n - 1, t, LDT, w, q, LDT,
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
                    wsrt[i] = creal(w[i]);
            } else {
                for (INT i = 0; i < n; i++)
                    wsrt[i] = cimag(w[i]);
            }
            for (INT i = 0; i < n - 1; i++) {
                INT kmin = i;
                f64 vmin = wsrt[i];
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

            zlacpy("F", n, n, q, LDT, qsav, LDT);
            zlacpy("F", n, n, t, LDT, tsav1, LDT);
            INT m;
            f64 s, sep;
            ztrsen("B", "V", select, n, t, LDT, q, LDT, wtmp,
                   &m, &s, &sep, work, LWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            f64 septmp = sep / vmul;
            f64 stmp = s;

            zhst01(n, 0, n - 1, tsav, LDT, t, LDT, q, LDT, work, LWORK,
                   rwork, result);
            f64 vmax = fmax(result[0], result[1]);
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            f64 v = fmax(TWO * (f64)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            f64 tol, tolin;
            if (v > septmp)
                tol = ONE;
            else
                tol = v / septmp;
            if (v > sepin)
                tolin = ONE;
            else
                tolin = v / sepin;
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
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
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
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

            if (sin_val <= (f64)(2 * n) * eps && stmp <= (f64)(2 * n) * eps)
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
            zlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            zlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ztrsen("E", "V", select, n, ttmp, LDT, qtmp, LDT, wtmp,
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
                    if (creal(ttmp[i + j * LDT]) != creal(t[i + j * LDT]) ||
                        cimag(ttmp[i + j * LDT]) != cimag(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (creal(qtmp[i + j * LDT]) != creal(q[i + j * LDT]) ||
                        cimag(qtmp[i + j * LDT]) != cimag(q[i + j * LDT]))
                        vmax = ONE / eps;
                }

            zlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            zlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ztrsen("V", "V", select, n, ttmp, LDT, qtmp, LDT, wtmp,
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
                    if (creal(ttmp[i + j * LDT]) != creal(t[i + j * LDT]) ||
                        cimag(ttmp[i + j * LDT]) != cimag(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (creal(qtmp[i + j * LDT]) != creal(q[i + j * LDT]) ||
                        cimag(qtmp[i + j * LDT]) != cimag(q[i + j * LDT]))
                        vmax = ONE / eps;
                }

            zlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            zlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ztrsen("E", "N", select, n, ttmp, LDT, qtmp, LDT, wtmp,
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
                    if (creal(ttmp[i + j * LDT]) != creal(t[i + j * LDT]) ||
                        cimag(ttmp[i + j * LDT]) != cimag(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (creal(qtmp[i + j * LDT]) != creal(qsav[i + j * LDT]) ||
                        cimag(qtmp[i + j * LDT]) != cimag(qsav[i + j * LDT]))
                        vmax = ONE / eps;
                }

            zlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            zlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            ztrsen("V", "N", select, n, ttmp, LDT, qtmp, LDT, wtmp,
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
                    if (creal(ttmp[i + j * LDT]) != creal(t[i + j * LDT]) ||
                        cimag(ttmp[i + j * LDT]) != cimag(t[i + j * LDT]))
                        vmax = ONE / eps;
                    if (creal(qtmp[i + j * LDT]) != creal(qsav[i + j * LDT]) ||
                        cimag(qtmp[i + j * LDT]) != cimag(qsav[i + j * LDT]))
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
