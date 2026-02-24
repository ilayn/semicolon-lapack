/**
 * @file dget39.c
 * @brief DGET39 tests DLAQTR, a routine for solving real or special complex
 *        quasi upper triangular systems.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * DGET39 tests DLAQTR, a routine for solving the real or
 * special complex quasi upper triangular system
 *
 *      op(T)*p = scale*c,
 * or
 *      op(T + iB)*(p+iq) = scale*(c+id),
 *
 * in real arithmetic. T is upper quasi-triangular.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples where INFO is nonzero.
 * @param[out]    knt     Total number of examples tested.
 */
#define LDT  10
#define LDT2 20

void dget39(f64* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Hardcoded test matrix dimensions */

    static const INT idim[6] = { 4, 5, 5, 5, 5, 5 };

    /* Hardcoded test matrix values.
       Fortran DATA: IVAL(i,j,ndim) with i=1..5, j=1..5, ndim=1..6
       Stored as ival[ndim][j][i] in row-major for easy C access,
       but since Fortran stores column-major we lay it out as:
       ival[ndim * 25 + j * 5 + i] matching Fortran's column-major order. */

    /* ival[ndim][i][j] = T(i,j) for test matrix ndim.
       Derived from Fortran DATA IVAL / ... / in column-major order. */

    static const INT ival[6][5][5] = {
        /* ndim=0 (n=4): 4x4 matrix, 5th row/col unused */
        {{ 3, 1, 3, 4, 0},
         { 0, 1, 2, 3, 0},
         { 0,-1, 1, 2, 0},
         { 0, 0, 0, 2, 0},
         { 0, 0, 0, 0, 0}},
        /* ndim=1 (n=5) */
        {{ 1, 2, 3, 4, 1},
         { 0, 2, 3, 2, 1},
         { 0, 0, 4, 2, 1},
         { 0, 0, 0, 3, 1},
         { 0, 0, 0, 0, 5}},
        /* ndim=2 (n=5) */
        {{ 1, 2, 3, 4, 1},
         { 0, 4, 3, 2, 1},
         { 0,-2, 4, 2, 1},
         { 0, 0, 0, 3, 1},
         { 0, 0, 0, 0, 1}},
        /* ndim=3 (n=5) */
        {{ 1, 2, 9, 4, 2},
         { 0, 1, 8, 9, 2},
         { 0,-1, 1, 1, 2},
         { 0, 0, 0, 2, 2},
         { 0, 0, 0,-1, 2}},
        /* ndim=4 (n=5) */
        {{ 9, 6, 3, 5, 2},
         { 0, 4, 2, 1, 2},
         { 0, 0, 1,-1, 2},
         { 0, 0, 1, 1, 2},
         { 0, 0, 0, 0, 2}},
        /* ndim=5 (n=5) */
        {{ 4, 2, 1, 2, 2},
         { 0, 2, 4, 4, 2},
         { 0, 0, 4, 2, 2},
         { 0, 0, 0, 2, 2},
         { 0, 0, 0,-1, 2}}
    };

    /* Get machine parameters */

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S");
    f64 bignum = ONE / smlnum;

    /* Set up test case parameters */

    f64 vm1[5], vm2[5], vm3[5], vm4[5], vm5[3];

    vm1[0] = ONE;
    vm1[1] = sqrt(smlnum);
    vm1[2] = sqrt(vm1[1]);
    vm1[3] = sqrt(bignum);
    vm1[4] = sqrt(vm1[3]);

    vm2[0] = ONE;
    vm2[1] = sqrt(smlnum);
    vm2[2] = sqrt(vm2[1]);
    vm2[3] = sqrt(bignum);
    vm2[4] = sqrt(vm2[3]);

    vm3[0] = ONE;
    vm3[1] = sqrt(smlnum);
    vm3[2] = sqrt(vm3[1]);
    vm3[3] = sqrt(bignum);
    vm3[4] = sqrt(vm3[3]);

    vm4[0] = ONE;
    vm4[1] = sqrt(smlnum);
    vm4[2] = sqrt(vm4[1]);
    vm4[3] = sqrt(bignum);
    vm4[4] = sqrt(vm4[3]);

    vm5[0] = ONE;
    vm5[1] = eps;
    vm5[2] = sqrt(smlnum);

    /* Initialization */

    *knt = 0;
    *rmax = ZERO;
    *ninfo = 0;
    smlnum = smlnum / eps;

    /* Local arrays */

    f64 t[LDT * LDT];
    f64 b[LDT];
    f64 d[LDT2];
    f64 x[LDT2];
    f64 y[LDT2];
    f64 work[LDT];
    f64 dum[1];

    /* Begin test loop */

    for (INT ivm5 = 0; ivm5 < 3; ivm5++) {
        for (INT ivm4 = 0; ivm4 < 5; ivm4++) {
            for (INT ivm3 = 0; ivm3 < 5; ivm3++) {
                for (INT ivm2 = 0; ivm2 < 5; ivm2++) {
                    for (INT ivm1 = 0; ivm1 < 5; ivm1++) {
                        for (INT ndim = 0; ndim < 6; ndim++) {

                            INT n = idim[ndim];
                            for (INT i = 0; i < n; i++) {
                                for (INT j = 0; j < n; j++) {
                                    t[i + j * LDT] = (f64)ival[ndim][i][j] * vm1[ivm1];
                                    /* Fortran: IF(I.GE.J) T(I,J) = T(I,J)*VM5(IVM5) */
                                    if (i >= j)
                                        t[i + j * LDT] *= vm5[ivm5];
                                }
                            }

                            f64 w = ONE * vm2[ivm2];

                            for (INT i = 0; i < n; i++) {
                                b[i] = cos((f64)(i + 1)) * vm3[ivm3];
                            }

                            for (INT i = 0; i < 2 * n; i++) {
                                d[i] = sin((f64)(i + 1)) * vm4[ivm4];
                            }

                            f64 norm = dlange("1", n, n, t, LDT, work);
                            INT k = cblas_idamax(n, b, 1);
                            f64 normtb = norm + fabs(b[k]) + fabs(w);

                            f64 scale;
                            INT info;
                            f64 xnorm, resid, domin;

                            /* Test 1: T*x = scale*d (no transpose, real) */

                            cblas_dcopy(n, d, 1, x, 1);
                            (*knt)++;
                            dlaqtr(0, 1, n, t, LDT, dum, 0.0,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            /* || T*x - scale*d || /
                               max(ulp*||T||*||x||,smlnum/ulp*||T||,smlnum) */

                            cblas_dcopy(n, d, 1, y, 1);
                            cblas_dgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -scale, y, 1);
                            xnorm = cblas_dasum(n, x, 1);
                            resid = cblas_dasum(n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * norm > domin) domin = (smlnum / eps) * norm;
                            if ((norm * eps) * xnorm > domin) domin = (norm * eps) * xnorm;
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                            /* Test 2: T'*x = scale*d (transpose, real) */

                            cblas_dcopy(n, d, 1, x, 1);
                            (*knt)++;
                            dlaqtr(1, 1, n, t, LDT, dum, 0.0,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            cblas_dcopy(n, d, 1, y, 1);
                            cblas_dgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -scale, y, 1);
                            xnorm = cblas_dasum(n, x, 1);
                            resid = cblas_dasum(n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * norm > domin) domin = (smlnum / eps) * norm;
                            if ((norm * eps) * xnorm > domin) domin = (norm * eps) * xnorm;
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                            /* Test 3: (T+i*B)*(x1+i*x2) = scale*(d1+i*d2)
                               (no transpose, complex) */

                            cblas_dcopy(2 * n, d, 1, x, 1);
                            (*knt)++;
                            dlaqtr(0, 0, n, t, LDT, b, w,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            /* ||(T+i*B)*(x1+i*x2) - scale*(d1+i*d2)|| /
                               max(ulp*(||T||+||B||)*(||x1||+||x2||),
                                       smlnum/ulp * (||T||+||B||), smlnum) */

                            cblas_dcopy(2 * n, d, 1, y, 1);
                            y[0] = cblas_ddot(n, b, 1, x + n, 1) + scale * y[0];
                            for (INT i = 1; i < n; i++) {
                                y[i] = w * x[i + n] + scale * y[i];
                            }
                            cblas_dgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -ONE, y, 1);

                            y[n] = cblas_ddot(n, b, 1, x, 1) - scale * y[n];
                            for (INT i = 1; i < n; i++) {
                                y[i + n] = w * x[i] - scale * y[i + n];
                            }
                            cblas_dgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x + n, 1, ONE, y + n, 1);

                            resid = cblas_dasum(2 * n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * normtb > domin) domin = (smlnum / eps) * normtb;
                            if (eps * (normtb * cblas_dasum(2 * n, x, 1)) > domin)
                                domin = eps * (normtb * cblas_dasum(2 * n, x, 1));
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                            /* Test 4: (T+i*B)'*(x1+i*x2) = scale*(d1+i*d2)
                               (transpose, complex) */

                            cblas_dcopy(2 * n, d, 1, x, 1);
                            (*knt)++;
                            dlaqtr(1, 0, n, t, LDT, b, w,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            cblas_dcopy(2 * n, d, 1, y, 1);
                            y[0] = b[0] * x[n] - scale * y[0];
                            for (INT i = 1; i < n; i++) {
                                y[i] = b[i] * x[n] + w * x[i + n] - scale * y[i];
                            }
                            cblas_dgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, ONE, y, 1);

                            y[n] = b[0] * x[0] + scale * y[n];
                            for (INT i = 1; i < n; i++) {
                                y[i + n] = b[i] * x[0] + w * x[i] + scale * y[i + n];
                            }
                            cblas_dgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x + n, 1, -ONE, y + n, 1);

                            resid = cblas_dasum(2 * n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * normtb > domin) domin = (smlnum / eps) * normtb;
                            if (eps * (normtb * cblas_dasum(2 * n, x, 1)) > domin)
                                domin = eps * (normtb * cblas_dasum(2 * n, x, 1));
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                        }
                    }
                }
            }
        }
    }
}
