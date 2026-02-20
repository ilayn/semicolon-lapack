/**
 * @file sget39.c
 * @brief SGET39 tests SLAQTR, a routine for solving real or special complex
 *        quasi upper triangular systems.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void slaqtr(const int ltran, const int lreal, const int n,
                   const f32* T, const int ldt, const f32* B, const f32 w,
                   f32* scale, f32* X, f32* work, int* info);

/**
 * SGET39 tests SLAQTR, a routine for solving the real or
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

void sget39(f32* rmax, int* lmax, int* ninfo, int* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Hardcoded test matrix dimensions */

    static const int idim[6] = { 4, 5, 5, 5, 5, 5 };

    /* Hardcoded test matrix values.
       Fortran DATA: IVAL(i,j,ndim) with i=1..5, j=1..5, ndim=1..6
       Stored as ival[ndim][j][i] in row-major for easy C access,
       but since Fortran stores column-major we lay it out as:
       ival[ndim * 25 + j * 5 + i] matching Fortran's column-major order. */

    /* ival[ndim][i][j] = T(i,j) for test matrix ndim.
       Derived from Fortran DATA IVAL / ... / in column-major order. */

    static const int ival[6][5][5] = {
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

    f32 eps = slamch("P");
    f32 smlnum = slamch("S");
    f32 bignum = ONE / smlnum;

    /* Set up test case parameters */

    f32 vm1[5], vm2[5], vm3[5], vm4[5], vm5[3];

    vm1[0] = ONE;
    vm1[1] = sqrtf(smlnum);
    vm1[2] = sqrtf(vm1[1]);
    vm1[3] = sqrtf(bignum);
    vm1[4] = sqrtf(vm1[3]);

    vm2[0] = ONE;
    vm2[1] = sqrtf(smlnum);
    vm2[2] = sqrtf(vm2[1]);
    vm2[3] = sqrtf(bignum);
    vm2[4] = sqrtf(vm2[3]);

    vm3[0] = ONE;
    vm3[1] = sqrtf(smlnum);
    vm3[2] = sqrtf(vm3[1]);
    vm3[3] = sqrtf(bignum);
    vm3[4] = sqrtf(vm3[3]);

    vm4[0] = ONE;
    vm4[1] = sqrtf(smlnum);
    vm4[2] = sqrtf(vm4[1]);
    vm4[3] = sqrtf(bignum);
    vm4[4] = sqrtf(vm4[3]);

    vm5[0] = ONE;
    vm5[1] = eps;
    vm5[2] = sqrtf(smlnum);

    /* Initialization */

    *knt = 0;
    *rmax = ZERO;
    *ninfo = 0;
    smlnum = smlnum / eps;

    /* Local arrays */

    f32 t[LDT * LDT];
    f32 b[LDT];
    f32 d[LDT2];
    f32 x[LDT2];
    f32 y[LDT2];
    f32 work[LDT];
    f32 dum[1];

    /* Begin test loop */

    for (int ivm5 = 0; ivm5 < 3; ivm5++) {
        for (int ivm4 = 0; ivm4 < 5; ivm4++) {
            for (int ivm3 = 0; ivm3 < 5; ivm3++) {
                for (int ivm2 = 0; ivm2 < 5; ivm2++) {
                    for (int ivm1 = 0; ivm1 < 5; ivm1++) {
                        for (int ndim = 0; ndim < 6; ndim++) {

                            int n = idim[ndim];
                            for (int i = 0; i < n; i++) {
                                for (int j = 0; j < n; j++) {
                                    t[i + j * LDT] = (f32)ival[ndim][i][j] * vm1[ivm1];
                                    /* Fortran: IF(I.GE.J) T(I,J) = T(I,J)*VM5(IVM5) */
                                    if (i >= j)
                                        t[i + j * LDT] *= vm5[ivm5];
                                }
                            }

                            f32 w = ONE * vm2[ivm2];

                            for (int i = 0; i < n; i++) {
                                b[i] = cosf((f32)(i + 1)) * vm3[ivm3];
                            }

                            for (int i = 0; i < 2 * n; i++) {
                                d[i] = sinf((f32)(i + 1)) * vm4[ivm4];
                            }

                            f32 norm = slange("1", n, n, t, LDT, work);
                            int k = cblas_isamax(n, b, 1);
                            f32 normtb = norm + fabsf(b[k]) + fabsf(w);

                            f32 scale;
                            int info;
                            f32 xnorm, resid, domin;

                            /* Test 1: T*x = scale*d (no transpose, real) */

                            cblas_scopy(n, d, 1, x, 1);
                            (*knt)++;
                            slaqtr(0, 1, n, t, LDT, dum, 0.0f,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            /* || T*x - scale*d || /
                               max(ulp*||T||*||x||,smlnum/ulp*||T||,smlnum) */

                            cblas_scopy(n, d, 1, y, 1);
                            cblas_sgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -scale, y, 1);
                            xnorm = cblas_sasum(n, x, 1);
                            resid = cblas_sasum(n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * norm > domin) domin = (smlnum / eps) * norm;
                            if ((norm * eps) * xnorm > domin) domin = (norm * eps) * xnorm;
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                            /* Test 2: T'*x = scale*d (transpose, real) */

                            cblas_scopy(n, d, 1, x, 1);
                            (*knt)++;
                            slaqtr(1, 1, n, t, LDT, dum, 0.0f,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            cblas_scopy(n, d, 1, y, 1);
                            cblas_sgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -scale, y, 1);
                            xnorm = cblas_sasum(n, x, 1);
                            resid = cblas_sasum(n, y, 1);
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

                            cblas_scopy(2 * n, d, 1, x, 1);
                            (*knt)++;
                            slaqtr(0, 0, n, t, LDT, b, w,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            /* ||(T+i*B)*(x1+i*x2) - scale*(d1+i*d2)|| /
                               max(ulp*(||T||+||B||)*(||x1||+||x2||),
                                       smlnum/ulp * (||T||+||B||), smlnum) */

                            cblas_scopy(2 * n, d, 1, y, 1);
                            y[0] = cblas_sdot(n, b, 1, x + n, 1) + scale * y[0];
                            for (int i = 1; i < n; i++) {
                                y[i] = w * x[i + n] + scale * y[i];
                            }
                            cblas_sgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, -ONE, y, 1);

                            y[n] = cblas_sdot(n, b, 1, x, 1) - scale * y[n];
                            for (int i = 1; i < n; i++) {
                                y[i + n] = w * x[i] - scale * y[i + n];
                            }
                            cblas_sgemv(CblasColMajor, CblasNoTrans,
                                        n, n, ONE, t, LDT,
                                        x + n, 1, ONE, y + n, 1);

                            resid = cblas_sasum(2 * n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * normtb > domin) domin = (smlnum / eps) * normtb;
                            if (eps * (normtb * cblas_sasum(2 * n, x, 1)) > domin)
                                domin = eps * (normtb * cblas_sasum(2 * n, x, 1));
                            resid = resid / domin;
                            if (resid > *rmax) {
                                *rmax = resid;
                                *lmax = *knt;
                            }

                            /* Test 4: (T+i*B)'*(x1+i*x2) = scale*(d1+i*d2)
                               (transpose, complex) */

                            cblas_scopy(2 * n, d, 1, x, 1);
                            (*knt)++;
                            slaqtr(1, 0, n, t, LDT, b, w,
                                   &scale, x, work, &info);
                            if (info != 0)
                                (*ninfo)++;

                            cblas_scopy(2 * n, d, 1, y, 1);
                            y[0] = b[0] * x[n] - scale * y[0];
                            for (int i = 1; i < n; i++) {
                                y[i] = b[i] * x[n] + w * x[i + n] - scale * y[i];
                            }
                            cblas_sgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x, 1, ONE, y, 1);

                            y[n] = b[0] * x[0] + scale * y[n];
                            for (int i = 1; i < n; i++) {
                                y[i + n] = b[i] * x[0] + w * x[i] + scale * y[i + n];
                            }
                            cblas_sgemv(CblasColMajor, CblasTrans,
                                        n, n, ONE, t, LDT,
                                        x + n, 1, -ONE, y + n, 1);

                            resid = cblas_sasum(2 * n, y, 1);
                            domin = smlnum;
                            if ((smlnum / eps) * normtb > domin) domin = (smlnum / eps) * normtb;
                            if (eps * (normtb * cblas_sasum(2 * n, x, 1)) > domin)
                                domin = eps * (normtb * cblas_sasum(2 * n, x, 1));
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
