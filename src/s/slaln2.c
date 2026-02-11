/**
 * @file slaln2.c
 * @brief SLALN2 solves a 1-by-1 or 2-by-2 linear system of equations.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include <float.h>

/**
 * SLALN2 solves a system of the form  (ca A - w D ) X = s B
 * or (ca A**T - w D) X = s B   with possible scaling ("s") and
 * perturbation of A.  (A**T means A-transpose.)
 *
 * A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
 * real diagonal matrix, w is a real or complex value, and X and B are
 * NA x 1 matrices -- real if w is real, complex if w is complex.  NA
 * may be 1 or 2.
 *
 * If w is complex, X and B are represented as NA x 2 matrices,
 * the first column of each being the real part and the second
 * being the imaginary part.
 *
 * "s" is a scaling factor (<= 1), computed by SLALN2, which is
 * so chosen that X can be computed without overflow.  X is further
 * scaled if necessary to assure that norm(ca A - w D)*norm(X) is less
 * than overflow.
 *
 * If both singular values of (ca A - w D) are less than SMIN,
 * SMIN*identity will be used instead of (ca A - w D).  If only one
 * singular value is less than SMIN, one element of (ca A - w D) will be
 * perturbed enough to make the smallest singular value roughly SMIN.
 * If both singular values are at least SMIN, (ca A - w D) will not be
 * perturbed.  In any case, the perturbation will be at most some small
 * multiple of max( SMIN, ulp*norm(ca A - w D) ).  The singular values
 * are computed by infinity-norm approximations, and thus will only be
 * correct to a factor of 2 or so.
 *
 * Note: all input quantities are assumed to be smaller than overflow
 * by a reasonable factor.  (See BIGNUM.)
 *
 * @param[in] ltrans  If nonzero, A-transpose will be used. If zero, A will be used.
 * @param[in] na      The size of the matrix A. It may (only) be 1 or 2.
 * @param[in] nw      1 if "w" is real, 2 if "w" is complex. It may only be 1 or 2.
 * @param[in] smin    The desired lower bound on the singular values of A.
 * @param[in] ca      The coefficient c, which A is multiplied by.
 * @param[in] A       Double precision array, dimension (lda, na). The NA x NA matrix A.
 * @param[in] lda     The leading dimension of A. lda >= na.
 * @param[in] d1      The 1,1 element in the diagonal matrix D.
 * @param[in] d2      The 2,2 element in the diagonal matrix D. Not used if na=1.
 * @param[in] B       Double precision array, dimension (ldb, nw). The NA x NW matrix B.
 * @param[in] ldb     The leading dimension of B. ldb >= na.
 * @param[in] wr      The real part of the scalar "w".
 * @param[in] wi      The imaginary part of the scalar "w". Not used if nw=1.
 * @param[out] X      Double precision array, dimension (ldx, nw). The NA x NW solution matrix X.
 * @param[in] ldx     The leading dimension of X. ldx >= na.
 * @param[out] scale  The scale factor that B must be multiplied by. At most 1.
 * @param[out] xnorm  The infinity-norm of X.
 * @param[out] info
 *                         - = 0: No error, (ca A - w D) did not have to be perturbed.
 *                         - = 1: (ca A - w D) had to be perturbed.
 */
void slaln2(const int ltrans, const int na, const int nw, const float smin,
            const float ca, const float* A, const int lda,
            const float d1, const float d2,
            const float* B, const int ldb,
            const float wr, const float wi,
            float* X, const int ldx,
            float* scale, float* xnorm, int* info)
{
    /* Local scalars */
    int icmax, j;
    float bbnd, bi1, bi2, bignum, bnorm, br1, br2, ci21;
    float ci22, cmax, cnorm, cr21, cr22, csi, csr, li21;
    float lr21, smini, smlnum, temp, u22abs, ui11, ui11r;
    float ui12, ui12s, ui22, ur11, ur11r, ur12, ur12s;
    float ur22, xi1, xi2, xr1, xr2;

    /* Local arrays - CR and CI are 2x2, stored column-major
     * CRV and CIV are linear views of CR and CI respectively
     * CR(i,j) = CRV[i + j*2] for 0-based, or CRV[(i-1) + (j-1)*2] for 1-based */
    float cr[4], ci[4];  /* Column-major: cr[0]=cr(1,1), cr[1]=cr(2,1), cr[2]=cr(1,2), cr[3]=cr(2,2) */

    /* Pivot tables (converted from 1-based Fortran to 0-based C) */
    /* ZSWAP: whether to swap rows in final result */
    static const int zswap[4] = {0, 0, 1, 1};
    /* RSWAP: whether to swap B rows before solving */
    static const int rswap[4] = {0, 1, 0, 1};
    /* IPIVOT: pivot selection table (Fortran: IPIVOT(4,4), stored column-major)
     * Original Fortran DATA: 1,2,3,4, 2,1,4,3, 3,4,1,2, 4,3,2,1
     * Converted to 0-based: 0,1,2,3, 1,0,3,2, 2,3,0,1, 3,2,1,0 */
    static const int ipivot[4][4] = {
        {0, 1, 2, 3},  /* column 0 (icmax=0) */
        {1, 0, 3, 2},  /* column 1 (icmax=1) */
        {2, 3, 0, 1},  /* column 2 (icmax=2) */
        {3, 2, 1, 0}   /* column 3 (icmax=3) */
    };

    /* Compute BIGNUM */
    smlnum = 2.0f * FLT_MIN;
    bignum = 1.0f / smlnum;
    smini = (smin > smlnum) ? smin : smlnum;

    /* Don't check for input errors */
    *info = 0;

    /* Standard Initializations */
    *scale = 1.0f;

    if (na == 1) {
        /* 1 x 1 (i.e., scalar) system   C X = B */

        if (nw == 1) {
            /* Real 1x1 system.
             * C = ca A - w D */
            csr = ca * A[0] - wr * d1;
            cnorm = fabsf(csr);

            /* If | C | < SMINI, use C = SMINI */
            if (cnorm < smini) {
                csr = smini;
                cnorm = smini;
                *info = 1;
            }

            /* Check scaling for  X = B / C */
            bnorm = fabsf(B[0]);
            if (cnorm < 1.0f && bnorm > 1.0f) {
                if (bnorm > bignum * cnorm) {
                    *scale = 1.0f / bnorm;
                }
            }

            /* Compute X */
            X[0] = (B[0] * (*scale)) / csr;
            *xnorm = fabsf(X[0]);

        } else {
            /* Complex 1x1 system (w is complex)
             * C = ca A - w D */
            csr = ca * A[0] - wr * d1;
            csi = -wi * d1;
            cnorm = fabsf(csr) + fabsf(csi);

            /* If | C | < SMINI, use C = SMINI */
            if (cnorm < smini) {
                csr = smini;
                csi = 0.0f;
                cnorm = smini;
                *info = 1;
            }

            /* Check scaling for  X = B / C */
            bnorm = fabsf(B[0]) + fabsf(B[ldb]);
            if (cnorm < 1.0f && bnorm > 1.0f) {
                if (bnorm > bignum * cnorm) {
                    *scale = 1.0f / bnorm;
                }
            }

            /* Compute X using complex division */
            sladiv((*scale) * B[0], (*scale) * B[ldb], csr, csi, &X[0], &X[ldx]);
            *xnorm = fabsf(X[0]) + fabsf(X[ldx]);
        }

    } else {
        /* 2x2 System
         *
         * Compute the real part of  C = ca A - w D  (or  ca A**T - w D )
         * CR is stored column-major: cr[0]=CR(1,1), cr[1]=CR(2,1), cr[2]=CR(1,2), cr[3]=CR(2,2) */
        cr[0] = ca * A[0] - wr * d1;           /* CR(1,1) */
        cr[3] = ca * A[1 + lda] - wr * d2;     /* CR(2,2) */
        if (ltrans) {
            cr[2] = ca * A[1];                  /* CR(1,2) = ca*A(2,1) */
            cr[1] = ca * A[lda];                /* CR(2,1) = ca*A(1,2) */
        } else {
            cr[1] = ca * A[1];                  /* CR(2,1) = ca*A(2,1) */
            cr[2] = ca * A[lda];                /* CR(1,2) = ca*A(1,2) */
        }

        if (nw == 1) {
            /* Real 2x2 system  (w is real)
             *
             * Find the largest element in C */
            cmax = 0.0f;
            icmax = -1;

            for (j = 0; j < 4; j++) {
                if (fabsf(cr[j]) > cmax) {
                    cmax = fabsf(cr[j]);
                    icmax = j;
                }
            }

            /* If norm(C) < SMINI, use SMINI*identity. */
            if (cmax < smini) {
                bnorm = fabsf(B[0]);
                temp = fabsf(B[1]);
                if (temp > bnorm) bnorm = temp;

                if (smini < 1.0f && bnorm > 1.0f) {
                    if (bnorm > bignum * smini) {
                        *scale = 1.0f / bnorm;
                    }
                }
                temp = (*scale) / smini;
                X[0] = temp * B[0];
                X[1] = temp * B[1];
                *xnorm = temp * bnorm;
                *info = 1;
                return;
            }

            /* Gaussian elimination with complete pivoting. */
            ur11 = cr[icmax];
            cr21 = cr[ipivot[icmax][1]];
            ur12 = cr[ipivot[icmax][2]];
            cr22 = cr[ipivot[icmax][3]];
            ur11r = 1.0f / ur11;
            lr21 = ur11r * cr21;
            ur22 = cr22 - ur12 * lr21;

            /* If smaller pivot < SMINI, use SMINI */
            if (fabsf(ur22) < smini) {
                ur22 = smini;
                *info = 1;
            }

            if (rswap[icmax]) {
                br1 = B[1];
                br2 = B[0];
            } else {
                br1 = B[0];
                br2 = B[1];
            }
            br2 = br2 - lr21 * br1;

            bbnd = fabsf(br1 * (ur22 * ur11r));
            temp = fabsf(br2);
            if (temp > bbnd) bbnd = temp;

            if (bbnd > 1.0f && fabsf(ur22) < 1.0f) {
                if (bbnd >= bignum * fabsf(ur22)) {
                    *scale = 1.0f / bbnd;
                }
            }

            xr2 = (br2 * (*scale)) / ur22;
            xr1 = ((*scale) * br1) * ur11r - xr2 * (ur11r * ur12);

            if (zswap[icmax]) {
                X[0] = xr2;
                X[1] = xr1;
            } else {
                X[0] = xr1;
                X[1] = xr2;
            }
            *xnorm = fabsf(xr1);
            temp = fabsf(xr2);
            if (temp > *xnorm) *xnorm = temp;

            /* Further scaling if  norm(A) norm(X) > overflow */
            if (*xnorm > 1.0f && cmax > 1.0f) {
                if (*xnorm > bignum / cmax) {
                    temp = cmax / bignum;
                    X[0] = temp * X[0];
                    X[1] = temp * X[1];
                    *xnorm = temp * (*xnorm);
                    *scale = temp * (*scale);
                }
            }

        } else {
            /* Complex 2x2 system  (w is complex)
             *
             * Find the largest element in C
             * CI is stored column-major like CR */
            ci[0] = -wi * d1;    /* CI(1,1) */
            ci[1] = 0.0f;         /* CI(2,1) */
            ci[2] = 0.0f;         /* CI(1,2) */
            ci[3] = -wi * d2;    /* CI(2,2) */

            cmax = 0.0f;
            icmax = -1;

            for (j = 0; j < 4; j++) {
                if (fabsf(cr[j]) + fabsf(ci[j]) > cmax) {
                    cmax = fabsf(cr[j]) + fabsf(ci[j]);
                    icmax = j;
                }
            }

            /* If norm(C) < SMINI, use SMINI*identity. */
            if (cmax < smini) {
                bnorm = fabsf(B[0]) + fabsf(B[ldb]);
                temp = fabsf(B[1]) + fabsf(B[1 + ldb]);
                if (temp > bnorm) bnorm = temp;

                if (smini < 1.0f && bnorm > 1.0f) {
                    if (bnorm > bignum * smini) {
                        *scale = 1.0f / bnorm;
                    }
                }
                temp = (*scale) / smini;
                X[0] = temp * B[0];
                X[1] = temp * B[1];
                X[ldx] = temp * B[ldb];
                X[1 + ldx] = temp * B[1 + ldb];
                *xnorm = temp * bnorm;
                *info = 1;
                return;
            }

            /* Gaussian elimination with complete pivoting. */
            ur11 = cr[icmax];
            ui11 = ci[icmax];
            cr21 = cr[ipivot[icmax][1]];
            ci21 = ci[ipivot[icmax][1]];
            ur12 = cr[ipivot[icmax][2]];
            ui12 = ci[ipivot[icmax][2]];
            cr22 = cr[ipivot[icmax][3]];
            ci22 = ci[ipivot[icmax][3]];

            if (icmax == 0 || icmax == 3) {
                /* Code when off-diagonals of pivoted C are real */
                if (fabsf(ur11) > fabsf(ui11)) {
                    temp = ui11 / ur11;
                    ur11r = 1.0f / (ur11 * (1.0f + temp * temp));
                    ui11r = -temp * ur11r;
                } else {
                    temp = ur11 / ui11;
                    ui11r = -1.0f / (ui11 * (1.0f + temp * temp));
                    ur11r = -temp * ui11r;
                }
                lr21 = cr21 * ur11r;
                li21 = cr21 * ui11r;
                ur12s = ur12 * ur11r;
                ui12s = ur12 * ui11r;
                ur22 = cr22 - ur12 * lr21;
                ui22 = ci22 - ur12 * li21;
            } else {
                /* Code when diagonals of pivoted C are real */
                ur11r = 1.0f / ur11;
                ui11r = 0.0f;
                lr21 = cr21 * ur11r;
                li21 = ci21 * ur11r;
                ur12s = ur12 * ur11r;
                ui12s = ui12 * ur11r;
                ur22 = cr22 - ur12 * lr21 + ui12 * li21;
                ui22 = -ur12 * li21 - ui12 * lr21;
            }
            u22abs = fabsf(ur22) + fabsf(ui22);

            /* If smaller pivot < SMINI, use SMINI */
            if (u22abs < smini) {
                ur22 = smini;
                ui22 = 0.0f;
                *info = 1;
            }

            if (rswap[icmax]) {
                br2 = B[0];
                br1 = B[1];
                bi2 = B[ldb];
                bi1 = B[1 + ldb];
            } else {
                br1 = B[0];
                br2 = B[1];
                bi1 = B[ldb];
                bi2 = B[1 + ldb];
            }
            br2 = br2 - lr21 * br1 + li21 * bi1;
            bi2 = bi2 - li21 * br1 - lr21 * bi1;

            bbnd = (fabsf(br1) + fabsf(bi1)) * (u22abs * (fabsf(ur11r) + fabsf(ui11r)));
            temp = fabsf(br2) + fabsf(bi2);
            if (temp > bbnd) bbnd = temp;

            if (bbnd > 1.0f && u22abs < 1.0f) {
                if (bbnd >= bignum * u22abs) {
                    *scale = 1.0f / bbnd;
                    br1 = (*scale) * br1;
                    bi1 = (*scale) * bi1;
                    br2 = (*scale) * br2;
                    bi2 = (*scale) * bi2;
                }
            }

            sladiv(br2, bi2, ur22, ui22, &xr2, &xi2);
            xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
            xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;

            if (zswap[icmax]) {
                X[0] = xr2;
                X[1] = xr1;
                X[ldx] = xi2;
                X[1 + ldx] = xi1;
            } else {
                X[0] = xr1;
                X[1] = xr2;
                X[ldx] = xi1;
                X[1 + ldx] = xi2;
            }
            *xnorm = fabsf(xr1) + fabsf(xi1);
            temp = fabsf(xr2) + fabsf(xi2);
            if (temp > *xnorm) *xnorm = temp;

            /* Further scaling if  norm(A) norm(X) > overflow */
            if (*xnorm > 1.0f && cmax > 1.0f) {
                if (*xnorm > bignum / cmax) {
                    temp = cmax / bignum;
                    X[0] = temp * X[0];
                    X[1] = temp * X[1];
                    X[ldx] = temp * X[ldx];
                    X[1 + ldx] = temp * X[1 + ldx];
                    *xnorm = temp * (*xnorm);
                    *scale = temp * (*scale);
                }
            }
        }
    }
}
