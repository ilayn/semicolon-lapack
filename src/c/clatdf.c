/**
 * @file clatdf.c
 * @brief Compute contribution to reciprocal Dif-estimate using complete pivoting LU.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

#define MAXDIM 2

/**
 * CLATDF computes the contribution to the reciprocal Dif-estimate
 * by solving for x in `Z * x = b`, where b is chosen such that the norm
 * of x is as large as possible. It is assumed that LU decomposition
 * of `Z` has been computed by CGETC2. On entry `rhs` = f holds the
 * contribution from earlier solved sub-systems, and on return `rhs` = x.
 *
 * The factorization of `Z` returned by CGETC2 has the form
 * @rst
 * .. code-block:: text
 *
 *     Z = P * L * U * Q
 * @endrst
 * where P and Q are permutation matrices. L is lower
 * triangular with unit diagonal elements and U is upper triangular.
 *
 * @param[in]     ijob   `ijob=2`: First compute an approximative null-vector e
 *                       of Z using CGECON, e is normalized and solve for
 *                       Zx = +-e - f with the sign giving the greater value
 *                       of 2-norm(x). About 5 times as expensive as Default.
 *                       `ijob!=2`: Local look ahead strategy where all entries of
 *                       the r.h.s. b is chosen as either +1 or -1 (Default).
 * @param[in]     n      The number of columns of the matrix `Z`.
 * @param[in]     Z      Array of dimension (`ldz`, `n`).
 *                       The LU part of the factorization of the n-by-n
 *                       matrix `Z` computed by CGETC2: Z = P * L * U * Q
 * @param[in]     ldz    The leading dimension of the array `Z`. `ldz>=max(1,n)`.
 * @param[in,out] rhs    Array of dimension `n`.
 *                       On entry, `rhs` contains contributions from other
 *                       subsystems. On exit, `rhs` contains the solution of the
 *                       subsystem with entries according to the value of `ijob`.
 * @param[in,out] rdsum  On entry, the sum of squares of computed contributions to
 *                       the Dif-estimate under computation by CTGSYL, where the
 *                       scaling factor `rdscal` has been factored out.
 *                       On exit, the corresponding sum of squares updated with the
 *                       contributions from the current sub-system.
 * @param[in,out] rdscal On entry, scaling factor used to prevent overflow in
 *                       `rdsum`. On exit, `rdscal` is updated w.r.t. the current
 *                       contributions in `rdsum`.
 * @param[in]     ipiv   Array of dimension `n`.
 *                       The pivot indices; for `0<=i<n`, row i of the
 *                       matrix has been interchanged with row `ipiv[i]`.
 * @param[in]     jpiv   Array of dimension `n`.
 *                       The pivot indices; for `0<=j<n`, column j of the
 *                       matrix has been interchanged with column `jpiv[j]`.
 *
 * @par Further Details:
 * @rst
 * This routine is a further developed implementation of algorithm BSOLVE in
 * [1] using complete pivoting in the LU factorization.
 * @endrst
 *
 * @par Contributors:
 * @rst
 * | Bo Kagstrom and Peter Poromaa, Department of Computing Science,
 *   Umea University, S-901 87 Umea, Sweden.
 * @endrst
 *
 * @par References:
 * @rst
 * [1] Bo Kagstrom and Lars Westin, Generalized Schur Methods with Condition
 * Estimators for Solving the Generalized Sylvester Equation, IEEE
 * Transactions on Automatic Control, Vol. 34, No. 7, July 1989, pp 745-751.
 * https://doi.org/10.1109/9.29404
 *
 * [2] Peter Poromaa, On Efficient and Robust Estimators for the Separation
 * between two Regular Matrix Pairs with Applications in Condition
 * Estimation. Report IMINF-95.05, Departement of Computing Science,
 * Umea University, S-901 87 Umea, Sweden, 1995.
 * @endrst
 */
void clatdf(
    const INT ijob,
    const INT n,
    const c64* restrict Z,
    const INT ldz,
    c64* restrict rhs,
    f32* rdsum,
    f32* rdscal,
    const INT* restrict ipiv,
    const INT* restrict jpiv)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, k, info;
    f32 rtemp, scale, sminu, splus;
    c64 bm, bp, pmone, temp;

    f32 rwork[MAXDIM];
    c64 work[4 * MAXDIM];
    c64 xm[MAXDIM];
    c64 xp[MAXDIM];

    if (ijob != 2) {

        /* Apply permutations IPIV to RHS */
        for (i = 0; i < n - 1; i++) {
            if (ipiv[i] != i) {
                temp = rhs[i];
                rhs[i] = rhs[ipiv[i]];
                rhs[ipiv[i]] = temp;
            }
        }

        /* Solve for L-part choosing RHS either to +1 or -1. */
        pmone = -CONE;

        for (j = 0; j < n - 1; j++) {
            bp = rhs[j] + CONE;
            bm = rhs[j] - CONE;
            splus = ONE;

            /*
             * Look-ahead for L-part RHS(1:N-1) = + or -1, SPLUS and
             * SMIN computed more efficiently than in BSOLVE [1].
             */
            c64 dotresult;
            cblas_cdotc_sub(n - j - 1, &Z[j + 1 + j * ldz], 1,
                            &Z[j + 1 + j * ldz], 1, &dotresult);
            splus = splus + crealf(dotresult);
            cblas_cdotc_sub(n - j - 1, &Z[j + 1 + j * ldz], 1,
                            &rhs[j + 1], 1, &dotresult);
            sminu = crealf(dotresult);
            splus = splus * crealf(rhs[j]);

            if (splus > sminu) {
                rhs[j] = bp;
            } else if (sminu > splus) {
                rhs[j] = bm;
            } else {
                /*
                 * In this case the updating sums are equal and we can
                 * choose RHS(J) +1 or -1. The first time this happens
                 * we choose -1, thereafter +1. This is a simple way to
                 * get good estimates of matrices like Byers well-known
                 * example (see [1]). (Not done in BSOLVE.)
                 */
                rhs[j] = rhs[j] + pmone;
                pmone = CONE;
            }

            /* Compute the remaining r.h.s. */
            temp = -rhs[j];
            cblas_caxpy(n - j - 1, &temp, &Z[j + 1 + j * ldz], 1, &rhs[j + 1], 1);
        }

        /*
         * Solve for U-part, look-ahead for RHS(N) = +-1. This is not done
         * in BSOLVE and will hopefully give us a better estimate because
         * any ill-conditioning of the original matrix is transferred to U
         * and not to L. U(N, N) is an approximation to sigma_min(LU).
         */
        cblas_ccopy(n - 1, rhs, 1, work, 1);
        work[n - 1] = rhs[n - 1] + CONE;
        rhs[n - 1] = rhs[n - 1] - CONE;
        splus = ZERO;
        sminu = ZERO;
        for (i = n - 1; i >= 0; i--) {
            temp = CONE / Z[i + i * ldz];
            work[i] = work[i] * temp;
            rhs[i] = rhs[i] * temp;
            for (k = i + 1; k < n; k++) {
                work[i] = work[i] - work[k] * (Z[i + k * ldz] * temp);
                rhs[i] = rhs[i] - rhs[k] * (Z[i + k * ldz] * temp);
            }
            splus = splus + cabsf(work[i]);
            sminu = sminu + cabsf(rhs[i]);
        }
        if (splus > sminu) {
            cblas_ccopy(n, work, 1, rhs, 1);
        }

        /* Apply the permutations JPIV to the computed solution (RHS) */
        for (i = n - 2; i >= 0; i--) {
            if (jpiv[i] != i) {
                temp = rhs[i];
                rhs[i] = rhs[jpiv[i]];
                rhs[jpiv[i]] = temp;
            }
        }

        /* Compute the sum of squares */
        classq(n, rhs, 1, rdscal, rdsum);

    } else {

        /* IJOB = 2, Compute approximate nullvector XM of Z */
        cgecon("I", n, Z, ldz, ONE, &rtemp, work, rwork, &info);
        cblas_ccopy(n, &work[n], 1, xm, 1);

        /* Compute RHS */
        for (i = n - 2; i >= 0; i--) {
            if (ipiv[i] != i) {
                c64 tmp = xm[i];
                xm[i] = xm[ipiv[i]];
                xm[ipiv[i]] = tmp;
            }
        }
        c64 dotresult;
        cblas_cdotc_sub(n, xm, 1, xm, 1, &dotresult);
        temp = CONE / csqrtf(dotresult);
        cblas_cscal(n, &temp, xm, 1);
        cblas_ccopy(n, xm, 1, xp, 1);
        cblas_caxpy(n, &CONE, rhs, 1, xp, 1);
        c64 neg_cone = -CONE;
        cblas_caxpy(n, &neg_cone, xm, 1, rhs, 1);
        cgesc2(n, Z, ldz, rhs, ipiv, jpiv, &scale);
        cgesc2(n, Z, ldz, xp, ipiv, jpiv, &scale);
        if (cblas_scasum(n, xp, 1) > cblas_scasum(n, rhs, 1)) {
            cblas_ccopy(n, xp, 1, rhs, 1);
        }

        /* Compute the sum of squares */
        classq(n, rhs, 1, rdscal, rdsum);

    }
}
