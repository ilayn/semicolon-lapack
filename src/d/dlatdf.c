/**
 * @file dlatdf.c
 * @brief Compute contribution to reciprocal Dif-estimate using complete pivoting LU.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

#define MAXDIM 8

/**
 * DLATDF uses the LU factorization of the n-by-n matrix Z computed by
 * DGETC2 and computes a contribution to the reciprocal Dif-estimate
 * by solving Z * x = b for x, and choosing the r.h.s. b such that
 * the norm of x is as large as possible. On entry RHS = b holds the
 * contribution from earlier solved sub-systems, and on return RHS = x.
 *
 * The factorization of Z returned by DGETC2 has the form Z = P*L*U*Q,
 * where P and Q are permutation matrices. L is lower triangular with
 * unit diagonal elements and U is upper triangular.
 *
 * @param[in]     ijob   IJOB = 2: First compute an approximative null-vector e
 *                       of Z using DGECON, e is normalized and solve for
 *                       Zx = +-e - f with the sign giving the greater value
 *                       of 2-norm(x). About 5 times as expensive as Default.
 *                       IJOB != 2: Local look ahead strategy where all entries of
 *                       the r.h.s. b is chosen as either +1 or -1 (Default).
 * @param[in]     n      The number of columns of the matrix Z.
 * @param[in]     Z      The LU part of the factorization of the n-by-n
 *                       matrix Z computed by DGETC2: Z = P * L * U * Q
 *                       Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= max(1, n).
 * @param[in,out] rhs    On entry, RHS contains contributions from other subsystems.
 *                       On exit, RHS contains the solution of the subsystem with
 *                       entries according to the value of IJOB.
 *                       Array of dimension n.
 * @param[in,out] rdsum  On entry, the sum of squares of computed contributions to
 *                       the Dif-estimate under computation by DTGSYL, where the
 *                       scaling factor RDSCAL has been factored out.
 *                       On exit, the corresponding sum of squares updated with the
 *                       contributions from the current sub-system.
 * @param[in,out] rdscal On entry, scaling factor used to prevent overflow in RDSUM.
 *                       On exit, RDSCAL is updated w.r.t. the current contributions
 *                       in RDSUM.
 * @param[in]     ipiv   The pivot indices; for 0 <= i < n, row i of the
 *                       matrix has been interchanged with row ipiv[i].
 *                       Array of dimension n, 0-based.
 * @param[in]     jpiv   The pivot indices; for 0 <= j < n, column j of the
 *                       matrix has been interchanged with column jpiv[j].
 *                       Array of dimension n, 0-based.
 */
void dlatdf(
    const INT ijob,
    const INT n,
    const f64* restrict Z,
    const INT ldz,
    f64* restrict rhs,
    f64* rdsum,
    f64* rdscal,
    const INT* restrict ipiv,
    const INT* restrict jpiv)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, k, info;
    f64 bm, bp, pmone, sminu, splus, temp;

    INT iwork[MAXDIM];
    f64 work[4 * MAXDIM];
    f64 xm[MAXDIM];
    f64 xp[MAXDIM];

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
        pmone = -ONE;

        for (j = 0; j < n - 1; j++) {
            bp = rhs[j] + ONE;
            bm = rhs[j] - ONE;
            splus = ONE;

            /*
             * Look-ahead for L-part RHS(1:N-1) = + or -1, SPLUS and
             * SMIN computed more efficiently than in BSOLVE [1].
             */
            splus = splus + cblas_ddot(n - j - 1, &Z[j + 1 + j * ldz], 1,
                                       &Z[j + 1 + j * ldz], 1);
            sminu = cblas_ddot(n - j - 1, &Z[j + 1 + j * ldz], 1,
                               &rhs[j + 1], 1);
            splus = splus * rhs[j];

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
                pmone = ONE;
            }

            /* Compute the remaining r.h.s. */
            temp = -rhs[j];
            cblas_daxpy(n - j - 1, temp, &Z[j + 1 + j * ldz], 1, &rhs[j + 1], 1);
        }

        /*
         * Solve for U-part, look-ahead for RHS(N) = +-1. This is not done
         * in BSOLVE and will hopefully give us a better estimate because
         * any ill-conditioning of the original matrix is transferred to U
         * and not to L. U(N, N) is an approximation to sigma_min(LU).
         */
        cblas_dcopy(n - 1, rhs, 1, xp, 1);
        xp[n - 1] = rhs[n - 1] + ONE;
        rhs[n - 1] = rhs[n - 1] - ONE;
        splus = ZERO;
        sminu = ZERO;
        for (i = n - 1; i >= 0; i--) {
            temp = ONE / Z[i + i * ldz];
            xp[i] = xp[i] * temp;
            rhs[i] = rhs[i] * temp;
            for (k = i + 1; k < n; k++) {
                xp[i] = xp[i] - xp[k] * (Z[i + k * ldz] * temp);
                rhs[i] = rhs[i] - rhs[k] * (Z[i + k * ldz] * temp);
            }
            splus = splus + fabs(xp[i]);
            sminu = sminu + fabs(rhs[i]);
        }
        if (splus > sminu) {
            cblas_dcopy(n, xp, 1, rhs, 1);
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
        dlassq(n, rhs, 1, rdscal, rdsum);

    } else {

        /* IJOB = 2, Compute approximate nullvector XM of Z */
        dgecon("I", n, Z, ldz, ONE, &temp, work, iwork, &info);
        cblas_dcopy(n, &work[n], 1, xm, 1);

        /* Compute RHS */
        for (i = n - 2; i >= 0; i--) {
            if (ipiv[i] != i) {
                f64 tmp = xm[i];
                xm[i] = xm[ipiv[i]];
                xm[ipiv[i]] = tmp;
            }
        }
        temp = ONE / sqrt(cblas_ddot(n, xm, 1, xm, 1));
        cblas_dscal(n, temp, xm, 1);
        cblas_dcopy(n, xm, 1, xp, 1);
        cblas_daxpy(n, ONE, rhs, 1, xp, 1);
        cblas_daxpy(n, -ONE, xm, 1, rhs, 1);
        dgesc2(n, Z, ldz, rhs, ipiv, jpiv, &temp);
        dgesc2(n, Z, ldz, xp, ipiv, jpiv, &temp);
        if (cblas_dasum(n, xp, 1) > cblas_dasum(n, rhs, 1)) {
            cblas_dcopy(n, xp, 1, rhs, 1);
        }

        /* Compute the sum of squares */
        dlassq(n, rhs, 1, rdscal, rdsum);

    }
}
