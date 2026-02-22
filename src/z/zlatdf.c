/**
 * @file zlatdf.c
 * @brief Compute contribution to reciprocal Dif-estimate using complete pivoting LU.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

#define MAXDIM 2

/**
 * ZLATDF computes the contribution to the reciprocal Dif-estimate
 * by solving for x in Z * x = b, where b is chosen such that the norm
 * of x is as large as possible. It is assumed that LU decomposition
 * of Z has been computed by ZGETC2. On entry RHS = f holds the
 * contribution from earlier solved sub-systems, and on return RHS = x.
 *
 * The factorization of Z returned by ZGETC2 has the form
 * Z = P * L * U * Q, where P and Q are permutation matrices. L is lower
 * triangular with unit diagonal elements and U is upper triangular.
 *
 * @param[in]     ijob   IJOB = 2: First compute an approximative null-vector e
 *                       of Z using ZGECON, e is normalized and solve for
 *                       Zx = +-e - f with the sign giving the greater value
 *                       of 2-norm(x). About 5 times as expensive as Default.
 *                       IJOB != 2: Local look ahead strategy where all entries of
 *                       the r.h.s. b is chosen as either +1 or -1 (Default).
 * @param[in]     n      The number of columns of the matrix Z.
 * @param[in]     Z      The LU part of the factorization of the n-by-n
 *                       matrix Z computed by ZGETC2: Z = P * L * U * Q
 *                       Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= max(1, n).
 * @param[in,out] rhs    On entry, RHS contains contributions from other subsystems.
 *                       On exit, RHS contains the solution of the subsystem with
 *                       entries according to the value of IJOB.
 *                       Array of dimension n.
 * @param[in,out] rdsum  On entry, the sum of squares of computed contributions to
 *                       the Dif-estimate under computation by ZTGSYL, where the
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
void zlatdf(
    const INT ijob,
    const INT n,
    const c128* restrict Z,
    const INT ldz,
    c128* restrict rhs,
    f64* rdsum,
    f64* rdscal,
    const INT* restrict ipiv,
    const INT* restrict jpiv)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j, k, info;
    f64 rtemp, scale, sminu, splus;
    c128 bm, bp, pmone, temp;

    f64 rwork[MAXDIM];
    c128 work[4 * MAXDIM];
    c128 xm[MAXDIM];
    c128 xp[MAXDIM];

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
            c128 dotresult;
            cblas_zdotc_sub(n - j - 1, &Z[j + 1 + j * ldz], 1,
                            &Z[j + 1 + j * ldz], 1, &dotresult);
            splus = splus + creal(dotresult);
            cblas_zdotc_sub(n - j - 1, &Z[j + 1 + j * ldz], 1,
                            &rhs[j + 1], 1, &dotresult);
            sminu = creal(dotresult);
            splus = splus * creal(rhs[j]);

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
            cblas_zaxpy(n - j - 1, &temp, &Z[j + 1 + j * ldz], 1, &rhs[j + 1], 1);
        }

        /*
         * Solve for U-part, look-ahead for RHS(N) = +-1. This is not done
         * in BSOLVE and will hopefully give us a better estimate because
         * any ill-conditioning of the original matrix is transferred to U
         * and not to L. U(N, N) is an approximation to sigma_min(LU).
         */
        cblas_zcopy(n - 1, rhs, 1, work, 1);
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
            splus = splus + cabs(work[i]);
            sminu = sminu + cabs(rhs[i]);
        }
        if (splus > sminu) {
            cblas_zcopy(n, work, 1, rhs, 1);
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
        zlassq(n, rhs, 1, rdscal, rdsum);

    } else {

        /* IJOB = 2, Compute approximate nullvector XM of Z */
        zgecon("I", n, Z, ldz, ONE, &rtemp, work, rwork, &info);
        cblas_zcopy(n, &work[n], 1, xm, 1);

        /* Compute RHS */
        for (i = n - 2; i >= 0; i--) {
            if (ipiv[i] != i) {
                c128 tmp = xm[i];
                xm[i] = xm[ipiv[i]];
                xm[ipiv[i]] = tmp;
            }
        }
        c128 dotresult;
        cblas_zdotc_sub(n, xm, 1, xm, 1, &dotresult);
        temp = CONE / csqrt(dotresult);
        cblas_zscal(n, &temp, xm, 1);
        cblas_zcopy(n, xm, 1, xp, 1);
        cblas_zaxpy(n, &CONE, rhs, 1, xp, 1);
        c128 neg_cone = -CONE;
        cblas_zaxpy(n, &neg_cone, xm, 1, rhs, 1);
        zgesc2(n, Z, ldz, rhs, ipiv, jpiv, &scale);
        zgesc2(n, Z, ldz, xp, ipiv, jpiv, &scale);
        if (cblas_dzasum(n, xp, 1) > cblas_dzasum(n, rhs, 1)) {
            cblas_zcopy(n, xp, 1, rhs, 1);
        }

        /* Compute the sum of squares */
        zlassq(n, rhs, 1, rdscal, rdsum);

    }
}
