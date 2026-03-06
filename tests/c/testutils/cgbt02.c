/**
 * @file cgbt02.c
 * @brief CGBT02 computes the residual for a banded system solution.
 *
 * Port of LAPACK TESTING/LIN/cgbt02.f
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGBT02 computes the residual for a solution of a banded system of
 * equations op(A)*X = B:
 *    RESID = norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS ),
 * where op(A) = A, A**T, or A**H, depending on TRANS, and EPS is the
 * machine epsilon.
 *
 * @param[in] trans  'N': A * X = B (No transpose)
 *                   'T': A**T * X = B (Transpose)
 *                   'C': A**H * X = B (Conjugate transpose)
 * @param[in] m      The number of rows of A. m >= 0.
 * @param[in] n      The number of columns of A. n >= 0.
 * @param[in] kl     The number of subdiagonals within the band. kl >= 0.
 * @param[in] ku     The number of superdiagonals within the band. ku >= 0.
 * @param[in] nrhs   The number of columns of B. nrhs >= 0.
 * @param[in] A      The original band matrix in band storage. Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, kl+ku+1).
 * @param[in] X      The computed solution vectors. Dimension (ldx, nrhs).
 * @param[in] ldx    The leading dimension of X.
 * @param[in,out] B  On entry, the right hand side vectors.
 *                   On exit, overwritten with B - op(A)*X.
 *                   Dimension (ldb, nrhs).
 * @param[in] ldb    The leading dimension of B.
 * @param[out] rwork Workspace, dimension (m) when trans='T' or 'C'.
 * @param[out] resid The maximum over NRHS of
 *                   norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS).
 */
void cgbt02(const char* trans, INT m, INT n, INT kl, INT ku, INT nrhs,
            const c64* A, INT lda,
            const c64* X, INT ldx,
            c64* B, INT ldb,
            f32* rwork,
            f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Quick return if m = 0 or n = 0 or nrhs = 0 */
    if (m <= 0 || n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    f32 eps = slamch("Epsilon");
    f32 anorm = ZERO;

    if (trans[0] == 'N' || trans[0] == 'n') {
        /* Find norm1(A). */
        INT kd = ku;
        for (INT j = 0; j < n; j++) {
            INT i1 = (kd + 1 - j - 1 > 0) ? kd + 1 - j - 1 : 0;
            INT i2_excl = (kd + m - j < kl + kd + 1) ? kd + m - j : kl + kd + 1;
            if (i2_excl > i1) {
                f32 temp = cblas_scasum(i2_excl - i1, &A[i1 + j * lda], 1);
                if (temp > anorm || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else {
        /* Find normI(A). */
        for (INT i = 0; i < m; i++) {
            rwork[i] = ZERO;
        }
        for (INT j = 0; j < n; j++) {
            INT kd = ku - j;
            for (INT i = (j - ku > 0 ? j - ku : 0); i < (j + kl + 1 < m ? j + kl + 1 : m); i++) {
                rwork[i] += cabs1f(A[(kd + i) + j * lda]);
            }
        }
        for (INT i = 0; i < m; i++) {
            f32 temp = rwork[i];
            if (temp > anorm || isnan(temp)) {
                anorm = temp;
            }
        }
    }

    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    INT n1;
    CBLAS_TRANSPOSE trans_enum;
    if (trans[0] == 'T' || trans[0] == 't') {
        n1 = n;
        trans_enum = CblasTrans;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        n1 = n;
        trans_enum = CblasConjTrans;
    } else {
        n1 = m;
        trans_enum = CblasNoTrans;
    }

    /* Compute B - op(A)*X */
    c64 cone = CMPLXF(ONE, 0.0f);
    c64 neg_cone = CMPLXF(-ONE, 0.0f);

    for (INT j = 0; j < nrhs; j++) {
        cblas_cgbmv(CblasColMajor, trans_enum,
                   m, n, kl, ku,
                   &neg_cone, A, lda,
                   &X[j * ldx], 1,
                   &cone, &B[j * ldb], 1);
    }

    /* Compute the maximum over the number of right hand sides of
       norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS). */
    *resid = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        f32 bnorm = cblas_scasum(n1, &B[j * ldb], 1);
        f32 xnorm = cblas_scasum(n1, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f32 temp = ((bnorm / anorm) / xnorm) / eps;
            if (temp > *resid) {
                *resid = temp;
            }
        }
    }
}
