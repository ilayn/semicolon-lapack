/**
 * @file zgbcon.c
 * @brief Estimates the reciprocal of the condition number of a complex banded matrix.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGBCON estimates the reciprocal of the condition number of a complex
 * general band matrix A, in either the 1-norm or the infinity-norm,
 * using the LU factorization computed by ZGBTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as
 *    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 * @param[in]     norm    Specifies whether the 1-norm condition number or the
 *                        infinity-norm condition number is required:
 *                        - '1' or 'O': 1-norm
 *                        - 'I': Infinity-norm
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     kl      The number of subdiagonals within the band of A (kl >= 0).
 * @param[in]     ku      The number of superdiagonals within the band of A (ku >= 0).
 * @param[in]     AB      The LU factorization of the band matrix A, as computed
 *                        by zgbtrf. U is stored as an upper triangular band matrix
 *                        with kl+ku superdiagonals in rows 0 to kl+ku, and the
 *                        multipliers used during the factorization are stored in
 *                        rows kl+ku+1 to 2*kl+ku. Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of the array AB (ldab >= 2*kl+ku+1).
 * @param[in]     ipiv    The pivot indices; for 0 <= i < n, row i of the matrix
 *                        was interchanged with row ipiv[i]. Array of dimension n.
 * @param[in]     anorm   If norm = '1' or "O", the 1-norm of the original matrix A.
 *                        If norm = "I", the infinity-norm of the original matrix A.
 * @param[out]    rcond   The reciprocal of the condition number of the matrix A,
 *                        computed as RCOND = 1/(norm(A) * norm(inv(A))).
 * @param[out]    work    Complex workspace array of dimension (2*n).
 * @param[out]    rwork   Real workspace array of dimension (n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void zgbcon(
    const char* norm,
    const int n,
    const int kl,
    const int ku,
    const c128* const restrict AB,
    const int ldab,
    const int* const restrict ipiv,
    const f64 anorm,
    f64* rcond,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int lnoti, onenrm;
    char normin;
    int ix, j, jp, kase, kase1, kd, lm;
    f64 ainvnm, scale, smlnum;
    c128 t;
    int isave[3];
    int linfo;

    /* Test the input parameters */
    *info = 0;
    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');
    if (!onenrm && norm[0] != 'I' && norm[0] != 'i') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (ldab < 2 * kl + ku + 1) {
        *info = -6;
    } else if (anorm < ZERO) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("ZGBCON", -(*info));
        return;
    }

    /* Quick return if possible */
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    smlnum = DBL_MIN;

    /* Estimate the norm of inv(A) */
    ainvnm = ZERO;
    normin = 'N';
    if (onenrm) {
        kase1 = 1;
    } else {
        kase1 = 2;
    }
    kd = kl + ku;  /* Row index of diagonal in band storage (0-based: row kl+ku) */
    lnoti = (kl > 0);
    kase = 0;

    while (1) {
        /* work[0:n-1] = X, work[n:2n-1] = V */
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (kase == kase1) {
            /* Multiply by inv(L) */
            if (lnoti) {
                for (j = 0; j < n - 1; j++) {
                    lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                    jp = ipiv[j];
                    t = work[jp];
                    if (jp != j) {
                        work[jp] = work[j];
                        work[j] = t;
                    }
                    /* work[j+1:j+lm] -= t * AB[kd+1:kd+lm, j] */
                    c128 neg_t = -t;
                    cblas_zaxpy(lm, &neg_t, &AB[kd + 1 + j * ldab], 1, &work[j + 1], 1);
                }
            }

            /* Multiply by inv(U) */
            zlatbs("U", "N", "N", &normin, n, kl + ku, AB, ldab, work, &scale,
                   rwork, &linfo);
        } else {
            /* Multiply by inv(U**H) */
            zlatbs("U", "C", "N", &normin, n, kl + ku, AB, ldab, work, &scale,
                   rwork, &linfo);

            /* Multiply by inv(L**H) */
            if (lnoti) {
                for (j = n - 2; j >= 0; j--) {
                    lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                    /* work[j] -= zdotc(AB[kd+1:kd+lm, j], work[j+1:j+lm]) */
                    c128 dotc;
                    cblas_zdotc_sub(lm, &AB[kd + 1 + j * ldab], 1, &work[j + 1], 1, &dotc);
                    work[j] -= dotc;
                    jp = ipiv[j];
                    if (jp != j) {
                        t = work[jp];
                        work[jp] = work[j];
                        work[j] = t;
                    }
                }
            }
        }

        /* Divide X by 1/SCALE if doing so will not cause overflow */
        normin = 'Y';
        if (scale != ONE) {
            ix = cblas_izamax(n, work, 1);
            if (scale < (fabs(creal(work[ix])) + fabs(cimag(work[ix]))) * smlnum || scale == ZERO) {
                return;
            }
            zdrscl(n, scale, work, 1);
        }
    }

    /* Compute the estimate of the reciprocal condition number */
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
