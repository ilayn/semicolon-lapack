/**
 * @file dgbt01.c
 * @brief DGBT01 reconstructs a band matrix from its L*U factorization.
 *
 * Port of LAPACK TESTING/LIN/dgbt01.f
 */

#include <math.h>
#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DGBT01 reconstructs a band matrix A from its L*U factorization and
 * computes the residual:
 *    norm(L*U - A) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * The expression L*U - A is computed one column at a time, so A and
 * AFAC are not modified.
 *
 * @param[in] m      The number of rows of the matrix A. m >= 0.
 * @param[in] n      The number of columns of the matrix A. n >= 0.
 * @param[in] kl     The number of subdiagonals within the band of A. kl >= 0.
 * @param[in] ku     The number of superdiagonals within the band of A. ku >= 0.
 * @param[in] A      The original matrix A in band storage, stored in rows 0 to
 *                   kl+ku. Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, kl+ku+1).
 * @param[in] AFAC   The factored form of A from DGBTRF. Dimension (ldafac, n).
 * @param[in] ldafac The leading dimension of AFAC. ldafac >= max(1, 2*kl+ku+1).
 * @param[in] ipiv   The pivot indices from DGBTRF. Dimension (min(m,n)).
 * @param[out] work  Workspace array, dimension (2*kl+ku+1).
 * @param[out] resid norm(L*U - A) / (N * norm(A) * EPS)
 */
void dgbt01(INT m, INT n, INT kl, INT ku,
            const f64* A, INT lda,
            const f64* AFAC, INT ldafac,
            const INT* ipiv,
            f64* work,
            f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Quick exit if m = 0 or n = 0. */
    *resid = ZERO;
    if (m <= 0 || n <= 0) {
        return;
    }

    /* Determine EPS and the norm of A. */
    f64 eps = dlamch("Epsilon");
    INT kd = ku;  /* Row index of diagonal in A storage (0-based: row ku) */
    f64 anorm = ZERO;

    for (INT j = 0; j < n; j++) {
        /* For column j, the band elements are in rows i1 to i2 of A */
        INT i1 = (kd + 1 - j - 1 > 0) ? kd + 1 - j - 1 : 0;  /* max(kd+1-j, 1) - 1 in 0-based */
        INT i2_excl = (kd + m - j < kl + kd + 1) ? kd + m - j : kl + kd + 1; /* min(kd+m-j, kl+kd+1) in 0-based */

        if (i2_excl > i1) {
            f64 col_sum = cblas_dasum(i2_excl - i1, &A[i1 + j * lda], 1);
            if (col_sum > anorm) {
                anorm = col_sum;
            }
        }
    }

    /* Compute one column at a time of L*U - A. */
    kd = kl + ku;  /* Row index of diagonal in AFAC storage */

    for (INT j = 0; j < n; j++) {
        /* Copy the j-th column of U to work. */
        INT ju = (kl + ku < j) ? kl + ku : j;  /* min(kl+ku, j) */
        INT jl = (kl < m - j - 1) ? kl : m - j - 1;  /* min(kl, m-j-1) */
        INT lenj = ((m < j + 1) ? m : j + 1) - (j + 1) + ju + 1;  /* min(m, j+1) - (j+1) + ju + 1 */

        if (lenj > 0) {
            /* Copy U column */
            cblas_dcopy(lenj, &AFAC[(kd - ju) + j * ldafac], 1, work, 1);

            /* Zero the rest */
            for (INT i = lenj; i < ju + jl + 1; i++) {
                work[i] = ZERO;
            }

            /* Multiply by the unit lower triangular matrix L.
             * L is stored as a product of transformations and permutations. */
            INT i_start = (m - 2 < j) ? m - 2 : j;  /* min(m-1, j+1) - 1 in 0-based */
            for (INT i = i_start; i >= j - ju; i--) {
                INT il = (kl < m - i - 1) ? kl : m - i - 1;  /* min(kl, m-i-1) */
                if (il > 0) {
                    INT iw = i - j + ju;  /* 0-based work index */
                    f64 t = work[iw];
                    cblas_daxpy(il, t, &AFAC[(kd + 1) + i * ldafac], 1, &work[iw + 1], 1);

                    INT ip = ipiv[i];  /* ipiv is 0-based in our implementation */
                    if (i != ip) {
                        INT ip_work = ip - j + ju;
                        work[iw] = work[ip_work];
                        work[ip_work] = t;
                    }
                }
            }

            /* Subtract the corresponding column of A. */
            INT jua = (ju < ku) ? ju : ku;  /* min(ju, ku) */
            if (jua + jl + 1 > 0) {
                cblas_daxpy(jua + jl + 1, -ONE,
                           &A[(ku - jua) + j * lda], 1,
                           &work[ju - jua], 1);
            }

            /* Compute the 1-norm of the column. */
            f64 col_resid = cblas_dasum(ju + jl + 1, work, 1);
            if (col_resid > *resid) {
                *resid = col_resid;
            }
        }
    }

    /* Compute norm(L*U - A) / (N * norm(A) * EPS) */
    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
