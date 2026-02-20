/**
 * @file sgbt01.c
 * @brief SGBT01 reconstructs a band matrix from its L*U factorization.
 *
 * Port of LAPACK TESTING/LIN/sgbt01.f
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

/**
 * SGBT01 reconstructs a band matrix A from its L*U factorization and
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
 * @param[in] AFAC   The factored form of A from SGBTRF. Dimension (ldafac, n).
 * @param[in] ldafac The leading dimension of AFAC. ldafac >= max(1, 2*kl+ku+1).
 * @param[in] ipiv   The pivot indices from SGBTRF. Dimension (min(m,n)).
 * @param[out] work  Workspace array, dimension (2*kl+ku+1).
 * @param[out] resid norm(L*U - A) / (N * norm(A) * EPS)
 */
void sgbt01(int m, int n, int kl, int ku,
            const f32* A, int lda,
            const f32* AFAC, int ldafac,
            const int* ipiv,
            f32* work,
            f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Quick exit if m = 0 or n = 0. */
    *resid = ZERO;
    if (m <= 0 || n <= 0) {
        return;
    }

    /* Determine EPS and the norm of A. */
    f32 eps = slamch("Epsilon");
    int kd = ku;  /* Row index of diagonal in A storage (0-based: row ku) */
    f32 anorm = ZERO;

    for (int j = 0; j < n; j++) {
        /* For column j, the band elements are in rows i1 to i2 of A */
        int i1 = (kd + 1 - j - 1 > 0) ? kd + 1 - j - 1 : 0;  /* max(kd+1-j, 1) - 1 in 0-based */
        int i2_excl = (kd + m - j < kl + kd + 1) ? kd + m - j : kl + kd + 1; /* min(kd+m-j, kl+kd+1) in 0-based */

        if (i2_excl > i1) {
            f32 col_sum = cblas_sasum(i2_excl - i1, &A[i1 + j * lda], 1);
            if (col_sum > anorm) {
                anorm = col_sum;
            }
        }
    }

    /* Compute one column at a time of L*U - A. */
    kd = kl + ku;  /* Row index of diagonal in AFAC storage */

    for (int j = 0; j < n; j++) {
        /* Copy the j-th column of U to work. */
        int ju = (kl + ku < j) ? kl + ku : j;  /* min(kl+ku, j) */
        int jl = (kl < m - j - 1) ? kl : m - j - 1;  /* min(kl, m-j-1) */
        int lenj = ((m < j + 1) ? m : j + 1) - (j + 1) + ju + 1;  /* min(m, j+1) - (j+1) + ju + 1 */

        if (lenj > 0) {
            /* Copy U column */
            cblas_scopy(lenj, &AFAC[(kd - ju) + j * ldafac], 1, work, 1);

            /* Zero the rest */
            for (int i = lenj; i < ju + jl + 1; i++) {
                work[i] = ZERO;
            }

            /* Multiply by the unit lower triangular matrix L.
             * L is stored as a product of transformations and permutations. */
            int i_start = (m - 2 < j) ? m - 2 : j;  /* min(m-1, j+1) - 1 in 0-based */
            for (int i = i_start; i >= j - ju; i--) {
                int il = (kl < m - i - 1) ? kl : m - i - 1;  /* min(kl, m-i-1) */
                if (il > 0) {
                    int iw = i - j + ju;  /* 0-based work index */
                    f32 t = work[iw];
                    cblas_saxpy(il, t, &AFAC[(kd + 1) + i * ldafac], 1, &work[iw + 1], 1);

                    int ip = ipiv[i];  /* ipiv is 0-based in our implementation */
                    if (i != ip) {
                        int ip_work = ip - j + ju;
                        work[iw] = work[ip_work];
                        work[ip_work] = t;
                    }
                }
            }

            /* Subtract the corresponding column of A. */
            int jua = (ju < ku) ? ju : ku;  /* min(ju, ku) */
            if (jua + jl + 1 > 0) {
                cblas_saxpy(jua + jl + 1, -ONE,
                           &A[(ku - jua) + j * lda], 1,
                           &work[ju - jua], 1);
            }

            /* Compute the 1-norm of the column. */
            f32 col_resid = cblas_sasum(ju + jl + 1, work, 1);
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
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
