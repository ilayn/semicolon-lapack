/**
 * @file sorhr_col.c
 * @brief SORHR_COL takes an M-by-N matrix Q_in with orthonormal columns and performs Householder Reconstruction.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SORHR_COL takes an M-by-N real matrix Q_in with orthonormal columns
 * as input, stored in A, and performs Householder Reconstruction (HR),
 * i.e. reconstructs Householder vectors V(i) implicitly representing
 * another M-by-N matrix Q_out, with the property that Q_in = Q_out*S,
 * where S is an N-by-N diagonal matrix with diagonal entries
 * equal to +1 or -1. The Householder vectors (columns V(i) of V) are
 * stored in A on output, and the diagonal entries of S are stored in D.
 * Block reflectors are also returned in T.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. m >= n >= 0.
 *
 * @param[in] nb
 *          The column block size to be used in the reconstruction
 *          of Householder column vector blocks. nb >= 1.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the M-by-N orthonormal matrix Q_in.
 *          On exit, below diagonal contains the unit lower-trapezoidal
 *          matrix V of Householder vectors, above diagonal contains U.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] T
 *          Double precision array, dimension (ldt, n).
 *          On exit, contains upper-triangular block reflectors.
 *
 * @param[in] ldt
 *          The leading dimension of the array T. ldt >= max(1, min(nb, n)).
 *
 * @param[out] D
 *          Double precision array, dimension min(m, n).
 *          The diagonal elements of the sign matrix S.
 *
 * @param[out] info
 *          = 0:  successful exit
 *          < 0:  if info = -i, the i-th argument had an illegal value
 */
void sorhr_col(
    const int m,
    const int n,
    const int nb,
    float* const restrict A,
    const int lda,
    float* restrict T,
    const int ldt,
    float* restrict D,
    int* info)
{
    int i, iinfo, j, jb, jbtemp1, jbtemp2, jnb;
    int minval;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (nb < 1) {
        *info = -3;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -5;
    } else {
        minval = (nb < n) ? nb : n;
        if (ldt < (1 > minval ? 1 : minval)) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("SORHR_COL", -(*info));
        return;
    }

    minval = (m < n) ? m : n;
    if (minval == 0) {
        return;
    }

    slaorhr_col_getrfnp(n, n, A, lda, D, &iinfo);

    if (m > n) {
        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m - n, n, 1.0f, A, lda, &A[n + 0 * lda], lda);
    }

    for (jb = 0; jb < n; jb += nb) {

        jnb = ((n - jb) < nb) ? (n - jb) : nb;

        jbtemp1 = jb;
        for (j = jb; j < jb + jnb; j++) {
            cblas_scopy(j - jbtemp1 + 1, &A[jb + j * lda], 1, &T[0 + j * ldt], 1);
        }

        for (j = jb; j < jb + jnb; j++) {
            if (D[j] == 1.0f) {
                cblas_sscal(j - jbtemp1 + 1, -1.0f, &T[0 + j * ldt], 1);
            }
        }

        jbtemp2 = jb - 1;
        for (j = jb; j < jb + jnb - 1; j++) {
            for (i = j - jbtemp2; i < ((nb < n) ? nb : n); i++) {
                T[i + j * ldt] = 0.0f;
            }
        }

        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                    jnb, jnb, 1.0f, &A[jb + jb * lda], lda, &T[0 + jb * ldt], ldt);

    }
}
