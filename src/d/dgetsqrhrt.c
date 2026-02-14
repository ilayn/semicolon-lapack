/**
 * @file dgetsqrhrt.c
 * @brief DGETSQRHRT computes a NB2-sized column blocked QR-factorization of a real M-by-N matrix A using TSQR and Householder reconstruction.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DGETSQRHRT computes a NB2-sized column blocked QR-factorization
 * of a real M-by-N matrix A with M >= N,
 *
 *    A = Q * R.
 *
 * The routine uses internally a NB1-sized column blocked and MB1-sized
 * row blocked TSQR-factorization and performs the reconstruction
 * of the Householder vectors from the TSQR output.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. m >= n >= 0.
 *
 * @param[in] mb1
 *          The row block size to be used in the blocked TSQR. mb1 > n.
 *
 * @param[in] nb1
 *          The column block size to be used in the blocked TSQR.
 *          n >= nb1 >= 1.
 *
 * @param[in] nb2
 *          The block size to be used in the blocked QR output. nb2 >= 1.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit, the upper triangular R factor and the Householder
 *          vectors V below the diagonal (compact WY representation).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] T
 *          Double precision array, dimension (ldt, n).
 *          The upper triangular block reflectors.
 *
 * @param[in] ldt
 *          The leading dimension of the array T. ldt >= nb2.
 *
 * @param[out] work
 *          Double precision workspace of size (max(1, lwork)).
 *          On exit, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void dgetsqrhrt(
    const int m,
    const int n,
    const int mb1,
    const int nb1,
    const int nb2,
    f64* const restrict A,
    const int lda,
    f64* restrict T,
    const int ldt,
    f64* restrict work,
    const int lwork,
    int* info)
{
    int lquery;
    int i, iinfo, j, lw1, lw2, lwt, ldwt, lworkopt;
    int nb1local, nb2local, num_all_row_blocks;
    int minval;

    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m < n) {
        *info = -2;
    } else if (mb1 <= n) {
        *info = -3;
    } else if (nb1 < 1) {
        *info = -4;
    } else if (nb2 < 1) {
        *info = -5;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -7;
    } else {
        minval = (nb2 < n) ? nb2 : n;
        if (ldt < (1 > minval ? 1 : minval)) {
            *info = -9;
        } else {

            if (lwork < n * n + 1 && !lquery) {
                *info = -11;
            } else {

                nb1local = (nb1 < n) ? nb1 : n;

                if (m - n > 0) {
                    num_all_row_blocks = (int)ceil((f64)(m - n) / (f64)(mb1 - n));
                } else {
                    num_all_row_blocks = 1;
                }
                if (num_all_row_blocks < 1) num_all_row_blocks = 1;

                lwt = num_all_row_blocks * n * nb1local;

                ldwt = nb1local;

                lw1 = nb1local * n;

                lw2 = nb1local * ((nb1local > (n - nb1local)) ? nb1local : (n - nb1local));

                lworkopt = (lwt + lw1 > lwt + n * n + lw2) ? lwt + lw1 : lwt + n * n + lw2;
                lworkopt = (lworkopt > lwt + n * n + n) ? lworkopt : lwt + n * n + n;
                lworkopt = (1 > lworkopt) ? 1 : lworkopt;

                if (lwork < lworkopt && !lquery) {
                    *info = -11;
                }

            }
        }
    }

    if (*info != 0) {
        xerbla("DGETSQRHRT", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f64)lworkopt;
        return;
    }

    minval = (m < n) ? m : n;
    if (minval == 0) {
        work[0] = (f64)lworkopt;
        return;
    }

    nb2local = (nb2 < n) ? nb2 : n;

    dlatsqr(m, n, mb1, nb1local, A, lda, work, ldwt, &work[lwt], lw1, &iinfo);

    for (j = 0; j < n; j++) {
        cblas_dcopy(j + 1, &A[0 + j * lda], 1, &work[lwt + n * j], 1);
    }

    dorgtsqr_row(m, n, mb1, nb1local, A, lda, work, ldwt, &work[lwt + n * n], lw2, &iinfo);

    dorhr_col(m, n, nb2local, A, lda, T, ldt, &work[lwt + n * n], &iinfo);

    for (i = 0; i < n; i++) {
        if (work[lwt + n * n + i] == -1.0) {
            for (j = i; j < n; j++) {
                A[i + j * lda] = -1.0 * work[lwt + n * j + i];
            }
        } else {
            cblas_dcopy(n - i, &work[lwt + n * i + i], n, &A[i + i * lda], lda);
        }
    }

    work[0] = (f64)lworkopt;
}
