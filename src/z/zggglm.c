/**
 * @file zggglm.c
 * @brief ZGGGLM solves a general Gauss-Markov linear model (GLM) problem.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGGGLM solves a general Gauss-Markov linear model (GLM) problem:
 *
 *         minimize || y ||_2   subject to   d = A*x + B*y
 *             x
 *
 * where A is an N-by-M matrix, B is an N-by-P matrix, and d is a
 * given N-vector. It is assumed that M <= N <= M+P, and
 *
 *            rank(A) = M    and    rank( A B ) = N.
 *
 * Under these assumptions, the constrained equation is always
 * consistent, and there is a unique solution x and a minimal 2-norm
 * solution y, which is obtained using a generalized QR factorization
 * of the matrices (A, B) given by
 *
 *    A = Q*(R),   B = Q*T*Z.
 *          (0)
 *
 * In particular, if matrix B is square nonsingular, then the problem
 * GLM is equivalent to the following weighted linear least squares
 * problem
 *
 *              minimize || inv(B)*(d-A*x) ||_2
 *                  x
 *
 * where inv(B) denotes the inverse of B.
 *
 * @param[in] n
 *          The number of rows of the matrices A and B. n >= 0.
 *
 * @param[in] m
 *          The number of columns of the matrix A. 0 <= m <= n.
 *
 * @param[in] p
 *          The number of columns of the matrix B. p >= n-m.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, m).
 *          On entry, the N-by-M matrix A.
 *          On exit, the upper triangular part of the array A contains
 *          the M-by-M upper triangular matrix R.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in,out] B
 *          Double complex array, dimension (ldb, p).
 *          On entry, the N-by-P matrix B.
 *          On exit, if n <= p, the upper triangle of the subarray
 *          B(1:n, p-n+1:p) contains the N-by-N upper triangular matrix T;
 *          if n > p, the elements on and above the (n-p)th subdiagonal
 *          contain the N-by-P upper trapezoidal matrix T.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[in,out] D
 *          Double complex array, dimension (n).
 *          On entry, D is the left hand side of the GLM equation.
 *          On exit, D is destroyed.
 *
 * @param[out] X
 *          Double complex array, dimension (m).
 *
 * @param[out] Y
 *          Double complex array, dimension (p).
 *
 *          On exit, X and Y are the solutions of the GLM problem.
 *
 * @param[out] work
 *          Double complex array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work. lwork >= max(1, n+m+p).
 *          For optimum performance, lwork >= m+min(n,p)+max(n,p)*NB,
 *          where NB is an upper bound for the optimal blocksizes for
 *          ZGEQRF, ZGERQF, ZUNMQR and ZUNMRQ.
 *
 *          If lwork = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the work array, returns
 *          this value as the first entry of the work array, and no error
 *          message related to lwork is issued by xerbla.
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - = 1: the upper triangular factor R associated with A in the
 *                           generalized QR factorization of the pair (A, B) is exactly
 *                           singular, so that rank(A) < M; the least squares
 *                           solution could not be computed.
 *                         - = 2: the bottom (N-M) by (N-M) part of the upper trapezoidal
 *                           factor T associated with B in the generalized QR
 *                           factorization of the pair (A, B) is exactly singular, so that
 *                           rank( A B ) < N; the least squares solution could not
 *                           be computed.
 */
void zggglm(
    const int n,
    const int m,
    const int p,
    c128* const restrict A,
    const int lda,
    c128* const restrict B,
    const int ldb,
    c128* restrict D,
    c128* restrict X,
    c128* restrict Y,
    c128* restrict work,
    const int lwork,
    int* info)
{
    const c128 zero = CMPLX(0.0, 0.0);
    const c128 one = CMPLX(1.0, 0.0);
    const c128 neg_one = CMPLX(-1.0, 0.0);

    int i, lopt, lwkmin, lwkopt, nb, nb1, nb2, nb3, nb4, np;
    int lquery;
    int max_val;

    *info = 0;
    np = (n < p) ? n : p;
    lquery = (lwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0 || m > n) {
        *info = -2;
    } else if (p < 0 || p < n - m) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }

    if (*info == 0) {
        if (n == 0) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            nb1 = lapack_get_nb("GEQRF");
            nb2 = lapack_get_nb("GERQF");
            nb3 = lapack_get_nb("UNMQR");
            nb4 = lapack_get_nb("UNMRQ");
            nb = nb1;
            if (nb2 > nb) nb = nb2;
            if (nb3 > nb) nb = nb3;
            if (nb4 > nb) nb = nb4;
            lwkmin = m + n + p;
            max_val = (n > p) ? n : p;
            lwkopt = m + np + max_val * nb;
        }
        work[0] = (c128)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZGGGLM", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        for (i = 0; i < m; i++) {
            X[i] = zero;
        }
        for (i = 0; i < p; i++) {
            Y[i] = zero;
        }
        return;
    }

    zggqrf(n, m, p, A, lda, work, B, ldb, &work[m],
           &work[m + np], lwork - m - np, info);
    lopt = (int)creal(work[m + np]);

    zunmqr("L", "C", n, 1, m, A, lda, work, D,
           (1 > n ? 1 : n), &work[m + np], lwork - m - np, info);
    lopt = (lopt > (int)creal(work[m + np])) ? lopt : (int)creal(work[m + np]);

    if (n > m) {
        ztrtrs("U", "N", "N", n - m, 1,
               &B[m + (m + p - n) * ldb], ldb, &D[m], n - m, info);

        if (*info > 0) {
            *info = 1;
            return;
        }

        cblas_zcopy(n - m, &D[m], 1, &Y[m + p - n], 1);
    }

    for (i = 0; i < m + p - n; i++) {
        Y[i] = zero;
    }

    cblas_zgemv(CblasColMajor, CblasNoTrans, m, n - m, &neg_one,
                &B[0 + (m + p - n) * ldb], ldb, &Y[m + p - n], 1, &one, D, 1);

    if (m > 0) {
        ztrtrs("U", "N", "N", m, 1, A, lda, D, m, info);

        if (*info > 0) {
            *info = 2;
            return;
        }

        cblas_zcopy(m, D, 1, X, 1);
    }

    int b_row_start = (1 > n - p + 1) ? 1 : (n - p + 1);
    b_row_start -= 1;

    zunmrq("L", "C", p, 1, np,
           &B[b_row_start + 0 * ldb], ldb, &work[m], Y,
           (1 > p ? 1 : p), &work[m + np], lwork - m - np, info);
    work[0] = (c128)(m + np + ((lopt > (int)creal(work[m + np])) ? lopt : (int)creal(work[m + np])));
}
