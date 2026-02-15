/**
 * @file cgglse.c
 * @brief CGGLSE solves the linear equality-constrained least squares (LSE) problem.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGGLSE solves the linear equality-constrained least squares (LSE)
 * problem:
 *
 *         minimize || c - A*x ||_2   subject to   B*x = d
 *
 * where A is an M-by-N matrix, B is a P-by-N matrix, c is a given
 * M-vector, and d is a given P-vector. It is assumed that
 * P <= N <= M+P, and
 *
 *          rank(B) = P and  rank( (A) ) = N.
 *                               ( (B) )
 *
 * These conditions ensure that the LSE problem has a unique solution,
 * which is obtained using a generalized RQ factorization of the
 * matrices (B, A) given by
 *
 *    B = (0 R)*Q,   A = Z*T*Q.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrices A and B. n >= 0.
 *
 * @param[in] p
 *          The number of rows of the matrix B. 0 <= p <= n <= m+p.
 *
 * @param[in,out] A
 *          Single complex array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array
 *          contain the min(M,N)-by-N upper trapezoidal matrix T.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in,out] B
 *          Single complex array, dimension (ldb, n).
 *          On entry, the P-by-N matrix B.
 *          On exit, the upper triangle of the subarray B(1:P,N-P+1:N)
 *          contains the P-by-P upper triangular matrix R.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, p).
 *
 * @param[in,out] C
 *          Single complex array, dimension (m).
 *          On entry, C contains the right hand side vector for the
 *          least squares part of the LSE problem.
 *          On exit, the residual sum of squares for the solution
 *          is given by the sum of squares of elements N-P+1 to M of
 *          vector C.
 *
 * @param[in,out] D
 *          Single complex array, dimension (p).
 *          On entry, D contains the right hand side vector for the
 *          constrained equation.
 *          On exit, D is destroyed.
 *
 * @param[out] X
 *          Single complex array, dimension (n).
 *          On exit, X is the solution of the LSE problem.
 *
 * @param[out] work
 *          Single complex array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work. lwork >= max(1, m+n+p).
 *          For optimum performance lwork >= p+min(m,n)+max(m,n)*NB,
 *          where NB is an upper bound for the optimal blocksizes for
 *          CGEQRF, CGERQF, CUNMQR and CUNMRQ.
 *
 *          If lwork = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the work array, returns
 *          this value as the first entry of the work array, and no error
 *          message related to lwork is issued by xerbla.
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - = 1: the upper triangular factor R associated with B in the
 *                           generalized RQ factorization of the pair (B, A) is exactly
 *                           singular, so that rank(B) < P; the least squares
 *                           solution could not be computed.
 *                         - = 2: the (N-P) by (N-P) part of the upper trapezoidal factor
 *                           T associated with A in the generalized RQ factorization
 *                           of the pair (B, A) is exactly singular, so that
 *                           rank( (A) ) < N; the least squares solution could not
 *                           ( (B) )
 *                           be computed.
 */
void cgglse(
    const int m,
    const int n,
    const int p,
    c64* restrict A,
    const int lda,
    c64* restrict B,
    const int ldb,
    c64* restrict C,
    c64* restrict D,
    c64* restrict X,
    c64* restrict work,
    const int lwork,
    int* info)
{
    const c64 cone = CMPLXF(1.0f, 0.0f);
    const c64 neg_cone = CMPLXF(-1.0f, 0.0f);

    int lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr;
    int lquery;
    int max_val;

    *info = 0;
    mn = (m < n) ? m : n;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (p < 0 || p > n || p < n - m) {
        *info = -3;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -5;
    } else if (ldb < (1 > p ? 1 : p)) {
        *info = -7;
    }

    if (*info == 0) {
        if (n == 0) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            nb1 = lapack_get_nb("GEQRF");
            nb2 = lapack_get_nb("GERQF");
            nb3 = lapack_get_nb("ORMQR");
            nb4 = lapack_get_nb("ORMRQ");
            nb = nb1;
            if (nb2 > nb) nb = nb2;
            if (nb3 > nb) nb = nb3;
            if (nb4 > nb) nb = nb4;
            lwkmin = m + n + p;
            max_val = (m > n) ? m : n;
            lwkopt = p + mn + max_val * nb;
        }
        work[0] = (c64)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("CGGLSE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    cggrqf(p, m, n, B, ldb, work, A, lda, &work[p],
           &work[p + mn], lwork - p - mn, info);
    lopt = (int)crealf(work[p + mn]);

    cunmqr("L", "C", m, 1, mn, A, lda, &work[p],
           C, (1 > m ? 1 : m), &work[p + mn], lwork - p - mn, info);
    lopt = (lopt > (int)crealf(work[p + mn])) ? lopt : (int)crealf(work[p + mn]);

    if (p > 0) {
        ctrtrs("U", "N", "N", p, 1, &B[0 + (n - p) * ldb], ldb, D, p, info);

        if (*info > 0) {
            *info = 1;
            return;
        }

        cblas_ccopy(p, D, 1, &X[n - p], 1);

        cblas_cgemv(CblasColMajor, CblasNoTrans, n - p, p, &neg_cone,
                    &A[0 + (n - p) * lda], lda, D, 1, &cone, C, 1);
    }

    if (n > p) {
        ctrtrs("U", "N", "N", n - p, 1, A, lda, C, n - p, info);

        if (*info > 0) {
            *info = 2;
            return;
        }

        cblas_ccopy(n - p, C, 1, X, 1);
    }

    if (m < n) {
        nr = m + p - n;
        if (nr > 0) {
            cblas_cgemv(CblasColMajor, CblasNoTrans, nr, n - m, &neg_cone,
                        &A[(n - p) + m * lda], lda, &D[nr], 1, &cone, &C[n - p], 1);
        }
    } else {
        nr = p;
    }
    if (nr > 0) {
        cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    nr, &A[(n - p) + (n - p) * lda], lda, D, 1);
        cblas_caxpy(nr, &neg_cone, D, 1, &C[n - p], 1);
    }

    cunmrq("L", "C", n, 1, p, B, ldb, work, X, n,
           &work[p + mn], lwork - p - mn, info);
    work[0] = (c64)(p + mn + ((lopt > (int)crealf(work[p + mn])) ? lopt : (int)crealf(work[p + mn])));
}
