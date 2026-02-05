/**
 * @file dqrt15.c
 * @brief DQRT15 generates a matrix with full or deficient rank and of
 *        various norms.
 *
 * Faithful port of LAPACK TESTING/LIN/dqrt15.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dlascl(const char* type, const int kl, const int ku,
                   const double cfrom, const double cto,
                   const int m, const int n, double* A, const int lda,
                   int* info);
extern void dlarf(const char* side, const int m, const int n,
                  const double* v, const int incv, const double tau,
                  double* C, const int ldc, double* work);
extern void xerbla(const char* srname, const int info);
/* dlaror declared in verify.h */

/**
 * DQRT15 generates a matrix with full or deficient rank and of various norms.
 *
 * @param[in] scale
 *     SCALE = 1: normally scaled matrix
 *     SCALE = 2: matrix scaled up
 *     SCALE = 3: matrix scaled down
 *
 * @param[in] rksel
 *     RKSEL = 1: full rank matrix
 *     RKSEL = 2: rank-deficient matrix
 *
 * @param[in] m
 *     The number of rows of the matrix A.
 *
 * @param[in] n
 *     The number of columns of A.
 *
 * @param[in] nrhs
 *     The number of columns of B.
 *
 * @param[out] A
 *     The M-by-N matrix A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *
 * @param[out] B
 *     A matrix that is in the range space of matrix A.
 *
 * @param[in] ldb
 *     The leading dimension of the array B.
 *
 * @param[out] S
 *     Singular values of A, dimension min(M,N).
 *
 * @param[out] rank
 *     Number of nonzero singular values of A.
 *
 * @param[out] norma
 *     One-norm of A.
 *
 * @param[out] normb
 *     One-norm of B.
 *
 * @param[out] work
 *     Workspace array of dimension LWORK.
 *
 * @param[in] lwork
 *     Length of work space required.
 *     LWORK >= MAX(M+MIN(M,N), NRHS*MIN(M,N), 2*N+M)
 */
void dqrt15(const int scale, const int rksel,
            const int m, const int n, const int nrhs,
            double* A, const int lda, double* B, const int ldb,
            double* S, int* rank, double* norma, double* normb,
            double* work, const int lwork)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double SVMIN = 0.1;

    int info, j, mn;
    double bignum, eps, smlnum, temp;
    double dummy[1];

    mn = (m < n) ? m : n;

    /* Check workspace */
    int req_lwork = m + mn;
    if (mn * nrhs > req_lwork) req_lwork = mn * nrhs;
    if (2 * n + m > req_lwork) req_lwork = 2 * n + m;
    if (lwork < req_lwork) {
        xerbla("DQRT15", 16);
        return;
    }

    smlnum = dlamch("S");  /* Safe minimum */
    bignum = ONE / smlnum;
    eps = dlamch("E");     /* Epsilon */
    smlnum = (smlnum / eps) / eps;
    bignum = ONE / smlnum;

    /* Seed RNG based on inputs for reproducibility */
    rng_seed(1988 + scale * 1000 + rksel * 100 + m * 10 + n);

    /* Determine rank and (unscaled) singular values */
    if (rksel == 1) {
        *rank = mn;
    } else if (rksel == 2) {
        *rank = (3 * mn) / 4;
        for (j = *rank; j < mn; j++) {
            S[j] = ZERO;
        }
    } else {
        xerbla("DQRT15", 2);
        return;
    }

    if (*rank > 0) {
        /* Nontrivial case */
        S[0] = ONE;
        for (j = 1; j < *rank; j++) {
            /* Generate random singular value > SVMIN */
            do {
                temp = rng_uniform();  /* Uniform(0,1) */
            } while (temp <= SVMIN);
            S[j] = fabs(temp);
        }

        /* Sort singular values in decreasing order */
        dlaord("D", *rank, S, 1);

        /* Generate 'rank' columns of a random orthogonal matrix in A */
        rng_fill(2, m, work);  /* Uniform(-1,1) */
        double nrm = cblas_dnrm2(m, work, 1);
        cblas_dscal(m, ONE / nrm, work, 1);
        dlaset("F", m, *rank, ZERO, ONE, A, lda);
        dlarf("L", m, *rank, work, 1, TWO, A, lda, &work[m]);

        /* workspace used: m + mn */

        /* Generate consistent rhs in the range space of A */
        rng_fill(2, (*rank) * nrhs, work);  /* Uniform(-1,1) */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, nrhs, *rank, ONE, A, lda, work, *rank,
                    ZERO, B, ldb);

        /* work space used: <= mn * nrhs */

        /* Generate (unscaled) matrix A */
        for (j = 0; j < *rank; j++) {
            cblas_dscal(m, S[j], &A[j * lda], 1);
        }
        if (*rank < n) {
            dlaset("F", m, n - *rank, ZERO, ZERO, &A[(*rank) * lda], lda);
        }
        /* Use different seed offset (2988) for dlaror to avoid correlation
           with the singular value generation (1988) */
        dlaror("R", "N", m, n, A, lda,
               2988 + scale * 1000 + rksel * 100 + m * 10 + n,
               work, &info);

    } else {
        /* work space used: 2*n + m */

        /* Generate null matrix and rhs */
        for (j = 0; j < mn; j++) {
            S[j] = ZERO;
        }
        dlaset("F", m, n, ZERO, ZERO, A, lda);
        dlaset("F", m, nrhs, ZERO, ZERO, B, ldb);
    }

    /* Scale the matrix */
    if (scale != 1) {
        *norma = dlange("M", m, n, A, lda, dummy);
        if (*norma != ZERO) {
            if (scale == 2) {
                /* Matrix scaled up */
                dlascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
                dlascl("G", 0, 0, *norma, bignum, mn, 1, S, mn, &info);
                dlascl("G", 0, 0, *norma, bignum, m, nrhs, B, ldb, &info);
            } else if (scale == 3) {
                /* Matrix scaled down */
                dlascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
                dlascl("G", 0, 0, *norma, smlnum, mn, 1, S, mn, &info);
                dlascl("G", 0, 0, *norma, smlnum, m, nrhs, B, ldb, &info);
            } else {
                xerbla("DQRT15", 1);
                return;
            }
        }
    }

    *norma = cblas_dasum(mn, S, 1);
    *normb = dlange("O", m, nrhs, B, ldb, dummy);
}
