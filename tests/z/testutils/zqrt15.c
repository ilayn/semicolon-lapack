/**
 * @file zqrt15.c
 * @brief ZQRT15 generates a matrix with full or deficient rank and of
 *        various norms.
 *
 * Faithful port of LAPACK TESTING/LIN/zqrt15.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/* dlaord is a real utility from d-prefix testutils */
void dlaord(const char* job, const INT n, f64* X, const INT incx);

/**
 * ZQRT15 generates a matrix with full or deficient rank and of various norms.
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
void zqrt15(const INT scale, const INT rksel,
            const INT m, const INT n, const INT nrhs,
            c128* A, const INT lda, c128* B, const INT ldb,
            f64* S, INT* rank, f64* norma, f64* normb,
            c128* work, const INT lwork,
            uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 SVMIN = 0.1;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT info, j, mn;
    f64 bignum, eps, smlnum, temp;
    f64 dummy[1];

    mn = (m < n) ? m : n;

    /* Check workspace */
    INT req_lwork = m + mn;
    if (mn * nrhs > req_lwork) req_lwork = mn * nrhs;
    if (2 * n + m > req_lwork) req_lwork = 2 * n + m;
    if (lwork < req_lwork) {
        xerbla("ZQRT15", 16);
        return;
    }

    smlnum = dlamch("S");
    bignum = ONE / smlnum;
    eps = dlamch("E");
    smlnum = (smlnum / eps) / eps;
    bignum = ONE / smlnum;

    /* Determine rank and (unscaled) singular values */
    if (rksel == 1) {
        *rank = mn;
    } else if (rksel == 2) {
        *rank = (3 * mn) / 4;
        for (j = *rank; j < mn; j++) {
            S[j] = ZERO;
        }
    } else {
        xerbla("ZQRT15", 2);
        return;
    }

    if (*rank > 0) {
        /* Nontrivial case */
        S[0] = ONE;
        for (j = 1; j < *rank; j++) {
            /* Generate random singular value > SVMIN */
            do {
                temp = rng_uniform(state);
            } while (temp <= SVMIN);
            S[j] = fabs(temp);
        }

        /* Sort singular values in decreasing order */
        dlaord("D", *rank, S, 1);

        /* Generate 'rank' columns of a random unitary matrix in A */
        zlarnv_rng(2, m, work, state);
        cblas_zdscal(m, ONE / cblas_dznrm2(m, work, 1), work, 1);
        zlaset("F", m, *rank, CZERO, CONE, A, lda);
        zlarf("L", m, *rank, work, 1, CMPLX(TWO, 0.0), A, lda, &work[m]);

        /* workspace used: m+mn */

        /* Generate consistent rhs in the range space of A */
        zlarnv_rng(2, (*rank) * nrhs, work, state);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, nrhs, *rank, &CONE, A, lda, work, *rank,
                    &CZERO, B, ldb);

        /* work space used: <= mn *nrhs */

        /* generate (unscaled) matrix A */
        for (j = 0; j < *rank; j++) {
            cblas_zdscal(m, S[j], &A[j * lda], 1);
        }
        if (*rank < n) {
            zlaset("F", m, n - *rank, CZERO, CZERO, &A[(*rank) * lda], lda);
        }
        zlaror("R", "N", m, n, A, lda, work, &info, state);

    } else {
        /* work space used 2*n+m */

        /* Generate null matrix and rhs */
        for (j = 0; j < mn; j++) {
            S[j] = ZERO;
        }
        zlaset("F", m, n, CZERO, CZERO, A, lda);
        zlaset("F", m, nrhs, CZERO, CZERO, B, ldb);
    }

    /* Scale the matrix */
    if (scale != 1) {
        *norma = zlange("M", m, n, A, lda, dummy);
        if (*norma != ZERO) {
            if (scale == 2) {
                /* matrix scaled up */
                zlascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
                dlascl("G", 0, 0, *norma, bignum, mn, 1, S, mn, &info);
                zlascl("G", 0, 0, *norma, bignum, m, nrhs, B, ldb, &info);
            } else if (scale == 3) {
                /* matrix scaled down */
                zlascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
                dlascl("G", 0, 0, *norma, smlnum, mn, 1, S, mn, &info);
                zlascl("G", 0, 0, *norma, smlnum, m, nrhs, B, ldb, &info);
            } else {
                xerbla("ZQRT15", 1);
                return;
            }
        }
    }

    *norma = cblas_dasum(mn, S, 1);
    *normb = zlange("O", m, nrhs, B, ldb, dummy);
}