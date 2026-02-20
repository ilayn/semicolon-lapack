/**
 * @file sqrt15.c
 * @brief SQRT15 generates a matrix with full or deficient rank and of
 *        various norms.
 *
 * Faithful port of LAPACK TESTING/LIN/sqrt15.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern void slascl(const char* type, const int kl, const int ku,
                   const f32 cfrom, const f32 cto,
                   const int m, const int n, f32* A, const int lda,
                   int* info);
extern void slarf(const char* side, const int m, const int n,
                  const f32* v, const int incv, const f32 tau,
                  f32* C, const int ldc, f32* work);
extern void xerbla(const char* srname, const int info);
/* slaror declared in verify.h */

/**
 * SQRT15 generates a matrix with full or deficient rank and of various norms.
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
void sqrt15(const int scale, const int rksel,
            const int m, const int n, const int nrhs,
            f32* A, const int lda, f32* B, const int ldb,
            f32* S, int* rank, f32* norma, f32* normb,
            f32* work, const int lwork,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 SVMIN = 0.1f;

    int info, j, mn;
    f32 bignum, eps, smlnum, temp;
    f32 dummy[1];

    mn = (m < n) ? m : n;

    /* Check workspace */
    int req_lwork = m + mn;
    if (mn * nrhs > req_lwork) req_lwork = mn * nrhs;
    if (2 * n + m > req_lwork) req_lwork = 2 * n + m;
    if (lwork < req_lwork) {
        xerbla("SQRT15", 16);
        return;
    }

    smlnum = slamch("S");  /* Safe minimum */
    bignum = ONE / smlnum;
    eps = slamch("E");     /* Epsilon */
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
        xerbla("SQRT15", 2);
        return;
    }

    if (*rank > 0) {
        /* Nontrivial case */
        S[0] = ONE;
        for (j = 1; j < *rank; j++) {
            /* Generate random singular value > SVMIN */
            do {
                temp = rng_uniform_f32(state);  /* Uniform(0,1) */
            } while (temp <= SVMIN);
            S[j] = fabsf(temp);
        }

        /* Sort singular values in decreasing order */
        slaord("D", *rank, S, 1);

        /* Generate 'rank' columns of a random orthogonal matrix in A */
        rng_fill_f32(state, 2, m, work);  /* Uniform(-1,1) */
        f32 nrm = cblas_snrm2(m, work, 1);
        cblas_sscal(m, ONE / nrm, work, 1);
        slaset("F", m, *rank, ZERO, ONE, A, lda);
        slarf("L", m, *rank, work, 1, TWO, A, lda, &work[m]);

        /* workspace used: m + mn */

        /* Generate consistent rhs in the range space of A */
        rng_fill_f32(state, 2, (*rank) * nrhs, work);  /* Uniform(-1,1) */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, nrhs, *rank, ONE, A, lda, work, *rank,
                    ZERO, B, ldb);

        /* work space used: <= mn * nrhs */

        /* Generate (unscaled) matrix A */
        for (j = 0; j < *rank; j++) {
            cblas_sscal(m, S[j], &A[j * lda], 1);
        }
        if (*rank < n) {
            slaset("F", m, n - *rank, ZERO, ZERO, &A[(*rank) * lda], lda);
        }
        slaror("R", "N", m, n, A, lda, work, &info, state);

    } else {
        /* work space used: 2*n + m */

        /* Generate null matrix and rhs */
        for (j = 0; j < mn; j++) {
            S[j] = ZERO;
        }
        slaset("F", m, n, ZERO, ZERO, A, lda);
        slaset("F", m, nrhs, ZERO, ZERO, B, ldb);
    }

    /* Scale the matrix */
    if (scale != 1) {
        *norma = slange("M", m, n, A, lda, dummy);
        if (*norma != ZERO) {
            if (scale == 2) {
                /* Matrix scaled up */
                slascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
                slascl("G", 0, 0, *norma, bignum, mn, 1, S, mn, &info);
                slascl("G", 0, 0, *norma, bignum, m, nrhs, B, ldb, &info);
            } else if (scale == 3) {
                /* Matrix scaled down */
                slascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
                slascl("G", 0, 0, *norma, smlnum, mn, 1, S, mn, &info);
                slascl("G", 0, 0, *norma, smlnum, m, nrhs, B, ldb, &info);
            } else {
                xerbla("SQRT15", 1);
                return;
            }
        }
    }

    *norma = cblas_sasum(mn, S, 1);
    *normb = slange("O", m, nrhs, B, ldb, dummy);
}
