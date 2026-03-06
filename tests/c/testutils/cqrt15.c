/**
 * @file cqrt15.c
 * @brief CQRT15 generates a matrix with full or deficient rank and of
 *        various norms.
 *
 * Faithful port of LAPACK TESTING/LIN/cqrt15.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"


/**
 * CQRT15 generates a matrix with full or deficient rank and of various norms.
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
void cqrt15(const INT scale, const INT rksel,
            const INT m, const INT n, const INT nrhs,
            c64* A, const INT lda, c64* B, const INT ldb,
            f32* S, INT* rank, f32* norma, f32* normb,
            c64* work, const INT lwork,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 SVMIN = 0.1f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT info, j, mn;
    f32 bignum, eps, smlnum, temp;
    f32 dummy[1];

    mn = (m < n) ? m : n;

    /* Check workspace */
    INT req_lwork = m + mn;
    if (mn * nrhs > req_lwork) req_lwork = mn * nrhs;
    if (2 * n + m > req_lwork) req_lwork = 2 * n + m;
    if (lwork < req_lwork) {
        xerbla("CQRT15", 16);
        return;
    }

    smlnum = slamch("S");
    bignum = ONE / smlnum;
    eps = slamch("E");
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
        xerbla("CQRT15", 2);
        return;
    }

    if (*rank > 0) {
        /* Nontrivial case */
        S[0] = ONE;
        for (j = 1; j < *rank; j++) {
            /* Generate random singular value > SVMIN */
            do {
                temp = rng_uniform_f32(state);
            } while (temp <= SVMIN);
            S[j] = fabsf(temp);
        }

        /* Sort singular values in decreasing order */
        slaord("D", *rank, S, 1);

        /* Generate 'rank' columns of a random unitary matrix in A */
        clarnv_rng(2, m, work, state);
        cblas_csscal(m, ONE / cblas_scnrm2(m, work, 1), work, 1);
        claset("F", m, *rank, CZERO, CONE, A, lda);
        clarf("L", m, *rank, work, 1, CMPLXF(TWO, 0.0f), A, lda, &work[m]);

        /* workspace used: m+mn */

        /* Generate consistent rhs in the range space of A */
        clarnv_rng(2, (*rank) * nrhs, work, state);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, nrhs, *rank, &CONE, A, lda, work, *rank,
                    &CZERO, B, ldb);

        /* work space used: <= mn *nrhs */

        /* generate (unscaled) matrix A */
        for (j = 0; j < *rank; j++) {
            cblas_csscal(m, S[j], &A[j * lda], 1);
        }
        if (*rank < n) {
            claset("F", m, n - *rank, CZERO, CZERO, &A[(*rank) * lda], lda);
        }
        claror("R", "N", m, n, A, lda, work, &info, state);

    } else {
        /* work space used 2*n+m */

        /* Generate null matrix and rhs */
        for (j = 0; j < mn; j++) {
            S[j] = ZERO;
        }
        claset("F", m, n, CZERO, CZERO, A, lda);
        claset("F", m, nrhs, CZERO, CZERO, B, ldb);
    }

    /* Scale the matrix */
    if (scale != 1) {
        *norma = clange("M", m, n, A, lda, dummy);
        if (*norma != ZERO) {
            if (scale == 2) {
                /* matrix scaled up */
                clascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
                slascl("G", 0, 0, *norma, bignum, mn, 1, S, mn, &info);
                clascl("G", 0, 0, *norma, bignum, m, nrhs, B, ldb, &info);
            } else if (scale == 3) {
                /* matrix scaled down */
                clascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
                slascl("G", 0, 0, *norma, smlnum, mn, 1, S, mn, &info);
                clascl("G", 0, 0, *norma, smlnum, m, nrhs, B, ldb, &info);
            } else {
                xerbla("CQRT15", 1);
                return;
            }
        }
    }

    *norma = cblas_sasum(mn, S, 1);
    *normb = clange("O", m, nrhs, B, ldb, dummy);
}
