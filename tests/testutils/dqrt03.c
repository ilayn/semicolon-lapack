/**
 * @file dqrt03.c
 * @brief DQRT03 tests DORMQR, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of DORMQR with the results of forming Q explicitly
 * via DORGQR and then performing DGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( M * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( M * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( M * norm(C) * EPS )   [Left, Trans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( M * norm(C) * EPS )   [Right, Trans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

// Forward declarations
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* const restrict A, const int lda,
                   f64* const restrict B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* const restrict A, const int lda);
extern void dorgqr(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);
extern void dormqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict C, const int ldc,
                   f64* const restrict work, const int lwork, int* info);

/* Simple xoshiro256+ for generating random test matrices */
static uint64_t dqrt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t dqrt03_next(void) {
    uint64_t *s = dqrt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f64 dqrt03_drand(void) {
    uint64_t x = dqrt03_next();
    return ((f64)(x >> 11)) * 0x1.0p-53 * 2.0 - 1.0;
}

/**
 * @param[in]     m       Order of Q. m >= 0.
 * @param[in]     n       Number of columns (Left) or rows (Right) of C.
 * @param[in]     k       Number of reflectors defining Q. m >= k >= 0.
 * @param[in]     AF      QR factorization from DGEQRF, dimension (lda, k).
 * @param[out]    C       Random matrix, dimension (lda, max(m,n)).
 * @param[out]    CC      Copy of C / result of DORMQR.
 * @param[out]    Q       The m-by-m matrix Q.
 * @param[in]     lda     Leading dimension.
 * @param[in]     tau     Scalar factors from DGEQRF, dimension k.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 4.
 */
void dqrt03(const int m, const int n, const int k,
            const f64 * const restrict AF,
            f64 * const restrict C,
            f64 * const restrict CC,
            f64 * const restrict Q,
            const int lda,
            const f64 * const restrict tau,
            f64 * const restrict work, const int lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    f64 eps = dlamch("E");
    int info;
    int mc, nc;
    f64 cnorm, resid;

    /* Copy the first k columns of the factorization to Q and generate Q */
    dlaset("F", m, m, -1.0e+10, -1.0e+10, Q, lda);
    if (m > 1) {
        dlacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }
    dorgqr(m, m, k, Q, lda, tau, work, lwork, &info);

    for (int iside = 0; iside < 2; iside++) {
        char side;
        if (iside == 0) {
            side = 'L';
            mc = m;
            nc = n;
        } else {
            side = 'R';
            mc = n;
            nc = m;
        }

        /* Generate MC by NC random matrix C */
        for (int j = 0; j < nc; j++) {
            for (int i = 0; i < mc; i++) {
                C[i + j * lda] = dqrt03_drand();
            }
        }
        cnorm = dlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (int itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            dlacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' to CC via DORMQR */
            dormqr(&side, &trans, mc, nc, k, AF, lda, tau, CC, lda,
                   work, lwork, &info);

            /* Form explicit product and subtract */
            if (side == 'L') {
                cblas_dgemm(CblasColMajor,
                            trans == 'N' ? CblasNoTrans : CblasTrans,
                            CblasNoTrans,
                            mc, nc, mc, -1.0, Q, lda, C, lda, 1.0, CC, lda);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans,
                            trans == 'N' ? CblasNoTrans : CblasTrans,
                            mc, nc, nc, -1.0, C, lda, Q, lda, 1.0, CC, lda);
            }

            /* Compute error */
            resid = dlange("1", mc, nc, CC, lda, rwork);
            result[iside * 2 + itrans] = resid /
                ((f64)(m > 1 ? m : 1) * cnorm * eps);
        }
    }
}
