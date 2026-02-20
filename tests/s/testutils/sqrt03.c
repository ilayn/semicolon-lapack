/**
 * @file sqrt03.c
 * @brief SQRT03 tests SORMQR, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of SORMQR with the results of forming Q explicitly
 * via SORGQR and then performing DGEMM.
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
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* const restrict A, const int lda);
extern void sorgqr(const int m, const int n, const int k,
                   f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);
extern void sormqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict C, const int ldc,
                   f32* const restrict work, const int lwork, int* info);

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

static f32 dqrt03_drand(void) {
    uint64_t x = dqrt03_next();
    return ((f32)(x >> 11)) * 0x1.0fp-53 * 2.0f - 1.0f;
}

/**
 * @param[in]     m       Order of Q. m >= 0.
 * @param[in]     n       Number of columns (Left) or rows (Right) of C.
 * @param[in]     k       Number of reflectors defining Q. m >= k >= 0.
 * @param[in]     AF      QR factorization from SGEQRF, dimension (lda, k).
 * @param[out]    C       Random matrix, dimension (lda, max(m,n)).
 * @param[out]    CC      Copy of C / result of SORMQR.
 * @param[out]    Q       The m-by-m matrix Q.
 * @param[in]     lda     Leading dimension.
 * @param[in]     tau     Scalar factors from SGEQRF, dimension k.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 4.
 */
void sqrt03(const int m, const int n, const int k,
            const f32 * const restrict AF,
            f32 * const restrict C,
            f32 * const restrict CC,
            f32 * const restrict Q,
            const int lda,
            const f32 * const restrict tau,
            f32 * const restrict work, const int lwork,
            f32 * const restrict rwork,
            f32 * restrict result)
{
    f32 eps = slamch("E");
    int info;
    int mc, nc;
    f32 cnorm, resid;

    /* Copy the first k columns of the factorization to Q and generate Q */
    slaset("F", m, m, -1.0e+10f, -1.0e+10f, Q, lda);
    if (m > 1) {
        slacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }
    sorgqr(m, m, k, Q, lda, tau, work, lwork, &info);

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
        cnorm = slange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0f) cnorm = 1.0f;

        for (int itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            slacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' to CC via SORMQR */
            sormqr(&side, &trans, mc, nc, k, AF, lda, tau, CC, lda,
                   work, lwork, &info);

            /* Form explicit product and subtract */
            if (side == 'L') {
                cblas_sgemm(CblasColMajor,
                            trans == 'N' ? CblasNoTrans : CblasTrans,
                            CblasNoTrans,
                            mc, nc, mc, -1.0f, Q, lda, C, lda, 1.0f, CC, lda);
            } else {
                cblas_sgemm(CblasColMajor, CblasNoTrans,
                            trans == 'N' ? CblasNoTrans : CblasTrans,
                            mc, nc, nc, -1.0f, C, lda, Q, lda, 1.0f, CC, lda);
            }

            /* Compute error */
            resid = slange("1", mc, nc, CC, lda, rwork);
            result[iside * 2 + itrans] = resid /
                ((f32)(m > 1 ? m : 1) * cnorm * eps);
        }
    }
}
