/**
 * @file srqt03.c
 * @brief SRQT03 tests SORMRQ, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of SORMRQ with the results of forming Q explicitly
 * via SORGRQ and then performing DGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( N * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( N * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( N * norm(C) * EPS )   [Left, Trans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( N * norm(C) * EPS )   [Right, Trans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "verify.h"
#include <stdint.h>
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
extern void sorgrq(const int m, const int n, const int k,
                   f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);
extern void sormrq(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict C, const int ldc,
                   f32* const restrict work, const int lwork, int* info);

/* Simple xoshiro256+ for generating random test matrices */
static uint64_t drqt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t drqt03_next(void) {
    uint64_t* s = drqt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f32 drqt03_drand(void) {
    uint64_t x = drqt03_next();
    return ((f32)(x >> 11)) * 0x1.0fp-53 * 2.0f - 1.0f;
}

/**
 * @param[in]     m       Number of rows or columns of C (see SORMRQ docs).
 * @param[in]     n       Order of the orthogonal matrix Q. n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     AF      RQ factorization from SGERQF, dimension (lda, n).
 * @param[out]    C       Random matrix.
 * @param[out]    CC      Copy of C / result.
 * @param[out]    Q       The n-by-n matrix Q.
 * @param[in]     lda     Leading dimension.
 * @param[in]     tau     Scalar factors, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace.
 * @param[out]    result  Array of dimension 4.
 */
void srqt03(const int m, const int n, const int k,
            const f32* const restrict AF,
            f32* const restrict C,
            f32* const restrict CC,
            f32* const restrict Q,
            const int lda,
            const f32* const restrict tau,
            f32* const restrict work, const int lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    f32 eps = slamch("E");
    int info;
    int mc, nc;
    f32 cnorm, resid;
    int minmn = m < n ? m : n;

    /* Quick return if possible */
    if (minmn == 0) {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        return;
    }

    /* Copy the last k rows of the factorization to Q and generate Q.
     * For RQ with k reflectors, they are stored in AF(m-k:m-1, 0:n-1).
     * We copy:
     *   - AF(m-k:m-1, 0:n-k-1) -> Q(n-k:n-1, 0:n-k-1) if k < n
     *   - Lower of AF(m-k+1:m-1, n-k:n-2) -> Q(n-k+1:n-1, n-k:n-2) if k > 1
     */
    slaset("F", n, n, -1.0e+10f, -1.0e+10f, Q, lda);
    if (k > 0 && n > k) {
        slacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(n - k) + 0 * lda], lda);
    }
    if (k > 1) {
        slacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(n - k + 1) + (n - k) * lda], lda);
    }

    /* Generate the n-by-n matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    sorgrq(n, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    for (int iside = 0; iside < 2; iside++) {
        char side;
        if (iside == 0) {
            side = 'L';
            mc = n;
            nc = m;
        } else {
            side = 'R';
            mc = m;
            nc = n;
        }

        /* Generate MC by NC random matrix C */
        for (int j = 0; j < nc; j++) {
            for (int i = 0; i < mc; i++) {
                C[i + j * lda] = drqt03_drand();
            }
        }
        cnorm = slange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0f) cnorm = 1.0f;

        for (int itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            slacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via SORMRQ.
             * SORMRQ uses AF(m-k:m-1, 0:n-1) and tau[minmn-k:minmn-1]. */
            if (k > 0) {
                sormrq(&side, &trans, mc, nc, k, &AF[(m - k) + 0 * lda], lda,
                       &tau[minmn - k], CC, lda, work, lwork, &info);
            }

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
                ((f32)(n > 1 ? n : 1) * cnorm * eps);
        }
    }
}
