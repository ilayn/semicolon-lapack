/**
 * @file dlqt03.c
 * @brief DLQT03 tests DORMLQ, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of DORMLQ with the results of forming Q explicitly
 * via DORGLQ and then performing DGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( N * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( N * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( N * norm(C) * EPS )   [Left, Trans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( N * norm(C) * EPS )   [Right, Trans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include <stdint.h>
/* Simple xoshiro256+ for generating random test matrices */
static uint64_t dlqt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t dlqt03_next(void) {
    uint64_t *s = dlqt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f64 dlqt03_drand(void) {
    uint64_t x = dlqt03_next();
    return ((f64)(x >> 11)) * 0x1.0p-53 * 2.0 - 1.0;
}

/**
 * @param[in]     m       Number of rows or columns of C (see DORMLQ docs).
 * @param[in]     n       Order of Q. n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     AF      LQ factorization from DGELQF, dimension (lda, n).
 * @param[out]    C       Random matrix.
 * @param[out]    CC      Copy of C / result.
 * @param[out]    Q       The n-by-n matrix Q.
 * @param[in]     lda     Leading dimension.
 * @param[in]     tau     Scalar factors, dimension k.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace.
 * @param[out]    result  Array of dimension 4.
 */
void dlqt03(const INT m, const INT n, const INT k,
            const f64 * const restrict AF,
            f64 * const restrict C,
            f64 * const restrict CC,
            f64 * const restrict Q,
            const INT lda,
            const f64 * const restrict tau,
            f64 * const restrict work, const INT lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    f64 eps = dlamch("E");
    INT info;
    INT mc, nc;
    f64 cnorm, resid;

    /* Copy the first k rows of the factorization to Q and generate Q */
    dlaset("F", n, n, -1.0e+10, -1.0e+10, Q, lda);
    if (n > 1) {
        dlacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }
    dorglq(n, n, k, Q, lda, tau, work, lwork, &info);

    for (INT iside = 0; iside < 2; iside++) {
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
        for (INT j = 0; j < nc; j++) {
            for (INT i = 0; i < mc; i++) {
                C[i + j * lda] = dlqt03_drand();
            }
        }
        cnorm = dlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            dlacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via DORMLQ */
            dormlq(&side, &trans, mc, nc, k, AF, lda, tau, CC, lda,
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
                ((f64)(n > 1 ? n : 1) * cnorm * eps);
        }
    }
}
