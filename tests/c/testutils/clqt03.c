/**
 * @file clqt03.c
 * @brief CLQT03 tests CUNMLQ, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of CUNMLQ with the results of forming Q explicitly
 * via CUNGLQ and then performing ZGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( N * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( N * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( N * norm(C) * EPS )   [Left, ConjTrans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( N * norm(C) * EPS )   [Right, ConjTrans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/* Simple xoshiro256+ for generating random test matrices */
static uint64_t zlqt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t zlqt03_next(void) {
    uint64_t* s = zlqt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f32 zlqt03_drand(void) {
    uint64_t x = zlqt03_next();
    return ((f32)(x >> 40)) * 0x1.0p-24f * 2.0f - 1.0f;
}

/**
 * @param[in]     m       Number of rows or columns of C (see CUNMLQ docs).
 * @param[in]     n       Order of Q. n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     AF      LQ factorization from CGELQF, dimension (lda, n).
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
void clqt03(const INT m, const INT n, const INT k,
            c64* const restrict AF,
            c64* const restrict C,
            c64* const restrict CC,
            c64* const restrict Q,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    f32 eps = slamch("E");
    INT info;
    INT mc, nc;
    f32 cnorm, resid;

    /* Copy the first k rows of the factorization to Q and generate Q */
    claset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (n > 1) {
        clacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }
    cunglq(n, n, k, Q, lda, tau, work, lwork, &info);

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
                C[i + j * lda] = CMPLXF(zlqt03_drand(), zlqt03_drand());
            }
        }
        cnorm = clange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0f) cnorm = 1.0f;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'C';

            /* Copy C to CC */
            clacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via CUNMLQ */
            cunmlq(&side, &trans, mc, nc, k, AF, lda, tau, CC, lda,
                   work, lwork, &info);

            /* Form explicit product and subtract */
            if (side == 'L') {
                cblas_cgemm(CblasColMajor,
                            trans == 'N' ? CblasNoTrans : CblasConjTrans,
                            CblasNoTrans,
                            mc, nc, mc, &CNEGONE, Q, lda, C, lda, &CONE, CC, lda);
            } else {
                cblas_cgemm(CblasColMajor, CblasNoTrans,
                            trans == 'N' ? CblasNoTrans : CblasConjTrans,
                            mc, nc, nc, &CNEGONE, C, lda, Q, lda, &CONE, CC, lda);
            }

            /* Compute error */
            resid = clange("1", mc, nc, CC, lda, rwork);
            result[iside * 2 + itrans] = resid /
                ((f32)(n > 1 ? n : 1) * cnorm * eps);
        }
    }
}
