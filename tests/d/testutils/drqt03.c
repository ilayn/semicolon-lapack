/**
 * @file drqt03.c
 * @brief DRQT03 tests DORMRQ, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of DORMRQ with the results of forming Q explicitly
 * via DORGRQ and then performing DGEMM.
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
static uint64_t drqt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t drqt03_next(void) {
    uint64_t* s = drqt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f64 drqt03_drand(void) {
    uint64_t x = drqt03_next();
    return ((f64)(x >> 11)) * 0x1.0p-53 * 2.0 - 1.0;
}

/**
 * @param[in]     m       Number of rows or columns of C (see DORMRQ docs).
 * @param[in]     n       Order of the orthogonal matrix Q. n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     AF      RQ factorization from DGERQF, dimension (lda, n).
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
void drqt03(const INT m, const INT n, const INT k,
            const f64* const restrict AF,
            f64* const restrict C,
            f64* const restrict CC,
            f64* const restrict Q,
            const INT lda,
            const f64* const restrict tau,
            f64* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    f64 eps = dlamch("E");
    INT info;
    INT mc, nc;
    f64 cnorm, resid;
    INT minmn = m < n ? m : n;

    /* Quick return if possible */
    if (minmn == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 0.0;
        result[3] = 0.0;
        return;
    }

    /* Copy the last k rows of the factorization to Q and generate Q.
     * For RQ with k reflectors, they are stored in AF(m-k:m-1, 0:n-1).
     * We copy:
     *   - AF(m-k:m-1, 0:n-k-1) -> Q(n-k:n-1, 0:n-k-1) if k < n
     *   - Lower of AF(m-k+1:m-1, n-k:n-2) -> Q(n-k+1:n-1, n-k:n-2) if k > 1
     */
    dlaset("F", n, n, -1.0e+10, -1.0e+10, Q, lda);
    if (k > 0 && n > k) {
        dlacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(n - k) + 0 * lda], lda);
    }
    if (k > 1) {
        dlacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(n - k + 1) + (n - k) * lda], lda);
    }

    /* Generate the n-by-n matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    dorgrq(n, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

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
                C[i + j * lda] = drqt03_drand();
            }
        }
        cnorm = dlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            dlacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via DORMRQ.
             * DORMRQ uses AF(m-k:m-1, 0:n-1) and tau[minmn-k:minmn-1]. */
            if (k > 0) {
                dormrq(&side, &trans, mc, nc, k, &AF[(m - k) + 0 * lda], lda,
                       &tau[minmn - k], CC, lda, work, lwork, &info);
            }

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
