/**
 * @file dqlt03.c
 * @brief DQLT03 tests DORMQL, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of DORMQL with the results of forming Q explicitly
 * via DORGQL and then performing DGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( M * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( M * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( M * norm(C) * EPS )   [Left, Trans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( M * norm(C) * EPS )   [Right, Trans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include <stdint.h>
/* Simple xoshiro256+ for generating random test matrices */
static uint64_t dqlt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t dqlt03_next(void) {
    uint64_t* s = dqlt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f64 dqlt03_drand(void) {
    uint64_t x = dqlt03_next();
    return ((f64)(x >> 11)) * 0x1.0p-53 * 2.0 - 1.0;
}

/**
 * @param[in]     m       Order of the orthogonal matrix Q. m >= 0.
 * @param[in]     n       Number of rows or columns of C (see DORMQL docs).
 * @param[in]     k       Number of reflectors. m >= k >= 0.
 * @param[in]     AF      QL factorization from DGEQLF, dimension (lda, n).
 * @param[out]    C       Random matrix.
 * @param[out]    CC      Copy of C / result.
 * @param[out]    Q       The m-by-m matrix Q.
 * @param[in]     lda     Leading dimension.
 * @param[in]     tau     Scalar factors, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace.
 * @param[out]    result  Array of dimension 4.
 */
void dqlt03(const INT m, const INT n, const INT k,
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

    /* Copy the last k columns of the factorization to Q and generate Q.
     * For QL with k reflectors, they are stored in AF(0:m-1, n-k:n-1).
     * We copy:
     *   - AF(0:m-k-1, n-k:n-1) -> Q(0:m-k-1, m-k:m-1) if k < m
     *   - Upper of AF(m-k:m-2, n-k+1:n-1) -> Q(m-k:m-2, m-k+1:m-1) if k > 1
     */
    dlaset("F", m, m, -1.0e+10, -1.0e+10, Q, lda);
    if (k > 0 && m > k) {
        dlacpy("F", m - k, k, &AF[0 + (n - k) * lda], lda,
               &Q[0 + (m - k) * lda], lda);
    }
    if (k > 1) {
        dlacpy("U", k - 1, k - 1, &AF[(m - k) + (n - k + 1) * lda], lda,
               &Q[(m - k) + (m - k + 1) * lda], lda);
    }

    /* Generate the m-by-m matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    dorgql(m, m, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    for (INT iside = 0; iside < 2; iside++) {
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
        for (INT j = 0; j < nc; j++) {
            for (INT i = 0; i < mc; i++) {
                C[i + j * lda] = dqlt03_drand();
            }
        }
        cnorm = dlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            dlacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via DORMQL.
             * DORMQL uses AF(0:m-1, n-k:n-1) and tau[minmn-k:minmn-1]. */
            if (k > 0) {
                dormql(&side, &trans, mc, nc, k, &AF[0 + (n - k) * lda], lda,
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
                ((f64)(m > 1 ? m : 1) * cnorm * eps);
        }
    }
}
