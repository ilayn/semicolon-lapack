/**
 * @file zrqt03.c
 * @brief ZRQT03 tests ZUNMRQ, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of ZUNMRQ with the results of forming Q explicitly
 * via ZUNGRQ and then performing ZGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( N * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( N * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( N * norm(C) * EPS )   [Left, ConjTrans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( N * norm(C) * EPS )   [Right, ConjTrans]
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * @param[in]     m       Number of rows or columns of C (see ZUNMRQ docs).
 * @param[in]     n       Order of the orthogonal matrix Q. n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     AF      RQ factorization from ZGERQF, dimension (lda, n).
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
void zrqt03(const INT m, const INT n, const INT k,
            c128* const restrict AF,
            c128* const restrict C,
            c128* const restrict CC,
            c128* const restrict Q,
            const INT lda,
            const c128* const restrict tau,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    f64 eps = dlamch("E");
    INT info;
    INT mc, nc;
    f64 cnorm, resid;
    INT minmn = m < n ? m : n;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    if (minmn == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 0.0;
        result[3] = 0.0;
        return;
    }

    zlaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (k > 0 && n > k) {
        zlacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(n - k) + 0 * lda], lda);
    }
    if (k > 1) {
        zlacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(n - k + 1) + (n - k) * lda], lda);
    }

    zungrq(n, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

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

        for (INT j = 0; j < nc; j++) {
            zlarnv_rng(2, mc, &C[j * lda], rng_state);
        }
        cnorm = zlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'C';

            zlacpy("F", mc, nc, C, lda, CC, lda);

            if (k > 0) {
                zunmrq(&side, &trans, mc, nc, k, &AF[(m - k) + 0 * lda], lda,
                       &tau[minmn - k], CC, lda, work, lwork, &info);
            }

            if (side == 'L') {
                cblas_zgemm(CblasColMajor,
                            trans == 'N' ? CblasNoTrans : CblasConjTrans,
                            CblasNoTrans,
                            mc, nc, mc, &CNEGONE, Q, lda, C, lda, &CONE, CC, lda);
            } else {
                cblas_zgemm(CblasColMajor, CblasNoTrans,
                            trans == 'N' ? CblasNoTrans : CblasConjTrans,
                            mc, nc, nc, &CNEGONE, C, lda, Q, lda, &CONE, CC, lda);
            }

            resid = zlange("1", mc, nc, CC, lda, rwork);
            result[iside * 2 + itrans] = resid /
                ((f64)(n > 1 ? n : 1) * cnorm * eps);
        }
    }
}
