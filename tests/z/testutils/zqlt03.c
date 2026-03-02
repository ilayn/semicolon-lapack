/**
 * @file zqlt03.c
 * @brief ZQLT03 tests ZUNMQL, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of ZUNMQL with the results of forming Q explicitly
 * via ZUNGQL and then performing ZGEMM.
 *
 * RESULT(0) = norm( Q*C - Q*C )  / ( M * norm(C) * EPS )   [Left, NoTrans]
 * RESULT(1) = norm( C*Q - C*Q )  / ( M * norm(C) * EPS )   [Right, NoTrans]
 * RESULT(2) = norm( Q'*C - Q'*C) / ( M * norm(C) * EPS )   [Left, ConjTrans]
 * RESULT(3) = norm( C*Q' - C*Q') / ( M * norm(C) * EPS )   [Right, ConjTrans]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/* Simple xoshiro256+ for generating random test matrices */
static uint64_t zqlt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t zqlt03_next(void) {
    uint64_t* s = zqlt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f64 zqlt03_drand(void) {
    uint64_t x = zqlt03_next();
    return ((f64)(x >> 11)) * 0x1.0p-53 * 2.0 - 1.0;
}

/**
 * @param[in]     m       Order of the unitary matrix Q. m >= 0.
 * @param[in]     n       Number of rows or columns of C (see ZUNMQL docs).
 * @param[in]     k       Number of reflectors. m >= k >= 0.
 * @param[in]     AF      QL factorization from ZGEQLF, dimension (lda, n).
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
void zqlt03(const INT m, const INT n, const INT k,
            const c128* const restrict AF,
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

    /* Quick return if possible */
    if (minmn == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 0.0;
        result[3] = 0.0;
        return;
    }

    /* Copy the last k columns of the factorization to Q and generate Q */
    zlaset("F", m, m, ROGUE, ROGUE, Q, lda);
    if (k > 0 && m > k) {
        zlacpy("F", m - k, k, &AF[0 + (n - k) * lda], lda,
               &Q[0 + (m - k) * lda], lda);
    }
    if (k > 1) {
        zlacpy("U", k - 1, k - 1, &AF[(m - k) + (n - k + 1) * lda], lda,
               &Q[(m - k) + (m - k + 1) * lda], lda);
    }

    /* Generate the m-by-m matrix Q */
    zungql(m, m, k, Q, lda, &tau[minmn - k], work, lwork, &info);

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
                C[i + j * lda] = CMPLX(zqlt03_drand(), zqlt03_drand());
            }
        }
        cnorm = zlange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0) cnorm = 1.0;

        for (INT itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'C';

            /* Copy C to CC */
            zlacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via ZUNMQL */
            if (k > 0) {
                zunmql(&side, &trans, mc, nc, k, &AF[0 + (n - k) * lda], lda,
                       &tau[minmn - k], CC, lda, work, lwork, &info);
            }

            /* Form explicit product and subtract */
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

            /* Compute error */
            resid = zlange("1", mc, nc, CC, lda, rwork);
            result[iside * 2 + itrans] = resid /
                ((f64)(m > 1 ? m : 1) * cnorm * eps);
        }
    }
}
