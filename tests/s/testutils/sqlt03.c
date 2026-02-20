/**
 * @file sqlt03.c
 * @brief SQLT03 tests SORMQL, which computes Q*C, Q'*C, C*Q or C*Q'.
 *
 * Compares the results of SORMQL with the results of forming Q explicitly
 * via SORGQL and then performing DGEMM.
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
extern void sorgql(const int m, const int n, const int k,
                   f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);
extern void sormql(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict C, const int ldc,
                   f32* const restrict work, const int lwork, int* info);

/* Simple xoshiro256+ for generating random test matrices */
static uint64_t sqlt03_state[4] = {1988, 1989, 1990, 1991};

static uint64_t sqlt03_next(void) {
    uint64_t* s = sqlt03_state;
    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

static f32 sqlt03_srand(void) {
    union { uint32_t u; float f; } x;
    uint32_t r = (uint32_t)(sqlt03_next() >> 32);
    r >>= 9;
    x.u = ((uint32_t)128 << 23) | r;
    return x.f - 3.0f;
}

/**
 * @param[in]     m       Order of the orthogonal matrix Q. m >= 0.
 * @param[in]     n       Number of rows or columns of C (see SORMQL docs).
 * @param[in]     k       Number of reflectors. m >= k >= 0.
 * @param[in]     AF      QL factorization from SGEQLF, dimension (lda, n).
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
void sqlt03(const int m, const int n, const int k,
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

    /* Copy the last k columns of the factorization to Q and generate Q.
     * For QL with k reflectors, they are stored in AF(0:m-1, n-k:n-1).
     * We copy:
     *   - AF(0:m-k-1, n-k:n-1) -> Q(0:m-k-1, m-k:m-1) if k < m
     *   - Upper of AF(m-k:m-2, n-k+1:n-1) -> Q(m-k:m-2, m-k+1:m-1) if k > 1
     */
    slaset("F", m, m, -1.0e+10f, -1.0e+10f, Q, lda);
    if (k > 0 && m > k) {
        slacpy("F", m - k, k, &AF[0 + (n - k) * lda], lda,
               &Q[0 + (m - k) * lda], lda);
    }
    if (k > 1) {
        slacpy("U", k - 1, k - 1, &AF[(m - k) + (n - k + 1) * lda], lda,
               &Q[(m - k) + (m - k + 1) * lda], lda);
    }

    /* Generate the m-by-m matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    sorgql(m, m, k, Q, lda, &tau[minmn - k], work, lwork, &info);

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
                C[i + j * lda] = sqlt03_srand();
            }
        }
        cnorm = slange("1", mc, nc, C, lda, rwork);
        if (cnorm == 0.0f) cnorm = 1.0f;

        for (int itrans = 0; itrans < 2; itrans++) {
            char trans = (itrans == 0) ? 'N' : 'T';

            /* Copy C to CC */
            slacpy("F", mc, nc, C, lda, CC, lda);

            /* Apply Q or Q' via SORMQL.
             * SORMQL uses AF(0:m-1, n-k:n-1) and tau[minmn-k:minmn-1]. */
            if (k > 0) {
                sormql(&side, &trans, mc, nc, k, &AF[0 + (n - k) * lda], lda,
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
                ((f32)(m > 1 ? m : 1) * cnorm * eps);
        }
    }
}
