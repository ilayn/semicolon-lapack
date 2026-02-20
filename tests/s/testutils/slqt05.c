/**
 * @file slqt05.c
 * @brief SLQT05 tests STPLQT and STPMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* const restrict A, const int lda);
extern void stplqt(const int m, const int n, const int l, const int mb,
                   f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb,
                   f32* const restrict T, const int ldt,
                   f32* const restrict work, int* info);
extern void sgemlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int mb,
                    const f32* const restrict V, const int ldv,
                    const f32* const restrict T, const int ldt,
                    f32* const restrict C, const int ldc,
                    f32* const restrict work, int* info);
extern void stpmlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int l, const int mb,
                    const f32* const restrict V, const int ldv,
                    const f32* const restrict T, const int ldt,
                    f32* const restrict A, const int lda,
                    f32* const restrict B, const int ldb,
                    f32* const restrict work, int* info);
/**
 * SLQT05 tests STPLQT and STPMLQT.
 *
 * @param[in]  m       Number of rows in lower part of the test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  l       Number of rows of upper trapezoidal part. 0 <= L <= M.
 * @param[in]  nb      Block size of the test matrix. NB <= N.
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - Q*R |
 *                     result[1] = | I - Q^H*Q |
 *                     result[2] = | Q*C - Q*C |
 *                     result[3] = | Q^H*C - Q^H*C |
 *                     result[4] = | C*Q - C*Q |
 *                     result[5] = | C*Q^H - C*Q^H |
 */
void slqt05(const int m, const int n, const int l, const int nb,
            f32* restrict result)
{
    f32 eps = slamch("E");
    int k = m;
    int n2 = m + n;
    int np1 = (n > 0) ? m + 1 : 1;
    int lwork = n2 * n2 * nb;
    int ldt = nb;
    int info;
    int j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f32* A = malloc(m * n2 * sizeof(f32));
    f32* AF = malloc(m * n2 * sizeof(f32));
    f32* Q = malloc(n2 * n2 * sizeof(f32));
    f32* R = malloc(n2 * n2 * sizeof(f32));
    f32* rwork = malloc(n2 * sizeof(f32));
    f32* work = malloc(lwork * sizeof(f32));
    f32* T = malloc(nb * m * sizeof(f32));
    f32* C = malloc(n2 * m * sizeof(f32));
    f32* CF = malloc(n2 * m * sizeof(f32));
    f32* D = malloc(m * n2 * sizeof(f32));
    f32* DF = malloc(m * n2 * sizeof(f32));

    slaset("F", m, n2, 0.0f, 0.0f, A, m);
    slaset("F", nb, m, 0.0f, 0.0f, T, nb);
    for (j = 1; j <= m; j++) {
        slarnv_rng(2, m - j + 1, &A[(j - 1) + (j - 1) * m], rng_state);
    }
    if (n > 0) {
        for (j = 1; j <= n - l; j++) {
            int col = m + j - 1;
            slarnv_rng(2, m, &A[col * m], rng_state);
        }
    }
    if (l > 0) {
        for (j = 1; j <= l; j++) {
            int col = n + m - l + j - 1;
            slarnv_rng(2, m - j + 1, &A[(j - 1) + col * m], rng_state);
        }
    }

    slacpy("F", m, n2, A, m, AF, m);

    stplqt(m, n, l, nb, AF, m, &AF[(np1 - 1) * m], m, T, ldt, work, &info);

    slaset("F", n2, n2, 0.0f, 1.0f, Q, n2);
    sgemlqt("L", "N", n2, n2, k, nb, AF, m, T, ldt, Q, n2, work, &info);

    slaset("F", n2, n2, 0.0f, 0.0f, R, n2);
    slacpy("L", m, n2, AF, m, R, n2);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n2, n2, -1.0f, A, m, Q, n2, 1.0f, R, n2);
    anorm = slange("1", m, n2, A, m, rwork);
    resid = slange("1", m, n2, R, n2, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * anorm * (n2 > 1 ? n2 : 1));
    } else {
        result[0] = 0.0f;
    }

    slaset("F", n2, n2, 0.0f, 1.0f, R, n2);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n2, n2, -1.0f, Q, n2, 1.0f, R, n2);
    resid = slansy("1", "U", n2, R, n2, rwork);
    result[1] = resid / (eps * (n2 > 1 ? n2 : 1));

    slaset("F", n2, m, 0.0f, 1.0f, C, n2);
    for (j = 0; j < m; j++) {
        slarnv_rng(2, n2, &C[j * n2], rng_state);
    }
    cnorm = slange("1", n2, m, C, n2, rwork);
    slacpy("F", n2, m, C, n2, CF, n2);

    stpmlqt("L", "N", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n2, m, n2, -1.0f, Q, n2, C, n2, 1.0f, CF, n2);
    resid = slange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    slacpy("F", n2, m, C, n2, CF, n2);

    stpmlqt("L", "T", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n2, m, n2, -1.0f, Q, n2, C, n2, 1.0f, CF, n2);
    resid = slange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < n2; j++) {
        slarnv_rng(2, m, &D[j * m], rng_state);
    }
    dnorm = slange("1", m, n2, D, m, rwork);
    slacpy("F", m, n2, D, m, DF, m);

    stpmlqt("R", "N", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n2, n2, -1.0f, D, m, Q, n2, 1.0f, DF, m);
    resid = slange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0f) {
        result[4] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    slacpy("F", m, n2, D, m, DF, m);

    stpmlqt("R", "T", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n2, n2, -1.0f, D, m, Q, n2, 1.0f, DF, m);
    resid = slange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0f) {
        result[5] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[5] = 0.0f;
    }

    free(A);
    free(AF);
    free(Q);
    free(R);
    free(rwork);
    free(work);
    free(T);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
