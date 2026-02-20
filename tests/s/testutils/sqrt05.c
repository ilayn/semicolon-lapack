/**
 * @file sqrt05.c
 * @brief SQRT05 tests STPQRT and STPMQRT.
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
extern void stpqrt(const int m, const int n, const int l, const int nb,
                   f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb,
                   f32* const restrict T, const int ldt,
                   f32* const restrict work, int* info);
extern void sgemqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int nb,
                    const f32* const restrict V, const int ldv,
                    const f32* const restrict T, const int ldt,
                    f32* const restrict C, const int ldc,
                    f32* const restrict work, int* info);
extern void stpmqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int l, const int nb,
                    const f32* const restrict V, const int ldv,
                    const f32* const restrict T, const int ldt,
                    f32* const restrict A, const int lda,
                    f32* const restrict B, const int ldb,
                    f32* const restrict work, int* info);
/**
 * SQRT05 tests STPQRT and STPMQRT.
 *
 * @param[in]  m       Number of rows in lower part of the test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  l       The number of rows of the upper trapezoidal part
 *                     the lower test matrix. 0 <= L <= M.
 * @param[in]  nb      Block size of the test matrix. NB <= N.
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - Q R |
 *                     result[1] = | I - Q^H Q |
 *                     result[2] = | Q C - Q C |
 *                     result[3] = | Q^H C - Q^H C |
 *                     result[4] = | C Q - C Q |
 *                     result[5] = | C Q^H - C Q^H |
 */
void sqrt05(const int m, const int n, const int l, const int nb, f32* restrict result)
{
    f32 eps = slamch("E");
    int k = n;
    int m2 = m + n;
    int np1 = (m > 0) ? n : 0;
    int lwork = m2 * m2 * nb;
    int ldt = nb;
    int info;
    int j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f32* A = malloc(m2 * n * sizeof(f32));
    f32* AF = malloc(m2 * n * sizeof(f32));
    f32* Q = malloc(m2 * m2 * sizeof(f32));
    f32* R = malloc(m2 * m2 * sizeof(f32));
    f32* rwork = malloc(m2 * sizeof(f32));
    f32* work = malloc(lwork * sizeof(f32));
    f32* T = malloc(nb * n * sizeof(f32));
    f32* C = malloc(m2 * n * sizeof(f32));
    f32* CF = malloc(m2 * n * sizeof(f32));
    f32* D = malloc(n * m2 * sizeof(f32));
    f32* DF = malloc(n * m2 * sizeof(f32));

    slaset("F", m2, n, 0.0f, 0.0f, A, m2);
    slaset("F", nb, n, 0.0f, 0.0f, T, nb);

    for (j = 0; j < n; j++) {
        slarnv_rng(2, j + 1, &A[j * m2], rng_state);
    }
    if (m > 0) {
        int ml = m - l;
        for (j = 0; j < n; j++) {
            if (ml > 0) {
                slarnv_rng(2, ml, &A[n + j * m2], rng_state);
            }
        }
    }
    if (l > 0) {
        int start_row = n + m - l;
        for (j = 0; j < n; j++) {
            int len = (j + 1 < l) ? (j + 1) : l;
            slarnv_rng(2, len, &A[start_row + j * m2], rng_state);
        }
    }

    slacpy("F", m2, n, A, m2, AF, m2);

    stpqrt(m, n, l, nb, AF, m2, &AF[np1], m2, T, ldt, work, &info);

    slaset("F", m2, m2, 0.0f, 1.0f, Q, m2);
    sgemqrt("R", "N", m2, m2, k, nb, AF, m2, T, ldt, Q, m2, work, &info);

    slaset("F", m2, n, 0.0f, 0.0f, R, m2);
    slacpy("U", m2, n, AF, m2, R, m2);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m2, n, m2, -1.0f, Q, m2, A, m2, 1.0f, R, m2);
    anorm = slange("1", m2, n, A, m2, rwork);
    resid = slange("1", m2, n, R, m2, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * anorm * (m2 > 1 ? m2 : 1));
    } else {
        result[0] = 0.0f;
    }

    slaset("F", m2, m2, 0.0f, 1.0f, R, m2);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                m2, m2, -1.0f, Q, m2, 1.0f, R, m2);
    resid = slansy("1", "U", m2, R, m2, rwork);
    result[1] = resid / (eps * (m2 > 1 ? m2 : 1));

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m2, &C[j * m2], rng_state);
    }
    cnorm = slange("1", m2, n, C, m2, rwork);
    slacpy("F", m2, n, C, m2, CF, m2);

    stpmqrt("L", "N", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m2, n, m2, -1.0f, Q, m2, C, m2, 1.0f, CF, m2);
    resid = slange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    slacpy("F", m2, n, C, m2, CF, m2);

    stpmqrt("L", "T", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m2, n, m2, -1.0f, Q, m2, C, m2, 1.0f, CF, m2);
    resid = slange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < m2; j++) {
        slarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = slange("1", n, m2, D, n, rwork);
    slacpy("F", n, m2, D, n, DF, n);

    stpmqrt("R", "N", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m2, m2, -1.0f, D, n, Q, m2, 1.0f, DF, n);
    resid = slange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    slacpy("F", n, m2, D, n, DF, n);

    stpmqrt("R", "T", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, m2, m2, -1.0f, D, n, Q, m2, 1.0f, DF, n);
    resid = slange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[5] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
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
