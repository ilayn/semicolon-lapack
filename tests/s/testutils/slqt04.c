/**
 * @file slqt04.c
 * @brief SLQT04 tests SGELQT and SGEMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * SLQT04 tests SGELQT and SGEMLQT.
 *
 * @param[in]  m       Number of rows of the test matrix.
 * @param[in]  n       Number of columns of the test matrix.
 * @param[in]  nb      Block size of the test matrix. NB <= min(M,N).
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - L*Q |
 *                     result[1] = | I - Q*Q^H |
 *                     result[2] = | Q*C - Q*C |
 *                     result[3] = | Q^H*C - Q^H*C |
 *                     result[4] = | C*Q - C*Q |
 *                     result[5] = | C*Q^H - C*Q^H |
 */
void slqt04(const INT m, const INT n, const INT nb, f32* restrict result)
{
    f32 eps = slamch("E");
    INT k = m < n ? m : n;
    INT ll = m > n ? m : n;
    INT lwork = ll * ll * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f32* A = malloc(m * n * sizeof(f32));
    f32* AF = malloc(m * n * sizeof(f32));
    f32* Q = malloc(n * n * sizeof(f32));
    f32* L = malloc(ll * n * sizeof(f32));
    f32* rwork = malloc(ll * sizeof(f32));
    f32* work = malloc(lwork * sizeof(f32));
    f32* T = malloc(nb * n * sizeof(f32));
    f32* C = malloc(m * n * sizeof(f32));
    f32* CF = malloc(m * n * sizeof(f32));
    f32* D = malloc(n * m * sizeof(f32));
    f32* DF = malloc(n * m * sizeof(f32));

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m, &A[j * m], rng_state);
    }
    slacpy("F", m, n, A, m, AF, m);

    sgelqt(m, n, nb, AF, m, T, ldt, work, &info);

    slaset("F", n, n, 0.0f, 1.0f, Q, n);
    sgemlqt("R", "N", n, n, k, nb, AF, m, T, ldt, Q, n, work, &info);

    slaset("F", m, n, 0.0f, 0.0f, L, ll);
    slacpy("L", m, n, AF, m, L, ll);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -1.0f, A, m, Q, n, 1.0f, L, ll);
    anorm = slange("1", m, n, A, m, rwork);
    resid = slange("1", m, n, L, ll, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0f;
    }

    slaset("F", n, n, 0.0f, 1.0f, L, ll);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, n, -1.0f, Q, n, 1.0f, L, ll);
    resid = slansy("1", "U", n, L, ll, rwork);
    result[1] = resid / (eps * (n > 1 ? n : 1));

    for (j = 0; j < m; j++) {
        slarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = slange("1", n, m, D, n, rwork);
    slacpy("F", n, m, D, n, DF, n);

    sgemlqt("L", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, n, -1.0f, Q, n, D, n, 1.0f, DF, n);
    resid = slange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[2] = 0.0f;
    }

    slacpy("F", n, m, D, n, DF, n);

    sgemlqt("L", "T", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, m, n, -1.0f, Q, n, D, n, 1.0f, DF, n);
    resid = slange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = slange("1", m, n, C, m, rwork);
    slacpy("F", m, n, C, m, CF, m);

    sgemlqt("R", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, -1.0f, C, m, Q, n, 1.0f, CF, m);
    resid = slange("1", n, m, DF, n, rwork);
    if (cnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    slacpy("F", m, n, C, m, CF, m);

    sgemlqt("R", "T", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -1.0f, C, m, Q, n, 1.0f, CF, m);
    resid = slange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[5] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[5] = 0.0f;
    }

    free(A);
    free(AF);
    free(Q);
    free(L);
    free(rwork);
    free(work);
    free(T);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
