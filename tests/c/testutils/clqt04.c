/**
 * @file clqt04.c
 * @brief CLQT04 tests CGELQT and CGEMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * CLQT04 tests CGELQT and CGEMLQT.
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
void clqt04(const INT m, const INT n, const INT nb, f32* restrict result)
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

    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    c64* A = malloc(m * n * sizeof(c64));
    c64* AF = malloc(m * n * sizeof(c64));
    c64* Q = malloc(n * n * sizeof(c64));
    c64* L = malloc(ll * n * sizeof(c64));
    f32* rwork = malloc(ll * sizeof(f32));
    c64* work = malloc(lwork * sizeof(c64));
    c64* T = malloc(nb * n * sizeof(c64));
    c64* C = malloc(m * n * sizeof(c64));
    c64* CF = malloc(m * n * sizeof(c64));
    c64* D = malloc(n * m * sizeof(c64));
    c64* DF = malloc(n * m * sizeof(c64));

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &A[j * m], rng_state);
    }
    clacpy("F", m, n, A, m, AF, m);

    cgelqt(m, n, nb, AF, m, T, ldt, work, &info);

    claset("F", n, n, CZERO, CONE, Q, n);
    cgemlqt("R", "N", n, n, k, nb, AF, m, T, ldt, Q, n, work, &info);

    claset("F", ll, n, CZERO, CZERO, L, ll);
    clacpy("L", m, n, AF, m, L, ll);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, m, Q, n, &CONE, L, ll);
    anorm = clange("1", m, n, A, m, rwork);
    resid = clange("1", m, n, L, ll, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0f;
    }

    claset("F", n, n, CZERO, CONE, L, ll);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, -1.0f, Q, n, 1.0f, L, ll);
    resid = clansy("1", "U", n, L, ll, rwork);
    result[1] = resid / (eps * (n > 1 ? n : 1));

    for (j = 0; j < m; j++) {
        clarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = clange("1", n, m, D, n, rwork);
    clacpy("F", n, m, D, n, DF, n);

    cgemlqt("L", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, n, D, n, &CONE, DF, n);
    resid = clange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[2] = 0.0f;
    }

    clacpy("F", n, m, D, n, DF, n);

    cgemlqt("L", "C", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, n, D, n, &CONE, DF, n);
    resid = clange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = clange("1", m, n, C, m, rwork);
    clacpy("F", m, n, C, m, CF, m);

    cgemlqt("R", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, &CNEGONE, C, m, Q, n, &CONE, CF, m);
    resid = clange("1", n, m, DF, n, rwork);
    if (cnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    clacpy("F", m, n, C, m, CF, m);

    cgemlqt("R", "C", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, C, m, Q, n, &CONE, CF, m);
    resid = clange("1", m, n, CF, m, rwork);
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
