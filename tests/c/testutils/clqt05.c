/**
 * @file clqt05.c
 * @brief CLQT05 tests CTPLQT and CTPMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * CLQT05 tests CTPLQT and CTPMLQT.
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
void clqt05(const INT m, const INT n, const INT l, const INT nb,
            f32* restrict result)
{
    f32 eps = slamch("E");
    INT k = m;
    INT n2 = m + n;
    INT np1 = (n > 0) ? m + 1 : 1;
    INT lwork = n2 * n2 * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    c64* A = malloc(m * n2 * sizeof(c64));
    c64* AF = malloc(m * n2 * sizeof(c64));
    c64* Q = malloc(n2 * n2 * sizeof(c64));
    c64* R = malloc(n2 * n2 * sizeof(c64));
    f32* rwork = malloc(n2 * sizeof(f32));
    c64* work = malloc(lwork * sizeof(c64));
    c64* T = malloc(nb * m * sizeof(c64));
    c64* C = malloc(n2 * m * sizeof(c64));
    c64* CF = malloc(n2 * m * sizeof(c64));
    c64* D = malloc(m * n2 * sizeof(c64));
    c64* DF = malloc(m * n2 * sizeof(c64));

    claset("F", m, n2, CZERO, CZERO, A, m);
    claset("F", nb, m, CZERO, CZERO, T, nb);
    for (j = 1; j <= m; j++) {
        clarnv_rng(2, m - j + 1, &A[(j - 1) + (j - 1) * m], rng_state);
    }
    if (n > 0) {
        for (j = 1; j <= n - l; j++) {
            INT col = m + j - 1;
            clarnv_rng(2, m, &A[col * m], rng_state);
        }
    }
    if (l > 0) {
        for (j = 1; j <= l; j++) {
            INT col = n + m - l + j - 1;
            clarnv_rng(2, m - j + 1, &A[(j - 1) + col * m], rng_state);
        }
    }

    clacpy("F", m, n2, A, m, AF, m);

    ctplqt(m, n, l, nb, AF, m, &AF[(np1 - 1) * m], m, T, ldt, work, &info);

    claset("F", n2, n2, CZERO, CONE, Q, n2);
    cgemlqt("L", "N", n2, n2, k, nb, AF, m, T, ldt, Q, n2, work, &info);

    claset("F", n2, n2, CZERO, CZERO, R, n2);
    clacpy("L", m, n2, AF, m, R, n2);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n2, n2, &CNEGONE, A, m, Q, n2, &CONE, R, n2);
    anorm = clange("1", m, n2, A, m, rwork);
    resid = clange("1", m, n2, R, n2, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * anorm * (n2 > 1 ? n2 : 1));
    } else {
        result[0] = 0.0f;
    }

    claset("F", n2, n2, CZERO, CONE, R, n2);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n2, n2, -1.0f, Q, n2, 1.0f, R, n2);
    resid = clansy("1", "U", n2, R, n2, rwork);
    result[1] = resid / (eps * (n2 > 1 ? n2 : 1));

    claset("F", n2, m, CZERO, CONE, C, n2);
    for (j = 0; j < m; j++) {
        clarnv_rng(2, n2, &C[j * n2], rng_state);
    }
    cnorm = clange("1", n2, m, C, n2, rwork);
    clacpy("F", n2, m, C, n2, CF, n2);

    ctpmlqt("L", "N", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n2, m, n2, &CNEGONE, Q, n2, C, n2, &CONE, CF, n2);
    resid = clange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    clacpy("F", n2, m, C, n2, CF, n2);

    ctpmlqt("L", "C", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n2, m, n2, &CNEGONE, Q, n2, C, n2, &CONE, CF, n2);
    resid = clange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < n2; j++) {
        clarnv_rng(2, m, &D[j * m], rng_state);
    }
    dnorm = clange("1", m, n2, D, m, rwork);
    clacpy("F", m, n2, D, m, DF, m);

    ctpmlqt("R", "N", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n2, n2, &CNEGONE, D, m, Q, n2, &CONE, DF, m);
    resid = clange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0f) {
        result[4] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    clacpy("F", m, n2, D, m, DF, m);

    ctpmlqt("R", "C", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n2, n2, &CNEGONE, D, m, Q, n2, &CONE, DF, m);
    resid = clange("1", m, n2, DF, m, rwork);
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
