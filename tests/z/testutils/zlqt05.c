/**
 * @file zlqt05.c
 * @brief ZLQT05 tests ZTPLQT and ZTPMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * ZLQT05 tests ZTPLQT and ZTPMLQT.
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
void zlqt05(const INT m, const INT n, const INT l, const INT nb,
            f64* restrict result)
{
    f64 eps = dlamch("E");
    INT k = m;
    INT n2 = m + n;
    INT np1 = (n > 0) ? m + 1 : 1;
    INT lwork = n2 * n2 * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    c128* A = malloc(m * n2 * sizeof(c128));
    c128* AF = malloc(m * n2 * sizeof(c128));
    c128* Q = malloc(n2 * n2 * sizeof(c128));
    c128* R = malloc(n2 * n2 * sizeof(c128));
    f64* rwork = malloc(n2 * sizeof(f64));
    c128* work = malloc(lwork * sizeof(c128));
    c128* T = malloc(nb * m * sizeof(c128));
    c128* C = malloc(n2 * m * sizeof(c128));
    c128* CF = malloc(n2 * m * sizeof(c128));
    c128* D = malloc(m * n2 * sizeof(c128));
    c128* DF = malloc(m * n2 * sizeof(c128));

    zlaset("F", m, n2, CZERO, CZERO, A, m);
    zlaset("F", nb, m, CZERO, CZERO, T, nb);
    for (j = 1; j <= m; j++) {
        zlarnv_rng(2, m - j + 1, &A[(j - 1) + (j - 1) * m], rng_state);
    }
    if (n > 0) {
        for (j = 1; j <= n - l; j++) {
            INT col = m + j - 1;
            zlarnv_rng(2, m, &A[col * m], rng_state);
        }
    }
    if (l > 0) {
        for (j = 1; j <= l; j++) {
            INT col = n + m - l + j - 1;
            zlarnv_rng(2, m - j + 1, &A[(j - 1) + col * m], rng_state);
        }
    }

    zlacpy("F", m, n2, A, m, AF, m);

    ztplqt(m, n, l, nb, AF, m, &AF[(np1 - 1) * m], m, T, ldt, work, &info);

    zlaset("F", n2, n2, CZERO, CONE, Q, n2);
    zgemlqt("L", "N", n2, n2, k, nb, AF, m, T, ldt, Q, n2, work, &info);

    zlaset("F", n2, n2, CZERO, CZERO, R, n2);
    zlacpy("L", m, n2, AF, m, R, n2);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n2, n2, &CNEGONE, A, m, Q, n2, &CONE, R, n2);
    anorm = zlange("1", m, n2, A, m, rwork);
    resid = zlange("1", m, n2, R, n2, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * anorm * (n2 > 1 ? n2 : 1));
    } else {
        result[0] = 0.0;
    }

    zlaset("F", n2, n2, CZERO, CONE, R, n2);
    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n2, n2, -1.0, Q, n2, 1.0, R, n2);
    resid = zlansy("1", "U", n2, R, n2, rwork);
    result[1] = resid / (eps * (n2 > 1 ? n2 : 1));

    zlaset("F", n2, m, CZERO, CONE, C, n2);
    for (j = 0; j < m; j++) {
        zlarnv_rng(2, n2, &C[j * n2], rng_state);
    }
    cnorm = zlange("1", n2, m, C, n2, rwork);
    zlacpy("F", n2, m, C, n2, CF, n2);

    ztpmlqt("L", "N", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n2, m, n2, &CNEGONE, Q, n2, C, n2, &CONE, CF, n2);
    resid = zlange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    zlacpy("F", n2, m, C, n2, CF, n2);

    ztpmlqt("L", "C", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n2, m, n2, &CNEGONE, Q, n2, C, n2, &CONE, CF, n2);
    resid = zlange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < n2; j++) {
        zlarnv_rng(2, m, &D[j * m], rng_state);
    }
    dnorm = zlange("1", m, n2, D, m, rwork);
    zlacpy("F", m, n2, D, m, DF, m);

    ztpmlqt("R", "N", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n2, n2, &CNEGONE, D, m, Q, n2, &CONE, DF, m);
    resid = zlange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0) {
        result[4] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    zlacpy("F", m, n2, D, m, DF, m);

    ztpmlqt("R", "C", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n2, n2, &CNEGONE, D, m, Q, n2, &CONE, DF, m);
    resid = zlange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0) {
        result[5] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[5] = 0.0;
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
