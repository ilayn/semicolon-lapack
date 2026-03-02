/**
 * @file zlqt04.c
 * @brief ZLQT04 tests ZGELQT and ZGEMLQT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * ZLQT04 tests ZGELQT and ZGEMLQT.
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
void zlqt04(const INT m, const INT n, const INT nb, f64* restrict result)
{
    f64 eps = dlamch("E");
    INT k = m < n ? m : n;
    INT ll = m > n ? m : n;
    INT lwork = ll * ll * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    c128* A = malloc(m * n * sizeof(c128));
    c128* AF = malloc(m * n * sizeof(c128));
    c128* Q = malloc(n * n * sizeof(c128));
    c128* L = malloc(ll * n * sizeof(c128));
    f64* rwork = malloc(ll * sizeof(f64));
    c128* work = malloc(lwork * sizeof(c128));
    c128* T = malloc(nb * n * sizeof(c128));
    c128* C = malloc(m * n * sizeof(c128));
    c128* CF = malloc(m * n * sizeof(c128));
    c128* D = malloc(n * m * sizeof(c128));
    c128* DF = malloc(n * m * sizeof(c128));

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &A[j * m], rng_state);
    }
    zlacpy("F", m, n, A, m, AF, m);

    zgelqt(m, n, nb, AF, m, T, ldt, work, &info);

    zlaset("F", n, n, CZERO, CONE, Q, n);
    zgemlqt("R", "N", n, n, k, nb, AF, m, T, ldt, Q, n, work, &info);

    zlaset("F", ll, n, CZERO, CZERO, L, ll);
    zlacpy("L", m, n, AF, m, L, ll);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, m, Q, n, &CONE, L, ll);
    anorm = zlange("1", m, n, A, m, rwork);
    resid = zlange("1", m, n, L, ll, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0;
    }

    zlaset("F", n, n, CZERO, CONE, L, ll);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, -1.0, Q, n, 1.0, L, ll);
    resid = zlansy("1", "U", n, L, ll, rwork);
    result[1] = resid / (eps * (n > 1 ? n : 1));

    for (j = 0; j < m; j++) {
        zlarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = zlange("1", n, m, D, n, rwork);
    zlacpy("F", n, m, D, n, DF, n);

    zgemlqt("L", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, n, D, n, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[2] = 0.0;
    }

    zlacpy("F", n, m, D, n, DF, n);

    zgemlqt("L", "C", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, n, D, n, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = zlange("1", m, n, C, m, rwork);
    zlacpy("F", m, n, C, m, CF, m);

    zgemlqt("R", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, &CNEGONE, C, m, Q, n, &CONE, CF, m);
    resid = zlange("1", n, m, DF, n, rwork);
    if (cnorm > 0.0) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    zlacpy("F", m, n, C, m, CF, m);

    zgemlqt("R", "C", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, C, m, Q, n, &CONE, CF, m);
    resid = zlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[5] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[5] = 0.0;
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
