/**
 * @file zunhr_col01.c
 * @brief ZUNHR_COL01 tests ZUNGTSQR and ZUNHR_COL using ZLATSQR, ZGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * ZUNHR_COL01 tests ZUNGTSQR and ZUNHR_COL using ZLATSQR, ZGEMQRT.
 *
 * @param[in]  m       Number of rows in test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  mb1     Number of rows in row block.
 * @param[in]  nb1     Number of columns in column block (input).
 * @param[in]  nb2     Number of columns in column block (output).
 * @param[out] result  Results of each of the six tests.
 */
void zunhr_col01(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f64* restrict result)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    f64 eps = dlamch("E");
    INT k = m < n ? m : n;
    INT l = m > n ? m : n;
    if (l < 1) l = 1;
    INT info;
    INT j, i;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    c128* A = malloc(m * n * sizeof(c128));
    c128* AF = malloc(m * n * sizeof(c128));
    c128* Q = malloc(l * l * sizeof(c128));
    c128* R = malloc(m * l * sizeof(c128));
    f64* rwork = malloc(l * sizeof(f64));
    c128* C = malloc(m * n * sizeof(c128));
    c128* CF = malloc(m * n * sizeof(c128));
    c128* D = malloc(n * m * sizeof(c128));
    c128* DF = malloc(n * m * sizeof(c128));

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &A[j * m], rng_state);
    }
    zlacpy("F", m, n, A, m, AF, m);

    INT nrb;
    if (mb1 - n > 0) {
        nrb = (m - n + mb1 - n - 1) / (mb1 - n);
        if (nrb < 1) nrb = 1;
    } else {
        nrb = 1;
    }

    c128* T1 = malloc(nb1 * n * nrb * sizeof(c128));
    c128* T2 = malloc(nb2 * n * sizeof(c128));
    c128* DIAG = malloc(n * sizeof(c128));

    INT nb1_ub = nb1 < n ? nb1 : n;
    INT nb2_ub = nb2 < n ? nb2 : n;

    c128 workquery;
    INT lwork;

    zlatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, &workquery, -1, &info);
    lwork = (INT)creal(workquery);
    zungtsqr(m, n, mb1, nb1, AF, m, T1, nb1, &workquery, -1, &info);
    INT lwork2 = (INT)creal(workquery);
    if (lwork2 > lwork) lwork = lwork2;
    if (nb2_ub * n > lwork) lwork = nb2_ub * n;
    if (nb2_ub * m > lwork) lwork = nb2_ub * m;

    c128* work = malloc(lwork * sizeof(c128));

    zlatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, work, lwork, &info);

    zlacpy("U", n, n, AF, m, R, m);

    zungtsqr(m, n, mb1, nb1, AF, m, T1, nb1, work, lwork, &info);

    zunhr_col(m, n, nb2, AF, m, T2, nb2, DIAG, &info);

    zlacpy("U", n, n, R, m, AF, m);

    for (i = 0; i < n; i++) {
        if (creal(DIAG[i]) == -1.0 && cimag(DIAG[i]) == 0.0) {
            cblas_zscal(n - i, &CNEGONE, &AF[i + i * m], m);
        }
    }

    zlaset("F", m, m, CZERO, CONE, Q, m);

    zgemqrt("L", "N", m, m, k, nb2_ub, AF, m, T2, nb2, Q, m, work, &info);

    zlaset("F", m, n, CZERO, CZERO, R, m);

    zlacpy("U", m, n, AF, m, R, m);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, A, m, &CONE, R, m);

    anorm = zlange("1", m, n, A, m, rwork);
    resid = zlange("1", m, n, R, m, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0;
    }

    zlaset("F", m, m, CZERO, CONE, R, m);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0, Q, m, 1.0, R, m);
    resid = zlansy("1", "U", m, R, m, rwork);
    result[1] = resid / (eps * (m > 1 ? m : 1));

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = zlange("1", m, n, C, m, rwork);
    zlacpy("F", m, n, C, m, CF, m);

    zgemqrt("L", "N", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = zlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    zlacpy("F", m, n, C, m, CF, m);

    zgemqrt("L", "C", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = zlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < m; j++) {
        zlarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = zlange("1", n, m, D, n, rwork);
    zlacpy("F", n, m, D, n, DF, n);

    zgemqrt("R", "N", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    zlacpy("F", n, m, D, n, DF, n);

    zgemqrt("R", "C", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[5] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[5] = 0.0;
    }

    free(A);
    free(AF);
    free(Q);
    free(R);
    free(rwork);
    free(work);
    free(T1);
    free(T2);
    free(DIAG);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
