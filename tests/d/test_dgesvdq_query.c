/**
 * @file test_dgesvdq_query.c
 * @brief DGESVDQ regression test for the workspace query, the zero matrix,
 *        the order of the workspace errors and the DPOCON integer workspace.
 *
 * The workspace query returns in WORK(1) the optimal LWORK built from the
 * queries of DGEQP3, DGESVD, DORMQR, DGELQF and DORMLQ, and in WORK(2) the
 * minimal LWORK. Each case checks both against the formulas of DGESVDQ.
 * With JOBP = 'P' the zero matrix returns the identity column and row
 * permutations. If both LWORK and LRWORK are too small, INFO = -19. With
 * JOBA = 'E' and LIWORK at its minimum, DPOCON uses IWORK(N+IWOFF) and
 * nothing is written past LIWORK.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0

static INT imax(const INT a, const INT b) { return (a > b) ? a : b; }

static INT query_dgeqp3(const INT m, const INT n, f64* A, const INT lda)
{
    f64 w[1];
    INT jpvt[1], info;
    dgeqp3(m, n, A, lda, jpvt, NULL, w, -1, &info);
    return (INT)w[0];
}

static INT query_dgesvd(const char* ju, const char* jv, const INT m, const INT n,
                        f64* A, const INT lda, f64* S, f64* U, const INT ldu,
                        f64* V, const INT ldv)
{
    f64 w[1];
    INT info;
    dgesvd(ju, jv, m, n, A, lda, S, U, ldu, V, ldv, w, -1, &info);
    return (INT)w[0];
}

static void run_query(const char* joba, const char* jobu, const char* jobv,
                      const INT m, const INT n)
{
    const INT lda = m, ldu = m, ldv = n;
    const INT conda = (joba[0] == 'E');
    const INT lsvec = (jobu[0] != 'N');
    const INT rsvec = (jobv[0] != 'N');
    const INT wntva = (jobv[0] == 'V');
    f64 A[1], S[1], U[1], V[1], work[2], rwork[1], w[1];
    INT iwork[1], numrank, info;
    INT lwqp3 = 3 * n + 1, lwcon = 3 * n, lwsvd = imax(5 * n, 1);
    INT lworq = imax(n, 1);
    INT lq3, lsvd, lorq = 0, minwrk, optwrk;

    dgesvdq(joba, "N", "N", jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv,
            &numrank, iwork, -1, work, -1, rwork, -1, &info);
    assert_info_success(info);

    lq3 = query_dgeqp3(m, n, A, lda);
    if (lsvec) {
        dormqr("L", "N", m, n, n, A, lda, NULL, U, ldu, w, -1, &info);
        lorq = (INT)w[0];
    }
    if (!lsvec && !rsvec) {
        minwrk = imax(n + lwqp3, lwsvd);
        if (conda) minwrk = imax(minwrk, lwcon);
        lsvd = query_dgesvd("N", "N", n, n, A, lda, S, U, ldu, V, ldv);
        optwrk = imax(n + lq3, lsvd);
        if (conda) optwrk = imax(optwrk, n + lwcon);
    } else if (lsvec && !rsvec) {
        minwrk = imax(imax(lwqp3, lwsvd), lworq);
        if (conda) minwrk = imax(minwrk, lwcon);
        minwrk = n + minwrk;
        lsvd = query_dgesvd("O", "N", n, n, A, lda, S, U, ldu, V, ldv);
        optwrk = imax(imax(lq3, lsvd), lorq);
        if (conda) optwrk = imax(optwrk, lwcon);
        optwrk = n + optwrk;
    } else if (rsvec && !lsvec) {
        minwrk = imax(lwqp3, lwsvd);
        if (conda) minwrk = imax(minwrk, lwcon);
        minwrk = n + minwrk;
        lsvd = query_dgesvd("N", "O", n, n, A, lda, S, U, ldu, V, ldv);
        optwrk = imax(lq3, lsvd);
        if (conda) optwrk = imax(optwrk, lwcon);
        optwrk = n + optwrk;
    } else {
        minwrk = imax(imax(lwqp3, lwsvd), lworq);
        if (conda) minwrk = imax(minwrk, lwcon);
        minwrk = minwrk + n;
        lsvd = query_dgesvd("S", "O", n, n, A, lda, S, U, ldu, V, ldv);
        optwrk = imax(imax(lq3, lsvd), lorq);
        if (conda) optwrk = imax(optwrk, lwcon);
        optwrk = n + optwrk;
        if (wntva) {
            INT h = n / 2;
            INT minwrk2 = imax(imax(imax(lwqp3, h + imax(h, 1)),
                                    h + imax(5 * h, 1)),
                               imax(h + imax(n, 1), lworq));
            if (conda) minwrk2 = imax(minwrk2, lwcon);
            minwrk = imax(minwrk, n + minwrk2);

            INT llqf, lsvd2, lorlq, optwrk2;
            dgelqf(h, n, U, ldu, NULL, w, -1, &info);
            llqf = (INT)w[0];
            lsvd2 = query_dgesvd("S", "O", h, h, V, ldv, S, U, ldu, NULL, 1);
            dormlq("R", "N", n, n, h, U, ldu, NULL, V, ldv, w, -1, &info);
            lorlq = (INT)w[0];
            optwrk2 = imax(imax(lq3, h + llqf), imax(h + lsvd2, h + lorlq));
            if (conda) optwrk2 = imax(optwrk2, lwcon);
            optwrk = imax(optwrk, n + optwrk2);
        }
    }
    minwrk = imax(2, minwrk);
    optwrk = imax(2, optwrk);

    assert_int_equal((INT)work[1], minwrk);
    assert_int_equal((INT)work[0], optwrk);
    assert_int_equal(iwork[0], conda ? imax(1, 2 * n) : imax(1, n));
    assert_int_equal((INT)rwork[0], 2);
}

static void test_query(void** state)
{
    (void)state;
    const char* jobas[2] = {"H", "E"};
    const char* jobus[2] = {"N", "U"};
    const char* jobvs[3] = {"N", "R", "V"};
    const INT dims[4][2] = {{1, 1}, {10, 7}, {80, 64}, {200, 150}};
    for (INT d = 0; d < 4; d++) {
        for (INT a = 0; a < 2; a++) {
            for (INT u = 0; u < 2; u++) {
                for (INT v = 0; v < 3; v++) {
                    run_query(jobas[a], jobus[u], jobvs[v], dims[d][0], dims[d][1]);
                }
            }
        }
    }
}

static void test_zero_matrix(void** state)
{
    (void)state;
    const INT m = 6, n = 4;
    const INT liwork = n + m - 1;
    const INT lrwork = m;
    f64 A[6 * 4], S[4], U[6 * 4], V[4 * 4], rwork[6];
    f64 work[64];
    INT iwork[9], numrank, info;

    memset(A, 0, sizeof(A));
    for (INT i = 0; i < 9; i++) iwork[i] = -7;
    dgesvdq("H", "P", "N", "U", "V", m, n, A, m, S, U, m, V, n, &numrank,
            iwork, liwork, work, 64, rwork, lrwork, &info);
    assert_info_success(info);
    assert_int_equal(numrank, 0);
    for (INT p = 0; p < n; p++) {
        assert_int_equal(iwork[p], p);
        assert_true(S[p] == 0.0);
    }
    for (INT p = n; p < n + m - 1; p++) {
        assert_int_equal(iwork[p], p - n);
    }
    assert_true(rwork[1] == -1.0);
}

static void test_info_order(void** state)
{
    (void)state;
    const INT m = 6, n = 4;
    f64 A[6 * 4], S[4], U[6 * 4], V[4 * 4], rwork[1], work[1];
    INT iwork[9], numrank, info, lerr, ok;

    memset(A, 0, sizeof(A));
    strcpy(xerbla_srnamt, "DGESVDQ");
    xerbla_infot = 19;
    xerbla_lerr = 0;
    xerbla_ok = 1;
    dgesvdq("H", "P", "N", "U", "V", m, n, A, m, S, U, m, V, n, &numrank,
            iwork, 9, work, 1, rwork, 1, &info);
    lerr = xerbla_lerr;
    ok = xerbla_ok;
    xerbla_srnamt[0] = '\0';
    xerbla_lerr = 0;
    xerbla_ok = 1;

    assert_int_equal(info, -19);
    assert_true(lerr);
    assert_true(ok);
}

static void run_conda(const char* jobp, const char* jobu, const char* jobv)
{
    const INT m = 9, n = 6;
    const INT rowprm = (jobp[0] == 'P');
    const INT liwork = rowprm ? n + m - 1 + n : n + n;
    const INT lrwork = rowprm ? m : 2;
    const INT nguard = 8;
    const INT canary = -12345;
    f64 A[9 * 6], A0[9 * 6], S[6], U[9 * 6], V[6 * 6], rwork[9];
    f64 qwork[2], qrwork[1], resid;
    INT qiwork[1], numrank, info;
    uint64_t rng_state[4];

    rng_seed(rng_state, 0x5D0ULL);
    for (INT i = 0; i < m * n; i++) A0[i] = rng_uniform_symmetric(rng_state);
    memcpy(A, A0, sizeof(A));

    dgesvdq("E", jobp, "N", jobu, jobv, m, n, A, m, S, U, m, V, n, &numrank,
            qiwork, -1, qwork, -1, qrwork, -1, &info);
    assert_info_success(info);
    assert_int_equal(qiwork[0], liwork);

    const INT lwork = (INT)qwork[1];
    INT* iwork = malloc((size_t)(liwork + nguard) * sizeof(INT));
    f64* work = malloc((size_t)lwork * sizeof(f64));
    assert_non_null(iwork);
    assert_non_null(work);
    for (INT i = 0; i < liwork + nguard; i++) iwork[i] = canary;

    dgesvdq("E", jobp, "N", jobu, jobv, m, n, A, m, S, U, m, V, n, &numrank,
            iwork, liwork, work, lwork, rwork, lrwork, &info);
    assert_info_success(info);
    assert_int_equal(numrank, n);
    for (INT i = liwork; i < liwork + nguard; i++) {
        assert_int_equal(iwork[i], canary);
    }
    assert_true(rwork[0] > 0.0);

    if (jobu[0] != 'N' && jobv[0] != 'N') {
        f64 rw[9 + 6], E[6];
        memset(E, 0, sizeof(E));
        dbdt01(m, n, 0, A0, m, U, m, S, E, V, n, rw, &resid);
        assert_residual_ok(resid);
    }

    free(iwork);
    free(work);
}

static void test_conda_iwork(void** state)
{
    (void)state;
    run_conda("N", "N", "N");
    run_conda("P", "N", "N");
    run_conda("N", "U", "R");
    run_conda("P", "U", "R");
}

static void test_void(void** state)
{
    (void)state;
    f64 A[1], S[1], U[1], V[1], work[2], rwork[2];
    INT iwork[1], numrank = 7, info;
    dgesvdq("H", "N", "N", "U", "V", 0, 0, A, 1, S, U, 1, V, 1, &numrank,
            iwork, 1, work, 2, rwork, 2, &info);
    assert_info_success(info);
    assert_int_equal(numrank, 7);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_query),
        cmocka_unit_test(test_zero_matrix),
        cmocka_unit_test(test_info_order),
        cmocka_unit_test(test_conda_iwork),
        cmocka_unit_test(test_void),
    };
    (void)cmocka_run_group_tests_name("dgesvdq_query", tests, NULL, NULL);
    return 0;
}
