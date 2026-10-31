/**
 * @file test_dgeqp3rk_lwork.c
 * @brief DGEQP3RK regression test for the reduced block size when
 *        IWS <= LWORK < LWKOPT and NRHS > 0.
 *
 * With less than the optimal workspace, DGEQP3RK reduces NB so that the
 * blocked code fits in LWORK. The blocked step uses 2*N entries for the
 * column norms and AUXV, and F of size (N+NRHS)-by-NB, so NB must satisfy
 * 2*N + NB*(N+NRHS+1) <= LWORK. The test factors A = [A B] with
 * N > NX, NRHS = 2 and LWORK values between IWS and LWKOPT, and checks
 * that WORK is not written past LWORK and that the factorization of the
 * first N columns is correct.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define THRESH 30.0

static void run_case(const INT m, const INT n, const INT nrhs, const INT lwork)
{
    const INT lda = m;
    const INT minmn = m < n ? m : n;
    const INT nguard = 256;
    const f64 canary = 12345.0;
    const INT lwchk = m * n + n;
    INT info, k;
    f64 maxc2nrmk, relmaxc2nrmk;
    uint64_t rng_state[4];

    f64* A = malloc(lda * (n + nrhs) * sizeof(f64));
    f64* AF = malloc(lda * (n + nrhs) * sizeof(f64));
    f64* tau = malloc(minmn * sizeof(f64));
    f64* work = malloc((lwork + nguard) * sizeof(f64));
    f64* wchk = malloc(lwchk * sizeof(f64));
    INT* jpiv = malloc(n * sizeof(INT));
    INT* iwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(AF); assert_non_null(tau);
    assert_non_null(work); assert_non_null(wchk); assert_non_null(jpiv);
    assert_non_null(iwork);

    rng_seed(rng_state, 0x3B7C1ULL + (uint64_t)lwork);
    for (INT i = 0; i < lda * (n + nrhs); i++) {
        A[i] = rng_uniform_symmetric(rng_state);
        AF[i] = A[i];
    }
    for (INT i = 0; i < lwork + nguard; i++) {
        work[i] = canary;
    }

    dgeqp3rk(m, n, nrhs, minmn, 0.0, 0.0, AF, lda, &k, &maxc2nrmk,
             &relmaxc2nrmk, jpiv, tau, work, lwork, iwork, &info);
    assert_info_success(info);
    assert_int_equal(k, minmn);

    INT nover = 0;
    for (INT i = lwork; i < lwork + nguard; i++) {
        if (work[i] != canary) nover++;
    }
    if (nover != 0) {
        fail_msg("%d of %d guard elements past WORK(LWORK) were written, lwork = %d",
                 (int)nover, (int)nguard, (int)lwork);
    }

    f64 resid = dqpt01(m, n, k, A, AF, lda, tau, jpiv, wchk, lwchk);
    assert_residual_below(resid, THRESH);

    free(A); free(AF); free(tau); free(work); free(wchk); free(jpiv);
    free(iwork);
}

static void test_reduced_nb(void** state)
{
    (void)state;
    const INT m = 140;
    const INT n = 140;
    const INT nrhs = 2;
    const INT iws = 3 * n + nrhs - 1;
    INT lwopt, k, info, jpiv[1], iwork[1];
    f64 query, maxc2nrmk, relmaxc2nrmk, a[1], tau[1];

    dgeqp3rk(m, n, nrhs, n, 0.0, 0.0, a, m, &k, &maxc2nrmk, &relmaxc2nrmk,
             jpiv, tau, &query, -1, iwork, &info);
    assert_info_success(info);
    lwopt = (INT)query;
    assert_true(lwopt > iws);

    const INT lworks[5] = { 1130, 2 * n + 2 * (n + nrhs + 1) - 1,
                            2 * n + 6 * (n + nrhs + 1) + 3, (iws + lwopt) / 2,
                            lwopt - 1 };
    for (INT i = 0; i < 5; i++) {
        run_case(m, n, nrhs, lworks[i]);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reduced_nb),
    };
    (void)cmocka_run_group_tests_name("dgeqp3rk_lwork", tests, NULL, NULL);
    return 0;
}
