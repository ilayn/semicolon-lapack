/**
 * @file test_cgeqp3rk_lwork.c
 * @brief CGEQP3RK regression test for the reduced block size when
 *        IWS <= LWORK < LWKOPT and NRHS > 0.
 *
 * With less than the optimal workspace, CGEQP3RK reduces NB so that the
 * blocked code fits in LWORK. The blocked step uses AUXV of length NB and
 * F of size (N+NRHS)-by-NB, so NB must satisfy NB*(N+NRHS+1) <= LWORK.
 * The test factors A = [A B] with N > NX, NRHS = 2, 40 and 600, and LWORK
 * values between IWS and LWKOPT, and checks that WORK is not written past
 * LWORK and that the factorization of the first N columns is correct.
 * NB = (LWORK-2*N)/(N+1) overflows WORK when NB*NRHS > 2*N, which the
 * N = 200, NRHS = 600 cases reach.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define THRESH 30.0f

static void run_case(const INT m, const INT n, const INT nrhs, const INT lwork)
{
    const INT lda = m;
    const INT minmn = m < n ? m : n;
    const INT nguard = 256;
    const c64 canary = CMPLXF(12345.0f, -1.0f);
    const INT lwchk = m * n + n;
    INT info, k;
    f32 maxc2nrmk, relmaxc2nrmk;
    uint64_t rng_state[4];

    c64* A = malloc(lda * (n + nrhs) * sizeof(c64));
    c64* AF = malloc(lda * (n + nrhs) * sizeof(c64));
    c64* tau = malloc(minmn * sizeof(c64));
    c64* work = malloc((lwork + nguard) * sizeof(c64));
    c64* wchk = malloc(lwchk * sizeof(c64));
    f32* rwork = malloc(2 * n * sizeof(f32));
    INT* jpiv = malloc(n * sizeof(INT));
    INT* iwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(AF); assert_non_null(tau);
    assert_non_null(work); assert_non_null(wchk); assert_non_null(rwork);
    assert_non_null(jpiv); assert_non_null(iwork);

    rng_seed(rng_state, 0x3B7C1ULL + (uint64_t)lwork);
    for (INT i = 0; i < lda * (n + nrhs); i++) {
        f32 re = rng_uniform_symmetric_f32(rng_state);
        f32 im = rng_uniform_symmetric_f32(rng_state);
        A[i] = CMPLXF(re, im);
        AF[i] = A[i];
    }
    for (INT i = 0; i < lwork + nguard; i++) {
        work[i] = canary;
    }

    cgeqp3rk(m, n, nrhs, minmn, 0.0f, 0.0f, AF, lda, &k, &maxc2nrmk,
             &relmaxc2nrmk, jpiv, tau, work, lwork, rwork, iwork, &info);
    assert_info_success(info);
    assert_int_equal(k, minmn);

    INT nover = 0;
    for (INT i = lwork; i < lwork + nguard; i++) {
        if (work[i] != canary) nover++;
    }
    if (nover != 0) {
        fail_msg("%d of %d guard elements past WORK(LWORK) were written, nrhs = %d, lwork = %d",
                 (int)nover, (int)nguard, (int)nrhs, (int)lwork);
    }

    f32 resid = cqpt01(m, n, k, A, AF, lda, tau, jpiv, wchk, lwchk);
    assert_residual_below(resid, THRESH);

    free(A); free(AF); free(tau); free(work); free(wchk); free(rwork);
    free(jpiv); free(iwork);
}

static void run_sizes(const INT m, const INT n, const INT nrhs)
{
    const INT iws = n + nrhs - 1;
    INT lwopt, k, info, jpiv[1], iwork[1];
    f32 maxc2nrmk, relmaxc2nrmk, rwork[1];
    c64 query, a[1], tau[1];

    cgeqp3rk(m, n, nrhs, n, 0.0f, 0.0f, a, m, &k, &maxc2nrmk, &relmaxc2nrmk,
             jpiv, tau, &query, -1, rwork, iwork, &info);
    assert_info_success(info);
    lwopt = (INT)crealf(query);
    assert_true(lwopt > iws);

    const INT lworks[4] = { 2 * n + 2 * (n + nrhs + 1) - 1,
                            2 * n + 6 * (n + nrhs + 1) + 3,
                            (iws + lwopt) / 2, lwopt - 1 };
    for (INT i = 0; i < 4; i++) {
        if (lworks[i] >= iws) run_case(m, n, nrhs, lworks[i]);
    }
}

static void test_reduced_nb(void** state)
{
    (void)state;
    run_sizes(140, 140, 2);
    run_sizes(140, 140, 40);
    run_sizes(200, 200, 600);
    run_case(200, 200, 600, 2 * 200 + 30 * (200 + 1));
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reduced_nb),
    };
    (void)cmocka_run_group_tests_name("cgeqp3rk_lwork", tests, NULL, NULL);
    return 0;
}
