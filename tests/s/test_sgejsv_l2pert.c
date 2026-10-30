/**
 * @file test_sgejsv_l2pert.c
 * @brief SGEJSV regression test for the L2PERT perturbation when only the
 *        singular values are computed.
 *
 * With JOBP = 'P' and a violently scaled triangular factor, SGEJSV perturbs
 * the strict upper triangle of the lower triangular matrix and only the
 * entries below the diagonal with |A(p,q)| <= TEMP1. The perturbation is
 * structured so that it does not change the singular values. Each case
 * computes the singular values of A = D1 * B * D2, with a random B and
 * graded diagonal matrices D1 and D2, with JOBU = JOBV = 'N' and
 * JOBP = 'P', and compares them with SGESVJ relative to n * eps * cond(B).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0f

static void run_case(const INT m, const INT n, const char* joba,
                     const char* jobt, const uint64_t seed)
{
    const INT lda = m;
    const INT lwork = 2 * m + n + n * n + 4 * n + 7;
    const f32 eps = slamch("E");
    INT info;
    f32 condb, err, sjsv, ssvj;
    uint64_t rng_state[4];

    f32* B = malloc((size_t)lda * n * sizeof(f32));
    f32* A = malloc((size_t)lda * n * sizeof(f32));
    f32* A1 = malloc((size_t)lda * n * sizeof(f32));
    f32* sva = calloc(n, sizeof(f32));
    f32* svj = calloc(n, sizeof(f32));
    f32* svb = calloc(n, sizeof(f32));
    f32* work = calloc(lwork, sizeof(f32));
    INT* iwork = calloc(m + 3 * n, sizeof(INT));
    f32 U[1], V[1];
    assert_non_null(B); assert_non_null(A); assert_non_null(A1);
    assert_non_null(sva); assert_non_null(svj); assert_non_null(svb);
    assert_non_null(work); assert_non_null(iwork);

    rng_seed(rng_state, seed);
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < m; i++) {
            B[i + j * lda] = rng_uniform_symmetric_f32(rng_state);
        }
    }
    for (INT j = 0; j < n; j++) {
        const f32 dj = powf(10.0f, -2.0f * (f32)j);
        for (INT i = 0; i < m; i++) {
            A[i + j * lda] = B[i + j * lda] * dj * powf(10.0f, -3.0f * (f32)i);
        }
    }

    memcpy(A1, B, (size_t)lda * n * sizeof(f32));
    sgesvj("G", "N", "N", m, n, A1, lda, svb, 0, V, 1, work, lwork, &info);
    assert_info_success(info);
    condb = svb[0] / svb[n - 1];

    memcpy(A1, A, (size_t)lda * n * sizeof(f32));
    sgesvj("G", "N", "N", m, n, A1, lda, svj, 0, V, 1, work, lwork, &info);
    assert_info_success(info);
    for (INT i = 0; i < n; i++) svj[i] = svj[i] * work[0];

    memcpy(A1, A, (size_t)lda * n * sizeof(f32));
    sgejsv(joba, "N", "N", "N", jobt, "P", m, n, A1, lda, sva, U, 1, V, 1,
           work, lwork, iwork, &info);
    assert_info_success(info);
    assert_int_equal(iwork[0], n);

    err = 0.0f;
    for (INT i = 0; i < n; i++) {
        sjsv = (work[0] / work[1]) * sva[i];
        ssvj = svj[i];
        f32 r = fabsf(sjsv - ssvj) / ssvj;
        if (r > err) err = r;
    }
    err = err / ((f32)n * eps * condb);
    assert_residual_ok(err);

    free(B); free(A); free(A1); free(sva); free(svj); free(svb);
    free(work); free(iwork);
}

static void test_rectangular(void** state)
{
    (void)state;
    const char* jobas[4] = {"C", "E", "F", "G"};
    for (INT k = 0; k < 4; k++) {
        run_case(8, 6, jobas[k], "N", 0x1290ULL + (uint64_t)k);
    }
}

static void test_square(void** state)
{
    (void)state;
    const char* jobas[4] = {"C", "E", "F", "G"};
    for (INT k = 0; k < 4; k++) {
        run_case(6, 6, jobas[k], "N", 0x5A1ULL + (uint64_t)k);
        run_case(6, 6, jobas[k], "T", 0x5A1ULL + (uint64_t)k);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_rectangular),
        cmocka_unit_test(test_square),
    };
    (void)cmocka_run_group_tests_name("sgejsv_l2pert", tests, NULL, NULL);
    return 0;
}
