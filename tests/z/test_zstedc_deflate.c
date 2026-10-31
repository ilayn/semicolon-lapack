/**
 * @file test_zstedc_deflate.c
 * @brief ZSTEDC regression test for the ZLAED8 deflation insertion
 *        (COMPZ = 'V', N > SMLSIZ).
 *
 * With COMPZ = 'V' and N > SMLSIZ, ZSTEDC runs ZLAED0, which merges
 * subproblems with ZLAED7 and deflates close eigenvalues in ZLAED8. When a
 * deflated eigenvalue is inserted into the sorted deflated list INDXP, the
 * insertion must start at the slot just filled. Starting one slot lower
 * reads an INDXP entry that has not been written yet, so the eigenvectors
 * depend on what IWORK held on entry. Each case builds a tridiagonal matrix
 * with heavy deflation (diagonal entries 1, 2 or 3 with off-diagonals of
 * size 0.1*sqrt(EPS), or a zero diagonal with unit off-diagonals), calls
 * ZSTEDC with Z = I and IWORK filled with 0 or with N - 1, and checks with
 * ZSTT21 that A = Z * D * Z**H and Z**H * Z = I.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0

static void run_case(const INT n, const INT type, const INT fill)
{
    INT info, liwork, lrwork, lwork;
    INT iwq;
    f64 rwq, dumma[1];
    c128 wq;
    f64 result[2];
    uint64_t rng_state[4];
    const f64 offd = 0.1 * sqrt(dlamch("E"));

    f64* AD = malloc((size_t)n * sizeof(f64));
    f64* AE = malloc((size_t)n * sizeof(f64));
    f64* D = malloc((size_t)n * sizeof(f64));
    f64* E = malloc((size_t)n * sizeof(f64));
    c128* Z = malloc((size_t)n * n * sizeof(c128));
    c128* wchk = malloc((size_t)n * n * sizeof(c128));
    f64* rwchk = malloc((size_t)n * sizeof(f64));
    assert_non_null(AD); assert_non_null(AE); assert_non_null(D);
    assert_non_null(E); assert_non_null(Z); assert_non_null(wchk);
    assert_non_null(rwchk);

    rng_seed(rng_state, 0x57EDCULL + (uint64_t)n);
    for (INT i = 0; i < n; i++) {
        if (type == 0) {
            AD[i] = (f64)(1 + (INT)(3.0 * rng_uniform(rng_state)));
            AE[i] = offd * rng_uniform_symmetric(rng_state);
        } else {
            AD[i] = 0.0;
            AE[i] = 1.0;
        }
    }
    memcpy(D, AD, (size_t)n * sizeof(f64));
    memcpy(E, AE, (size_t)n * sizeof(f64));
    zlaset("F", n, n, CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), Z, n);

    zstedc("V", n, D, E, Z, n, &wq, -1, &rwq, -1, &iwq, -1, &info);
    assert_info_success(info);
    lwork = (INT)creal(wq);
    lrwork = (INT)rwq;
    liwork = iwq;
    c128* work = malloc((size_t)lwork * sizeof(c128));
    f64* rwork = malloc((size_t)lrwork * sizeof(f64));
    INT* iwork = malloc((size_t)liwork * sizeof(INT));
    assert_non_null(work); assert_non_null(rwork); assert_non_null(iwork);
    for (INT i = 0; i < liwork; i++) {
        iwork[i] = fill;
    }

    zstedc("V", n, D, E, Z, n, work, lwork, rwork, lrwork, iwork, liwork,
           &info);
    assert_info_success(info);

    zstt21(n, 0, AD, AE, D, dumma, Z, n, wchk, rwchk, result);
    if (result[0] >= THRESH || result[1] >= THRESH) {
        fail_msg("n = %d, type = %d, IWORK fill = %d: resid = %.3e, orth = %.3e",
                 (int)n, (int)type, (int)fill, (double)result[0], (double)result[1]);
    }

    free(AD); free(AE); free(D); free(E); free(Z); free(wchk); free(rwchk);
    free(work); free(rwork); free(iwork);
}

static void run_type(const INT type)
{
    static const INT sizes[] = {30, 64, 100, 200};
    const INT nsizes = (INT)(sizeof(sizes) / sizeof(sizes[0]));
    for (INT i = 0; i < nsizes; i++) {
        run_case(sizes[i], type, 0);
        run_case(sizes[i], type, sizes[i] - 1);
    }
}

static void test_clustered(void** state)
{
    (void)state;
    run_type(0);
}

static void test_zero_diagonal(void** state)
{
    (void)state;
    run_type(1);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_clustered),
        cmocka_unit_test(test_zero_diagonal),
    };
    (void)cmocka_run_group_tests_name("zstedc_deflate", tests, NULL, NULL);
    return 0;
}
