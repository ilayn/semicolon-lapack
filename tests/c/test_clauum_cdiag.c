/**
 * @file test_clauum_cdiag.c
 * @brief CLAUU2 and CLAUUM with a triangular factor whose diagonal is not
 *        real.
 *
 * The product U * U**H or L**H * L is compared with a dense product for
 * both UPLO values. N = 70 takes the blocked path of CLAUUM. The diagonal
 * of the result must be real and the other triangle must be untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0f

typedef void (*lauum_fn)(const char*, const INT, c64*, const INT, INT*);

static void check_lauum(lauum_fn fn)
{
    const INT nvals[4] = {1, 2, 5, 70};
    const char* uplos[2] = {"U", "L"};
    const c64 guard = CMPLXF(-7.0f, 3.0f);
    const f32 eps = slamch("P");
    uint64_t rng_state[4];
    rng_seed(rng_state, 0x1A0D2C3BULL);

    for (INT in = 0; in < 4; in++) {
        const INT n = nvals[in];
        const INT lda = n + 1;
        c64* T = malloc(n * n * sizeof(c64));
        c64* A = malloc(lda * n * sizeof(c64));
        c64* R = malloc(n * n * sizeof(c64));
        assert_non_null(T); assert_non_null(A); assert_non_null(R);

        for (INT iu = 0; iu < 2; iu++) {
            const INT upper = (iu == 0);
            clarnv_rng(2, n * n, T, rng_state);
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    if ((upper && i > j) || (!upper && i < j)) {
                        T[i + j * n] = CMPLXF(0.0f, 0.0f);
                    }
                }
            }
            f32 tnorm = 0.0f;
            for (INT k = 0; k < n * n; k++) tnorm = fmaxf(tnorm, cabsf(T[k]));

            /* R = T * T**H (upper) or T**H * T (lower) */
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    c64 s = CMPLXF(0.0f, 0.0f);
                    for (INT p = 0; p < n; p++) {
                        if (upper) {
                            s += T[i + p * n] * conjf(T[j + p * n]);
                        } else {
                            s += conjf(T[p + i * n]) * T[p + j * n];
                        }
                    }
                    R[i + j * n] = s;
                }
            }

            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < lda; i++) {
                    A[i + j * lda] = guard;
                }
            }
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    if ((upper && i <= j) || (!upper && i >= j)) {
                        A[i + j * lda] = T[i + j * n];
                    }
                }
            }

            INT info;
            fn(uplos[iu], n, A, lda, &info);
            assert_info_success(info);

            f32 err = 0.0f;
            INT nbad = 0;
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < lda; i++) {
                    INT ref = (i < n) && ((upper && i <= j) || (!upper && i >= j));
                    if (ref) {
                        f32 d = cabsf(A[i + j * lda] - R[i + j * n]);
                        if (!(d <= err)) err = d;
                        if (i == j && cimagf(A[i + j * lda]) != 0.0f) nbad++;
                    } else if (A[i + j * lda] != guard) {
                        nbad++;
                    }
                }
            }
            assert_int_equal(nbad, 0);
            f32 resid = err / (n * eps * tnorm * tnorm);
            assert_residual_below(resid, THRESH);
        }
        free(T); free(A); free(R);
    }
}

static void test_lauu2(void** state)
{
    (void)state;
    check_lauum(clauu2);
}

static void test_lauum(void** state)
{
    (void)state;
    check_lauum(clauum);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_lauu2),
        cmocka_unit_test(test_lauum),
    };
    (void)cmocka_run_group_tests_name("zlauum_cdiag", tests, NULL, NULL);
    return 0;
}
