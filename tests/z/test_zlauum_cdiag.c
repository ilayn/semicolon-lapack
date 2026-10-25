/**
 * @file test_zlauum_cdiag.c
 * @brief ZLAUU2 and ZLAUUM with a triangular factor whose diagonal is not
 *        real.
 *
 * The product U * U**H or L**H * L is compared with a dense product for
 * both UPLO values. N = 70 takes the blocked path of ZLAUUM. The diagonal
 * of the result must be real and the other triangle must be untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0

typedef void (*lauum_fn)(const char*, const INT, c128*, const INT, INT*);

static void check_lauum(lauum_fn fn)
{
    const INT nvals[4] = {1, 2, 5, 70};
    const char* uplos[2] = {"U", "L"};
    const c128 guard = CMPLX(-7.0, 3.0);
    const f64 eps = dlamch("P");
    uint64_t rng_state[4];
    rng_seed(rng_state, 0x1A0D2C3BULL);

    for (INT in = 0; in < 4; in++) {
        const INT n = nvals[in];
        const INT lda = n + 1;
        c128* T = malloc(n * n * sizeof(c128));
        c128* A = malloc(lda * n * sizeof(c128));
        c128* R = malloc(n * n * sizeof(c128));
        assert_non_null(T); assert_non_null(A); assert_non_null(R);

        for (INT iu = 0; iu < 2; iu++) {
            const INT upper = (iu == 0);
            zlarnv_rng(2, n * n, T, rng_state);
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    if ((upper && i > j) || (!upper && i < j)) {
                        T[i + j * n] = CMPLX(0.0, 0.0);
                    }
                }
            }
            f64 tnorm = 0.0;
            for (INT k = 0; k < n * n; k++) tnorm = fmax(tnorm, cabs(T[k]));

            /* R = T * T**H (upper) or T**H * T (lower) */
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    c128 s = CMPLX(0.0, 0.0);
                    for (INT p = 0; p < n; p++) {
                        if (upper) {
                            s += T[i + p * n] * conj(T[j + p * n]);
                        } else {
                            s += conj(T[p + i * n]) * T[p + j * n];
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

            f64 err = 0.0;
            INT nbad = 0;
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < lda; i++) {
                    INT ref = (i < n) && ((upper && i <= j) || (!upper && i >= j));
                    if (ref) {
                        f64 d = cabs(A[i + j * lda] - R[i + j * n]);
                        if (!(d <= err)) err = d;
                        if (i == j && cimag(A[i + j * lda]) != 0.0) nbad++;
                    } else if (A[i + j * lda] != guard) {
                        nbad++;
                    }
                }
            }
            assert_int_equal(nbad, 0);
            f64 resid = err / (n * eps * tnorm * tnorm);
            assert_residual_below(resid, THRESH);
        }
        free(T); free(A); free(R);
    }
}

static void test_lauu2(void** state)
{
    (void)state;
    check_lauum(zlauu2);
}

static void test_lauum(void** state)
{
    (void)state;
    check_lauum(zlauum);
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
