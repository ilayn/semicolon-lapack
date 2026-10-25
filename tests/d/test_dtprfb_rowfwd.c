/**
 * @file test_dtprfb_rowfwd.c
 * @brief DTPRFB and DTPMLQT regression tests for STOREV = 'R', DIRECT = 'F',
 *        SIDE = 'L'.
 *
 * test_tprfb_ldwork calls DTPRFB with LDB > LDWORK and compares the result
 * with a dense application of H = I - W**T T W, W = [ I V ].
 * test_tpmlqt_left applies the Q of DTPLQT from the left with DTPMLQT while
 * the strictly upper triangle of the trapezoidal block of B, which DTPLQT
 * does not reference, holds nonzero entries, and compares with Q formed by
 * DGEMLQT as in DLQT05.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0

static void test_tprfb_ldwork(void** state)
{
    (void)state;
    const INT m = 6, n = 5, k = 4, l = 3;
    const INT ldv = k, ldt = k, lda = k, ldb = m + 7, ldwork = k;
    const INT nguard = 64;
    const f64 guard = -7.0;
    const char* transs[2] = {"N", "T"};
    const f64 eps = dlamch("P");
    uint64_t rng_state[4];

    f64* V = malloc(ldv * m * sizeof(f64));
    f64* T = malloc(ldt * k * sizeof(f64));
    f64* A = malloc(lda * n * sizeof(f64));
    f64* B = malloc(ldb * n * sizeof(f64));
    f64* Vd = malloc(k * m * sizeof(f64));
    f64* A0 = malloc(k * n * sizeof(f64));
    f64* B0 = malloc(m * n * sizeof(f64));
    f64* Y = malloc(k * n * sizeof(f64));
    f64* Z = malloc(k * n * sizeof(f64));
    f64* work = malloc((ldwork * n + nguard) * sizeof(f64));
    assert_non_null(V); assert_non_null(T); assert_non_null(A);
    assert_non_null(B); assert_non_null(Vd); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Y); assert_non_null(Z);
    assert_non_null(work);

    rng_seed(rng_state, 0x63B2670BULL);
    for (INT it = 0; it < 2; it++) {
        dlarnv_rng(2, ldv * m, V, rng_state);
        dlarnv_rng(2, ldt * k, T, rng_state);
        dlarnv_rng(2, lda * n, A, rng_state);
        for (INT j = 0; j < n; j++) {
            dlarnv_rng(2, m, &B[j * ldb], rng_state);
            for (INT i = m; i < ldb; i++) {
                B[i + j * ldb] = guard;
            }
        }
        for (INT i = 0; i < ldwork * n + nguard; i++) {
            work[i] = guard;
        }
        dlacpy("F", k, m, V, ldv, Vd, k);
        for (INT j = 1; j < l; j++) {
            for (INT i = 0; i < j; i++) {
                Vd[i + (m - l + j) * k] = 0.0;
            }
        }
        dlacpy("F", k, n, A, lda, A0, k);
        dlacpy("F", m, n, B, ldb, B0, m);

        dtprfb("L", transs[it], "F", "R", m, n, k, l, V, ldv, T, ldt,
               A, lda, B, ldb, work, ldwork);

        INT nbad = 0;
        for (INT i = ldwork * n; i < ldwork * n + nguard; i++) {
            if (work[i] != guard) nbad++;
        }
        for (INT j = 0; j < n; j++) {
            for (INT i = m; i < ldb; i++) {
                if (B[i + j * ldb] != guard) nbad++;
            }
        }
        assert_int_equal(nbad, 0);

        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < k; i++) {
                f64 s = A0[i + j * k];
                for (INT p = 0; p < m; p++) {
                    s += Vd[i + p * k] * B0[p + j * m];
                }
                Y[i + j * k] = s;
            }
        }
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < k; i++) {
                f64 s = 0.0;
                if (it == 0) {
                    for (INT p = i; p < k; p++) {
                        s += T[i + p * ldt] * Y[p + j * k];
                    }
                } else {
                    for (INT p = 0; p <= i; p++) {
                        s += T[p + i * ldt] * Y[p + j * k];
                    }
                }
                Z[i + j * k] = s;
            }
        }

        f64 diff = 0.0, cmax = 1.0;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < k; i++) {
                f64 r = A0[i + j * k] - Z[i + j * k];
                f64 d = fabs(A[i + j * lda] - r);
                if (!(d <= diff)) diff = d;
                if (fabs(r) > cmax) cmax = fabs(r);
                if (fabs(Z[i + j * k]) > cmax) cmax = fabs(Z[i + j * k]);
            }
            for (INT p = 0; p < m; p++) {
                f64 r = B0[p + j * m];
                for (INT i = 0; i < k; i++) {
                    r -= Vd[i + p * k] * Z[i + j * k];
                }
                f64 d = fabs(B[p + j * ldb] - r);
                if (!(d <= diff)) diff = d;
                if (fabs(r) > cmax) cmax = fabs(r);
            }
        }
        f64 resid = diff / (eps * (m + k) * cmax);
        assert_residual_ok(resid);
    }

    free(V); free(T); free(A); free(B); free(Vd); free(A0); free(B0);
    free(Y); free(Z); free(work);
}

static void test_tpmlqt_left(void** state)
{
    (void)state;
    const INT m = 5, n = 7, l = 4, nb = 2;
    const INT k = m, n2 = m + n, ldt = nb;
    const INT lwork = n2 * n2 * nb;
    const f64 eps = dlamch("E");
    const char* transs[2] = {"N", "T"};
    INT info;
    uint64_t rng_state[4];

    f64* AF = malloc(m * n2 * sizeof(f64));
    f64* AFZ = malloc(m * n2 * sizeof(f64));
    f64* Q = malloc(n2 * n2 * sizeof(f64));
    f64* T = malloc(nb * m * sizeof(f64));
    f64* C = malloc(n2 * m * sizeof(f64));
    f64* CF = malloc(n2 * m * sizeof(f64));
    f64* rwork = malloc(n2 * sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(AF); assert_non_null(AFZ); assert_non_null(Q);
    assert_non_null(T); assert_non_null(C); assert_non_null(CF);
    assert_non_null(rwork); assert_non_null(work);

    rng_seed(rng_state, 0x1376ULL);
    dlaset("F", m, n2, 0.0, 0.0, AF, m);
    dlaset("F", nb, m, 0.0, 0.0, T, nb);
    for (INT j = 0; j < m; j++) {
        dlarnv_rng(2, m - j, &AF[j + j * m], rng_state);
    }
    for (INT j = 0; j < n - l; j++) {
        dlarnv_rng(2, m, &AF[(m + j) * m], rng_state);
    }
    for (INT j = 0; j < l; j++) {
        dlarnv_rng(2, m - j, &AF[j + (m + n - l + j) * m], rng_state);
    }
    for (INT j = 1; j < l; j++) {
        dlarnv_rng(2, j, &AF[(m + n - l + j) * m], rng_state);
    }

    dtplqt(m, n, l, nb, AF, m, &AF[m * m], m, T, ldt, work, &info);
    assert_info_success(info);

    dlacpy("F", m, n2, AF, m, AFZ, m);
    for (INT j = 1; j < l; j++) {
        for (INT i = 0; i < j; i++) {
            AFZ[i + (m + n - l + j) * m] = 0.0;
        }
    }
    dlaset("F", n2, n2, 0.0, 1.0, Q, n2);
    dgemlqt("L", "N", n2, n2, k, nb, AFZ, m, T, ldt, Q, n2, work, &info);
    assert_info_success(info);

    for (INT j = 0; j < m; j++) {
        dlarnv_rng(2, n2, &C[j * n2], rng_state);
    }
    f64 cnorm = dlange("1", n2, m, C, n2, rwork);

    for (INT it = 0; it < 2; it++) {
        dlacpy("F", n2, m, C, n2, CF, n2);
        dtpmlqt("L", transs[it], n, m, k, l, nb, &AF[m * m], m, T, ldt,
                CF, n2, &CF[m], n2, work, &info);
        assert_info_success(info);
        cblas_dgemm(CblasColMajor, (it == 0) ? CblasNoTrans : CblasTrans,
                    CblasNoTrans, n2, m, n2, -1.0, Q, n2, C, n2, 1.0, CF, n2);
        f64 resid = dlange("1", n2, m, CF, n2, rwork) / (eps * n2 * cnorm);
        assert_residual_ok(resid);
    }

    free(AF); free(AFZ); free(Q); free(T); free(C); free(CF);
    free(rwork); free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_tprfb_ldwork),
        cmocka_unit_test(test_tpmlqt_left),
    };
    (void)cmocka_run_group_tests_name("dtprfb_rowfwd", tests, NULL, NULL);
    return 0;
}
