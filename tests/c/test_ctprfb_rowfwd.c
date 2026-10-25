/**
 * @file test_ctprfb_rowfwd.c
 * @brief CTPRFB and CTPMLQT regression tests for STOREV = 'R', DIRECT = 'F',
 *        SIDE = 'L'.
 *
 * test_tprfb_ldwork calls CTPRFB with LDB > LDWORK and compares the result
 * with a dense application of H = I - W**H T W, W = [ I V ].
 * test_tpmlqt_left applies the Q of CTPLQT from the left with CTPMLQT while
 * the strictly upper triangle of the trapezoidal block of B, which CTPLQT
 * does not reference, holds nonzero entries, and compares with Q formed by
 * CGEMLQT as in CLQT05.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0f

static void test_tprfb_ldwork(void** state)
{
    (void)state;
    const INT m = 6, n = 5, k = 4, l = 3;
    const INT ldv = k, ldt = k, lda = k, ldb = m + 7, ldwork = k;
    const INT nguard = 64;
    const c64 guard = CMPLXF(-7.0f, 0.0f);
    const char* transs[2] = {"N", "C"};
    const f32 eps = slamch("P");
    uint64_t rng_state[4];

    c64* V = malloc(ldv * m * sizeof(c64));
    c64* T = malloc(ldt * k * sizeof(c64));
    c64* A = malloc(lda * n * sizeof(c64));
    c64* B = malloc(ldb * n * sizeof(c64));
    c64* Vd = malloc(k * m * sizeof(c64));
    c64* A0 = malloc(k * n * sizeof(c64));
    c64* B0 = malloc(m * n * sizeof(c64));
    c64* Y = malloc(k * n * sizeof(c64));
    c64* Z = malloc(k * n * sizeof(c64));
    c64* work = malloc((ldwork * n + nguard) * sizeof(c64));
    assert_non_null(V); assert_non_null(T); assert_non_null(A);
    assert_non_null(B); assert_non_null(Vd); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Y); assert_non_null(Z);
    assert_non_null(work);

    rng_seed(rng_state, 0x63B2670BULL);
    for (INT it = 0; it < 2; it++) {
        clarnv_rng(2, ldv * m, V, rng_state);
        clarnv_rng(2, ldt * k, T, rng_state);
        clarnv_rng(2, lda * n, A, rng_state);
        for (INT j = 0; j < n; j++) {
            clarnv_rng(2, m, &B[j * ldb], rng_state);
            for (INT i = m; i < ldb; i++) {
                B[i + j * ldb] = guard;
            }
        }
        for (INT i = 0; i < ldwork * n + nguard; i++) {
            work[i] = guard;
        }
        clacpy("F", k, m, V, ldv, Vd, k);
        for (INT j = 1; j < l; j++) {
            for (INT i = 0; i < j; i++) {
                Vd[i + (m - l + j) * k] = CMPLXF(0.0f, 0.0f);
            }
        }
        clacpy("F", k, n, A, lda, A0, k);
        clacpy("F", m, n, B, ldb, B0, m);

        ctprfb("L", transs[it], "F", "R", m, n, k, l, V, ldv, T, ldt,
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
                c64 s = A0[i + j * k];
                for (INT p = 0; p < m; p++) {
                    s += Vd[i + p * k] * B0[p + j * m];
                }
                Y[i + j * k] = s;
            }
        }
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < k; i++) {
                c64 s = CMPLXF(0.0f, 0.0f);
                if (it == 0) {
                    for (INT p = i; p < k; p++) {
                        s += T[i + p * ldt] * Y[p + j * k];
                    }
                } else {
                    for (INT p = 0; p <= i; p++) {
                        s += conjf(T[p + i * ldt]) * Y[p + j * k];
                    }
                }
                Z[i + j * k] = s;
            }
        }

        f32 diff = 0.0f, cmax = 1.0f;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < k; i++) {
                c64 r = A0[i + j * k] - Z[i + j * k];
                f32 d = cabsf(A[i + j * lda] - r);
                if (!(d <= diff)) diff = d;
                if (cabsf(r) > cmax) cmax = cabsf(r);
                if (cabsf(Z[i + j * k]) > cmax) cmax = cabsf(Z[i + j * k]);
            }
            for (INT p = 0; p < m; p++) {
                c64 r = B0[p + j * m];
                for (INT i = 0; i < k; i++) {
                    r -= conjf(Vd[i + p * k]) * Z[i + j * k];
                }
                f32 d = cabsf(B[p + j * ldb] - r);
                if (!(d <= diff)) diff = d;
                if (cabsf(r) > cmax) cmax = cabsf(r);
            }
        }
        f32 resid = diff / (eps * (m + k) * cmax);
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
    const f32 eps = slamch("E");
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);
    const char* transs[2] = {"N", "C"};
    INT info;
    uint64_t rng_state[4];

    c64* AF = malloc(m * n2 * sizeof(c64));
    c64* AFZ = malloc(m * n2 * sizeof(c64));
    c64* Q = malloc(n2 * n2 * sizeof(c64));
    c64* T = malloc(nb * m * sizeof(c64));
    c64* C = malloc(n2 * m * sizeof(c64));
    c64* CF = malloc(n2 * m * sizeof(c64));
    f32* rwork = malloc(n2 * sizeof(f32));
    c64* work = malloc(lwork * sizeof(c64));
    assert_non_null(AF); assert_non_null(AFZ); assert_non_null(Q);
    assert_non_null(T); assert_non_null(C); assert_non_null(CF);
    assert_non_null(rwork); assert_non_null(work);

    rng_seed(rng_state, 0x1376ULL);
    claset("F", m, n2, CZERO, CZERO, AF, m);
    claset("F", nb, m, CZERO, CZERO, T, nb);
    for (INT j = 0; j < m; j++) {
        clarnv_rng(2, m - j, &AF[j + j * m], rng_state);
    }
    for (INT j = 0; j < n - l; j++) {
        clarnv_rng(2, m, &AF[(m + j) * m], rng_state);
    }
    for (INT j = 0; j < l; j++) {
        clarnv_rng(2, m - j, &AF[j + (m + n - l + j) * m], rng_state);
    }
    for (INT j = 1; j < l; j++) {
        clarnv_rng(2, j, &AF[(m + n - l + j) * m], rng_state);
    }

    ctplqt(m, n, l, nb, AF, m, &AF[m * m], m, T, ldt, work, &info);
    assert_info_success(info);

    clacpy("F", m, n2, AF, m, AFZ, m);
    for (INT j = 1; j < l; j++) {
        for (INT i = 0; i < j; i++) {
            AFZ[i + (m + n - l + j) * m] = CZERO;
        }
    }
    claset("F", n2, n2, CZERO, CONE, Q, n2);
    cgemlqt("L", "N", n2, n2, k, nb, AFZ, m, T, ldt, Q, n2, work, &info);
    assert_info_success(info);

    for (INT j = 0; j < m; j++) {
        clarnv_rng(2, n2, &C[j * n2], rng_state);
    }
    f32 cnorm = clange("1", n2, m, C, n2, rwork);

    for (INT it = 0; it < 2; it++) {
        clacpy("F", n2, m, C, n2, CF, n2);
        ctpmlqt("L", transs[it], n, m, k, l, nb, &AF[m * m], m, T, ldt,
                CF, n2, &CF[m], n2, work, &info);
        assert_info_success(info);
        cblas_cgemm(CblasColMajor, (it == 0) ? CblasNoTrans : CblasConjTrans,
                    CblasNoTrans, n2, m, n2, &CNEGONE, Q, n2, C, n2, &CONE,
                    CF, n2);
        f32 resid = clange("1", n2, m, CF, n2, rwork) / (eps * n2 * cnorm);
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
    (void)cmocka_run_group_tests_name("ztprfb_rowfwd", tests, NULL, NULL);
    return 0;
}
