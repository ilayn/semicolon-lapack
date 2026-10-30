/**
 * @file test_dbdsvdx_split.c
 * @brief DBDSVDX regression test for bidiagonal matrices that split in D.
 *
 * A zero diagonal entry in the middle of an active submatrix splits the
 * TGK matrix, and the next TGK block starts at the off-diagonal entry right
 * after the zero, ISPLT = IDPTR + 1. Each case builds a bidiagonal matrix
 * with zero diagonal entries in the middle and at the ends, with and without
 * zero off-diagonal entries, and calls DBDSVDX for both UPLO, RANGE = 'A',
 * 'V' and 'I', and JOBZ = 'N' and 'V'. The singular values are compared
 * with DBDSQR, and the singular vectors are checked with DBDT04 and DORT01.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0
#define N 8

typedef struct {
    INT dzero[N];
    INT ezero[N];
} pattern_t;

static const pattern_t patterns[] = {
    {{0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 1, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}},
    {{1, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 0}},
    {{0, 1, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 1, 0, 0, 0, 0}},
    {{0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0}},
    {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 1, 0, 0}},
};

static void run_case(const pattern_t* pat, const uint64_t seed,
                     const char* uplo, const char* range)
{
    const INT n = N;
    const INT ldz = 2 * n;
    const f64 eps = dlamch("P");
    f64 D0[N], E0[N], D[N], E[N], Dr[N], Er[N], S[N], S2[N];
    f64 vl = 0.0, vu = 0.0;
    INT il = 0, iu = 0, ns, ns2, info, nexp, first;
    uint64_t rng_state[4];

    f64* Z = calloc((size_t)ldz * (n + 1), sizeof(f64));
    f64* U = calloc((size_t)n * n, sizeof(f64));
    f64* VT = calloc((size_t)n * n, sizeof(f64));
    f64* work = calloc(14 * n, sizeof(f64));
    f64* rwork = calloc(4 * n * n, sizeof(f64));
    INT* iwork = calloc(12 * n, sizeof(INT));
    assert_non_null(Z); assert_non_null(U); assert_non_null(VT);
    assert_non_null(work); assert_non_null(rwork); assert_non_null(iwork);

    rng_seed(rng_state, seed);
    for (INT i = 0; i < n; i++) {
        f64 r = rng_uniform_symmetric(rng_state);
        D0[i] = (r < 0.0) ? r - 0.5 : r + 0.5;
        r = rng_uniform_symmetric(rng_state);
        E0[i] = (r < 0.0) ? r - 0.5 : r + 0.5;
        if (pat->dzero[i]) D0[i] = 0.0;
        if (pat->ezero[i]) E0[i] = 0.0;
    }
    E0[n - 1] = 0.0;

    memcpy(Dr, D0, sizeof(Dr));
    memcpy(Er, E0, sizeof(Er));
    dbdsqr(uplo, n, 0, 0, 0, Dr, Er, NULL, 1, NULL, 1, NULL, 1, rwork, &info);
    assert_info_success(info);

    if (range[0] == 'A') {
        first = 0;
        nexp = n;
    } else if (range[0] == 'I') {
        il = 1;
        iu = n - 3;
        first = il;
        nexp = iu - il + 1;
    } else {
        first = 1;
        nexp = n - 3;
        vl = 0.5 * (Dr[first + nexp - 1] + Dr[first + nexp]);
        vu = 0.5 * (Dr[first - 1] + Dr[first]);
    }

    for (INT wz = 0; wz < 2; wz++) {
        const char* jobz = wz ? "V" : "N";
        f64* Sout = wz ? S : S2;
        INT* nsout = wz ? &ns : &ns2;

        memcpy(D, D0, sizeof(D));
        memcpy(E, E0, sizeof(E));
        memset(Z, 0, (size_t)ldz * (n + 1) * sizeof(f64));
        dbdsvdx(uplo, jobz, range, n, D, E, vl, vu, il, iu, nsout, Sout,
                Z, ldz, work, iwork, &info);
        assert_info_success(info);
        assert_int_equal(*nsout, nexp);

        f64 err = 0.0;
        for (INT i = 0; i < nexp; i++) {
            f64 d = fabs(Sout[i] - Dr[first + i]);
            if (d > err) err = d;
        }
        err = err / ((f64)n * eps * Dr[0]);
        assert_residual_ok(err);

        if (wz) {
            f64 resid;
            for (INT j = 0; j < ns; j++) {
                for (INT i = 0; i < n; i++) {
                    U[i + j * n] = Z[i + j * ldz];
                    VT[j + i * n] = Z[n + i + j * ldz];
                }
            }
            dbdt04(uplo, n, D0, E0, S, ns, U, n, VT, n, rwork, &resid);
            assert_residual_ok(resid);
            dort01("Columns", n, ns, U, n, rwork, 4 * n * n, &resid);
            assert_residual_ok(resid);
            dort01("Rows", ns, n, VT, n, rwork, 4 * n * n, &resid);
            assert_residual_ok(resid);
        }
    }

    free(Z); free(U); free(VT); free(work); free(rwork); free(iwork);
}

static void run_range(const char* range)
{
    const char* uplos[2] = {"U", "L"};
    const INT npat = (INT)(sizeof(patterns) / sizeof(patterns[0]));
    for (INT p = 0; p < npat; p++) {
        for (INT u = 0; u < 2; u++) {
            run_case(&patterns[p], 0xB5D0ULL + (uint64_t)p, uplos[u], range);
        }
    }
}

static void test_range_a(void** state)
{
    (void)state;
    run_range("A");
}

static void test_range_v(void** state)
{
    (void)state;
    run_range("V");
}

static void test_range_i(void** state)
{
    (void)state;
    run_range("I");
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_range_a),
        cmocka_unit_test(test_range_v),
        cmocka_unit_test(test_range_i),
    };
    (void)cmocka_run_group_tests_name("dbdsvdx_split", tests, NULL, NULL);
    return 0;
}
