/**
 * @file test_dbdsdc_compact.c
 * @brief DBDSDC regression test for the compact form (COMPQ = 'P').
 *
 * With COMPQ = 'P', DBDSDC stores the compact singular vectors of DLASDA in
 * Q and IQ. IQ holds, column by column, the sorting permutation and the
 * UPLO indicator, then K, GIVPTR, PERM and GIVCOL of DLASDA (the IQ
 * offsets START + K*N, START + GIVPTR*N, ... with K = 1, GIVPTR = 2,
 * PERM = 3, GIVCOL = PERM + MLVL in the reference). Each case builds a
 * well conditioned bidiagonal matrix B that does not split, calls DBDSDC
 * with COMPQ = 'P', undoes the recorded sort to recover the singular values
 * in DLASDA order, and applies the compact factors with DLALSA to form
 * X = V * inv(S) * U**T * B, which must be the identity. For UPLO = 'L' the
 * rotations stored in Q are applied to B first. The last IQ entry must be
 * the UPLO indicator and the sorting permutation must be in range.
 * The random matrices never deflate by Givens rotations. The constant
 * matrices (D = 2.5, E = 0.5) do, because their subproblems of equal size
 * have equal singular values, so they also check GIVPTR, GIVCOL and GIVNUM;
 * at least one node must have GIVPTR > 0.
 */

#include "test_harness.h"
#include "test_rng.h"
#include "lapack_tuning.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0

static void run_case(const INT n, const char* uplo, const INT constant,
                     const uint64_t seed)
{
    const INT smlsiz = LAPACK_SMLSIZ;
    const INT mlvl = (INT)(log((f64)n / (f64)(smlsiz + 1)) / log(2.0)) + 1;
    const INT qstart = (uplo[0] == 'U') ? 2 : 4;
    const INT iu = 0;
    const INT ivt = iu + smlsiz;
    const INT difl = ivt + smlsiz + 1;
    const INT difr = difl + mlvl;
    const INT z = difr + 2 * mlvl;
    const INT ic = z + mlvl;
    const INT is = ic + 1;
    const INT poles = is + 1;
    const INT givnum = poles + 2 * mlvl;
    const INT k = 1;
    const INT givptr = 2;
    const INT perm = 3;
    const INT givcol = perm + mlvl;
    const f64 eps = dlamch("P");
    INT info;
    uint64_t rng_state[4];

    f64* D0 = malloc((size_t)n * sizeof(f64));
    f64* E0 = malloc((size_t)n * sizeof(f64));
    f64* D = malloc((size_t)n * sizeof(f64));
    f64* E = malloc((size_t)n * sizeof(f64));
    f64* S = malloc((size_t)n * sizeof(f64));
    f64* B = calloc((size_t)n * n, sizeof(f64));
    f64* BX = calloc((size_t)n * n, sizeof(f64));
    f64* X = calloc((size_t)n * n, sizeof(f64));
    f64* Q = calloc((size_t)n * (qstart + givnum + 2 * mlvl), sizeof(f64));
    INT* IQ = calloc((size_t)n * (givcol + 2 * mlvl), sizeof(INT));
    f64* work = calloc((size_t)6 * n, sizeof(f64));
    INT* iwork = calloc((size_t)8 * n, sizeof(INT));
    f64 udum = 0.0, vtdum = 0.0;
    assert_non_null(D0); assert_non_null(E0); assert_non_null(D);
    assert_non_null(E); assert_non_null(S); assert_non_null(B);
    assert_non_null(BX); assert_non_null(X); assert_non_null(Q);
    assert_non_null(IQ); assert_non_null(work); assert_non_null(iwork);

    rng_seed(rng_state, seed);
    for (INT i = 0; i < n; i++) {
        if (constant) {
            D0[i] = 2.5;
            E0[i] = 0.5;
        } else {
            D0[i] = 2.5 + 0.5 * rng_uniform_symmetric(rng_state);
            E0[i] = 0.5 + 0.4 * rng_uniform_symmetric(rng_state);
        }
    }
    E0[n - 1] = 0.0;
    memcpy(D, D0, (size_t)n * sizeof(f64));
    memcpy(E, E0, (size_t)n * sizeof(f64));

    dbdsdc(uplo, "P", n, D, E, &udum, 1, &vtdum, 1, Q, IQ, work, iwork, &info);
    assert_info_success(info);

    assert_int_equal(IQ[n - 1], (uplo[0] == 'U') ? 1 : 0);
    for (INT i = 0; i < n - 1; i++) {
        assert_true(IQ[i] >= i && IQ[i] < n);
    }
    if (constant) {
        const INT nd = (1 << mlvl) - 1;
        INT ngiv = 0;
        for (INT i = 0; i < nd; i++) {
            ngiv += IQ[i + givptr * n];
        }
        assert_true(ngiv > 0);
    }

    memcpy(S, D, (size_t)n * sizeof(f64));
    for (INT i = n - 2; i >= 0; i--) {
        f64 p = S[i];
        S[i] = S[IQ[i]];
        S[IQ[i]] = p;
    }

    for (INT i = 0; i < n; i++) {
        B[i + i * n] = D0[i];
        if (i < n - 1) {
            if (uplo[0] == 'U') {
                B[i + (i + 1) * n] = E0[i];
            } else {
                B[(i + 1) + i * n] = E0[i];
            }
        }
    }
    if (uplo[0] == 'L') {
        dlasr("L", "V", "F", n, n, &Q[2 * n], &Q[3 * n], B, n);
    }

    dlalsa(0, smlsiz, n, n, B, n, BX, n, &Q[(iu + qstart) * n], n,
           &Q[(ivt + qstart) * n], &IQ[k * n], &Q[(difl + qstart) * n],
           &Q[(difr + qstart) * n], &Q[(z + qstart) * n],
           &Q[(poles + qstart) * n], &IQ[givptr * n], &IQ[givcol * n], n,
           &IQ[perm * n], &Q[(givnum + qstart) * n], &Q[(ic + qstart) * n],
           &Q[(is + qstart) * n], work, iwork, &info);
    assert_info_success(info);

    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            BX[i + j * n] /= S[i];
        }
    }

    dlalsa(1, smlsiz, n, n, BX, n, X, n, &Q[(iu + qstart) * n], n,
           &Q[(ivt + qstart) * n], &IQ[k * n], &Q[(difl + qstart) * n],
           &Q[(difr + qstart) * n], &Q[(z + qstart) * n],
           &Q[(poles + qstart) * n], &IQ[givptr * n], &IQ[givcol * n], n,
           &IQ[perm * n], &Q[(givnum + qstart) * n], &Q[(ic + qstart) * n],
           &Q[(is + qstart) * n], work, iwork, &info);
    assert_info_success(info);

    for (INT i = 0; i < n; i++) {
        X[i + i * n] -= 1.0;
    }
    f64 resid = dlange("1", n, n, X, n, work) / ((f64)n * eps);
    assert_residual_below(resid, THRESH);

    free(D0); free(E0); free(D); free(E); free(S); free(B); free(BX); free(X);
    free(Q); free(IQ); free(work); free(iwork);
}

static void run_uplo(const char* uplo)
{
    static const INT sizes[] = {26, 40, 52, 53, 100, 211};
    static const INT csizes[] = {207, 211};
    const INT nsizes = (INT)(sizeof(sizes) / sizeof(sizes[0]));
    const INT ncsizes = (INT)(sizeof(csizes) / sizeof(csizes[0]));
    for (INT i = 0; i < nsizes; i++) {
        run_case(sizes[i], uplo, 0, 0xBD5DC0ULL + (uint64_t)i);
    }
    for (INT i = 0; i < ncsizes; i++) {
        run_case(csizes[i], uplo, 1, 0);
    }
}

static void test_upper(void** state)
{
    (void)state;
    run_uplo("U");
}

static void test_lower(void** state)
{
    (void)state;
    run_uplo("L");
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_upper),
        cmocka_unit_test(test_lower),
    };
    (void)cmocka_run_group_tests_name("dbdsdc_compact", tests, NULL, NULL);
    return 0;
}
