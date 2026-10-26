/**
 * @file test_zlarft.c
 * @brief ZLARFT and ZLARFT_LVL2 tests for every implemented DIRECT/STOREV pair.
 *
 * k elementary reflectors H(i) = I - tau[i] * u_i * u_i**H of order n are
 * generated with ZLARFG and stored in V the way each pair expects (u_i in
 * column i for STOREV = 'C', u_i**H in row i otherwise). The unit and zero
 * entries of V, which must not be referenced, hold a guard value. The
 * triangular factor T from ZLARFT_LVL2, and from ZLARFT with the default
 * crossover, NX = 2, NX = 5 and lower-case flags, is checked against the
 * explicit product H of the reflectors: ||H - (I - U*S*U**H)||_1 / (n*eps),
 * where the columns of U are the u_i and S = T, or S = T**H for the pairs
 * that return the conjugate-transposed T (DIRECT = 'F', STOREV = 'T' and
 * DIRECT = 'T', STOREV = 'R'). The entries of the T array outside its
 * triangle must keep their guard value.
 * test_flags checks the XERBLA calls for the invalid DIRECT/STOREV pairs.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0
#define GUARD 1.0e3
#define NMAX 96
#define NSIZES 10
#define NVARIANTS 5

static const INT NVAL[NSIZES] = {1, 4, 4, 7, 9, 12, 70, 70, 90, 96};
static const INT KVAL[NSIZES] = {1, 1, 4, 3, 9, 5, 63, 64, 65, 96};

typedef struct {
    const char* name;
    const char* direct;
    const char* storev;
    const char* direct_lc;
    const char* storev_lc;
    INT backward;   /* H = H(k-1) ... H(1) H(0) */
    INT rowwise;    /* u_i**H is stored in row i of V */
    INT trans;      /* the conjugate-transposed T is returned */
    INT lower;      /* T is lower triangular */
} larft_mode;

static const larft_mode MODES[6] = {
    {"QR",  "F", "C", "f", "c", 0, 0, 0, 0},
    {"LQ",  "F", "R", "f", "r", 0, 1, 0, 0},
    {"LQT", "F", "T", "f", "t", 0, 1, 1, 1},
    {"QL",  "B", "C", "b", "c", 1, 0, 0, 1},
    {"RQ",  "B", "R", "b", "r", 1, 1, 0, 1},
    {"RQT", "T", "R", "t", "r", 1, 1, 1, 0},
};

static INT in_triangle(const larft_mode* md, INT r, INT c, INT k)
{
    if (r >= k) return 0;
    return md->lower ? (r >= c) : (r <= c);
}

static void check_mode(const larft_mode* md, uint64_t seed)
{
    const f64 eps = dlamch("P");
    const c128 GUARDC = CMPLX(GUARD, -GUARD);
    uint64_t rng[4];
    c128* U = malloc(NMAX * NMAX * sizeof(c128));
    c128* V = malloc((NMAX + 2) * NMAX * sizeof(c128));
    c128* T = malloc((NMAX + 1) * NMAX * sizeof(c128));
    c128* H = malloc(NMAX * NMAX * sizeof(c128));
    c128* M = malloc(NMAX * NMAX * sizeof(c128));
    c128 tau[NMAX], x[NMAX], w[NMAX];

    rng_seed(rng, seed);
    for (INT is = 0; is < NSIZES; is++) {
        const INT n = NVAL[is], k = KVAL[is];
        const INT ldv = md->rowwise ? k + 1 : n + 2;
        const INT ldt = k + 1;

        for (INT i = 0; i < k; i++) {
            const INT m = md->backward ? n - k + i + 1 : n - i;
            const INT unit = md->backward ? n - k + i : i;
            c128 alpha = CMPLX(rng_uniform_symmetric(rng),
                               rng_uniform_symmetric(rng));
            for (INT j = 0; j < m - 1; j++) {
                x[j] = CMPLX(rng_uniform_symmetric(rng),
                             rng_uniform_symmetric(rng));
            }
            zlarfg(m, &alpha, x, 1, &tau[i]);
            for (INT r = 0; r < n; r++) {
                U[r + i * n] = 0.0;
            }
            U[unit + i * n] = 1.0;
            for (INT j = 0; j < m - 1; j++) {
                U[(md->backward ? j : unit + 1 + j) + i * n] = x[j];
            }
        }

        for (INT q = 0; q < (NMAX + 2) * NMAX; q++) {
            V[q] = GUARDC;
        }
        for (INT i = 0; i < k; i++) {
            for (INT r = 0; r < n; r++) {
                const INT stored = md->backward ? (r < n - k + i) : (r > i);
                if (!stored) continue;
                if (md->rowwise) {
                    V[i + r * ldv] = conj(U[r + i * n]);
                } else {
                    V[r + i * ldv] = U[r + i * n];
                }
            }
        }

        for (INT c = 0; c < n; c++) {
            for (INT r = 0; r < n; r++) {
                H[r + c * n] = (r == c) ? 1.0 : 0.0;
            }
        }
        for (INT p = 0; p < k; p++) {
            const INT i = md->backward ? k - 1 - p : p;
            for (INT r = 0; r < n; r++) {
                c128 s = 0.0;
                for (INT c = 0; c < n; c++) {
                    s += H[r + c * n] * U[c + i * n];
                }
                w[r] = s;
            }
            for (INT c = 0; c < n; c++) {
                for (INT r = 0; r < n; r++) {
                    H[r + c * n] -= tau[i] * w[r] * conj(U[c + i * n]);
                }
            }
        }

        for (INT iv = 0; iv < NVARIANTS; iv++) {
            for (INT q = 0; q < ldt * k; q++) {
                T[q] = -GUARDC;
            }
            if (iv == 2) {
                xlaenv(3, 2);
            } else if (iv == 3) {
                xlaenv(3, 5);
            }
            strcpy(xerbla_srnamt, (iv == 0) ? "ZLARFT_LVL2" : "ZLARFT");
            xerbla_infot = 0;
            xerbla_lerr = 0;
            xerbla_ok = 1;
            if (iv == 0) {
                zlarft_lvl2(md->direct, md->storev, n, k, V, ldv, tau, T, ldt);
            } else if (iv == 4) {
                zlarft(md->direct_lc, md->storev_lc, n, k, V, ldv, tau, T, ldt);
            } else {
                zlarft(md->direct, md->storev, n, k, V, ldv, tau, T, ldt);
            }
            const INT lerr = xerbla_lerr;
            xerbla_srnamt[0] = '\0';
            xerbla_lerr = 0;
            xerbla_ok = 1;
            xlaenv_reset();
            assert_false(lerr);

            INT touched = 0;
            for (INT c = 0; c < k; c++) {
                for (INT r = 0; r < ldt; r++) {
                    if (!in_triangle(md, r, c, k) && T[r + c * ldt] != -GUARDC) {
                        touched++;
                    }
                }
            }

            /* M = U * S */
            for (INT b = 0; b < k; b++) {
                for (INT r = 0; r < n; r++) {
                    c128 s = 0.0;
                    for (INT a = 0; a < k; a++) {
                        c128 sab = 0.0;
                        if (md->trans) {
                            if (in_triangle(md, b, a, k)) sab = conj(T[b + a * ldt]);
                        } else {
                            if (in_triangle(md, a, b, k)) sab = T[a + b * ldt];
                        }
                        s += U[r + a * n] * sab;
                    }
                    M[r + b * n] = s;
                }
            }

            /* ||H - I + M * U**H||_1 */
            f64 err = 0.0;
            for (INT c = 0; c < n; c++) {
                f64 colsum = 0.0;
                for (INT r = 0; r < n; r++) {
                    c128 e = H[r + c * n] - ((r == c) ? 1.0 : 0.0);
                    for (INT b = 0; b < k; b++) {
                        e += M[r + b * n] * conj(U[c + b * n]);
                    }
                    colsum += cabs(e);
                }
                if (!(colsum <= err)) err = colsum;
            }
            const f64 resid = err / ((f64)n * eps);

            char ctx[64];
            snprintf(ctx, sizeof(ctx), "%s n=%d k=%d variant=%d", md->name,
                     (int)n, (int)k, (int)iv);
            set_test_context(ctx);
            if (touched != 0) {
                print_message("%s: %d entries outside the triangle changed\n",
                              ctx, (int)touched);
            }
            assert_int_equal(touched, 0);
            assert_residual_below(resid, THRESH);
        }
    }
    clear_test_context();

    free(U);
    free(V);
    free(T);
    free(H);
    free(M);
}

static void test_qr(void** state)
{
    (void)state;
    check_mode(&MODES[0], 0x5EC1A4F7B2D39011ULL);
}

static void test_lq(void** state)
{
    (void)state;
    check_mode(&MODES[1], 0x9D0C2E8A41F7B363ULL);
}

static void test_lqt(void** state)
{
    (void)state;
    check_mode(&MODES[2], 0x37B9E1D05C6A2F84ULL);
}

static void test_ql(void** state)
{
    (void)state;
    check_mode(&MODES[3], 0xC4A8F2163E0B7D59ULL);
}

static void test_rq(void** state)
{
    (void)state;
    check_mode(&MODES[4], 0x6F1D3B9A0E2C8475ULL);
}

static void test_rqt(void** state)
{
    (void)state;
    check_mode(&MODES[5], 0xA2E7406C9B1F5D38ULL);
}

static void test_flags(void** state)
{
    (void)state;
    static const struct {
        const char* direct;
        const char* storev;
        INT infot;
    } BAD[5] = {
        {"X", "C", 1},
        {"F", "X", 2},
        {"B", "T", 2},
        {"T", "C", 1},
        {"T", "T", 1},
    };
    const c128 GUARDC = CMPLX(GUARD, -GUARD);
    c128 V[9], tau[2] = {CMPLX(0.5, 0.25), CMPLX(0.5, -0.25)}, T[4];

    for (INT q = 0; q < 9; q++) {
        V[q] = 1.0;
    }
    for (INT ir = 0; ir < 2; ir++) {
        for (INT ib = 0; ib < 5; ib++) {
            for (INT n = 0; n <= 3; n += 3) {
                const INT k = (n == 0) ? 0 : 2;
                for (INT q = 0; q < 4; q++) {
                    T[q] = GUARDC;
                }
                strcpy(xerbla_srnamt, (ir == 0) ? "ZLARFT" : "ZLARFT_LVL2");
                xerbla_infot = BAD[ib].infot;
                xerbla_lerr = 0;
                xerbla_ok = 1;
                if (ir == 0) {
                    zlarft(BAD[ib].direct, BAD[ib].storev, n, k, V, 3, tau, T, 2);
                } else {
                    zlarft_lvl2(BAD[ib].direct, BAD[ib].storev, n, k, V, 3, tau,
                                T, 2);
                }
                const INT lerr = xerbla_lerr;
                const INT ok = xerbla_ok;
                xerbla_srnamt[0] = '\0';
                xerbla_lerr = 0;
                xerbla_ok = 1;
                assert_true(lerr);
                assert_true(ok);
                for (INT q = 0; q < 4; q++) {
                    assert_true(T[q] == GUARDC);
                }
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_qr),
        cmocka_unit_test(test_lq),
        cmocka_unit_test(test_lqt),
        cmocka_unit_test(test_ql),
        cmocka_unit_test(test_rq),
        cmocka_unit_test(test_rqt),
        cmocka_unit_test(test_flags),
    };
    (void)cmocka_run_group_tests_name("zlarft", tests, NULL, NULL);
    return 0;
}
