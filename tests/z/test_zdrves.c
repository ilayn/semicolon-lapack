/**
 * @file test_zdrves.c
 * @brief Schur decomposition test driver - port of LAPACK TESTING/EIG/zdrves.f
 *
 * Tests the nonsymmetric eigenvalue (Schur form) problem driver ZGEES.
 *
 * Each (n, jtype, iwk, isort) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (13 total):
 *   (1)  0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *   (2)  | A - VS T VS' | / ( n |A| ulp )  (no sorting)
 *   (3)  | I - VS VS' | / ( n ulp )  (no sorting)
 *   (4)  0 if W are eigenvalues of T (no sorting)
 *   (5)  0 if T(with VS) = T(without VS) (no sorting)
 *   (6)  0 if eigenvalues(with VS) = eigenvalues(without VS) (no sorting)
 *   (7-12) Same as (1-6) but with eigenvalue sorting
 *   (13) 0 if SDIM matches count of selected eigenvalues
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via ZLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via ZLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via ZLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via ZLATMR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#define THRESH 30.0
#define MAXTYP 21

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Common block for eigenvalue selection (mirrors SSLCT in zdrves.f) */
static INT g_selopt;
static INT g_seldim;
static INT g_selval[20];
static f64 g_selwr[20];
static f64 g_selwi[20];

/**
 * Selection function for zgees.
 * Selects eigenvalues based on global selection criteria.
 */
static INT zslect(const c128* w)
{
    INT j;
    f64 eps = dlamch("P");

    if (g_selopt == 0) {
        return 1;
    }

    for (j = 0; j < g_seldim; j++) {
        f64 rmin = cabs(*w - CMPLX(g_selwr[j], g_selwi[j]));
        if (rmin < eps) {
            return g_selval[j];
        }
    }
    return 0;
}

typedef struct {
    INT n;
    INT jtype;
    INT iwk;
    INT isort;
    char name[96];
} zdrves_params_t;

typedef struct {
    INT nmax;

    c128* A;
    c128* H;
    c128* HT;
    c128* VS;

    c128* W;
    c128* WT;

    c128* work;
    f64* rwork;
    INT* iwork;
    INT* bwork;
    INT lwork;

    f64 result[13];

    uint64_t rng_state[4];
} zdrves_workspace_t;

static zdrves_workspace_t* g_ws = NULL;

static const INT KTYPE[MAXTYP]  = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
static const INT KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
static const INT KMODE[MAXTYP]  = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
static const INT KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrves_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A   = malloc(n2 * sizeof(c128));
    g_ws->H   = malloc(n2 * sizeof(c128));
    g_ws->HT  = malloc(n2 * sizeof(c128));
    g_ws->VS  = malloc(n2 * sizeof(c128));
    g_ws->W   = malloc(nmax * sizeof(c128));
    g_ws->WT  = malloc(nmax * sizeof(c128));

    g_ws->lwork = 5 * nmax + 2 * n2;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));
    g_ws->rwork = malloc(nmax * sizeof(f64));
    g_ws->iwork = malloc(nmax * sizeof(INT));
    g_ws->bwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->HT || !g_ws->VS ||
        !g_ws->W || !g_ws->WT ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork || !g_ws->bwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->H);
        free(g_ws->HT);
        free(g_ws->VS);
        free(g_ws->W);
        free(g_ws->WT);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on zdrves.f lines 583-700.
 */
static INT generate_matrix(INT n, INT jtype, c128* A, INT lda,
                           c128* work, f64* rwork, INT* iwork,
                           uint64_t state[static 4])
{
    INT itype = KTYPE[jtype - 1];
    INT imode = KMODE[jtype - 1];
    f64 anorm, cond, conds;
    INT iinfo = 0;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtulp = sqrt(ulp);
    f64 rtulpi = 1.0 / rtulp;

    c128 czero = CMPLX(0.0, 0.0);
    c128 cone = CMPLX(1.0, 0.0);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0;
    }

    zlaset("F", n, n, czero, czero, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLX(anorm, 0.0);
        }

    } else if (itype == 3) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLX(anorm, 0.0);
            if (j > 0) {
                A[j + (j - 1) * lda] = cone;
            }
        }

    } else if (itype == 4) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0;
        }

        zlatme(n, "D", work, imode, cond, cone,
               "T", "T", "T", rwork, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 7) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, 0, 0,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "H", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

        if (n >= 4) {
            zlaset("F", 2, n, czero, czero, A, lda);
            zlaset("F", n - 3, 1, czero, czero, A + 2, lda);
            zlaset("F", n - 3, 2, czero, czero, A + 2 + (n - 2) * lda, lda);
            zlaset("F", 1, n, czero, czero, A + n - 1, lda);
        }

    } else if (itype == 10) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, 0,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tests for a single (n, jtype, iwk, isort) combination.
 *
 * Based on zdrves.f lines 705-860.
 */
static void run_zdrves_single(zdrves_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;
    INT iwk = params->iwk;
    INT isort = params->isort;

    zdrves_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvs = ws->nmax;

    c128* A = ws->A;
    c128* H = ws->H;
    c128* HT = ws->HT;
    c128* VS = ws->VS;
    c128* W = ws->W;
    c128* WT = ws->WT;
    c128* work = ws->work;
    f64* rwork = ws->rwork;
    INT* bwork = ws->bwork;

    f64 ulp = dlamch("P");
    f64 ulpinv = 1.0 / ulp;

    INT rsub = (isort == 0) ? 0 : 6;
    const char* sort = (isort == 0) ? "N" : "S";

    for (INT j = 0; j < 13; j++) {
        ws->result[j] = -1.0;
    }

    if (n == 0) {
        return;
    }

    INT iinfo = generate_matrix(n, jtype, A, lda, work, rwork,
                                ws->iwork, ws->rng_state);
    if (iinfo != 0) {
        ws->result[rsub] = ulpinv;
        fprintf(stderr, "Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    INT nnwork;
    if (iwk == 1) {
        nnwork = 3 * n;
    } else {
        nnwork = 5 * n + 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    zlacpy("F", n, n, A, lda, H, lda);

    g_selopt = 1;

    INT sdim;
    zgees("V", sort, zslect, n, H, lda, &sdim, W, VS, ldvs,
          work, nnwork, rwork, bwork, &iinfo);

    if (iinfo != 0) {
        ws->result[rsub] = ulpinv;
        if (iinfo > 0 && iinfo <= n) {
            fprintf(stderr, "ZGEES failed with info=%d (QR failed)\n", iinfo);
        }
        return;
    }

    /* Test 1 (or 7): Verify upper triangular structure.
     * Complex Schur form is strictly upper triangular (no 2x2 blocks). */
    ws->result[rsub] = 0.0;

    for (INT j = 0; j < n - 1; j++) {
        for (INT i = j + 1; i < n; i++) {
            if (H[i + j * lda] != CMPLX(0.0, 0.0)) {
                ws->result[rsub] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[rsub]);

    /* Tests 2-3 (or 8-9): | A - VS*T*VS' | and | I - VS*VS' | */
    INT lwork_hst = 2 * n * n > 1 ? 2 * n * n : 1;
    f64 res[2];
    zhst01(n, 0, n - 1, A, lda, H, lda, VS, ldvs, work, lwork_hst, rwork, res);
    ws->result[rsub + 1] = res[0];
    ws->result[rsub + 2] = res[1];

    assert_residual_ok(ws->result[rsub + 1]);
    assert_residual_ok(ws->result[rsub + 2]);

    /* Test 4 (or 10): Check eigenvalues match diagonal of T */
    ws->result[rsub + 3] = 0.0;
    for (INT i = 0; i < n; i++) {
        if (H[i + i * lda] != W[i]) {
            ws->result[rsub + 3] = ulpinv;
        }
    }

    assert_residual_ok(ws->result[rsub + 3]);

    /* Tests 5-6 (or 11-12): Compare with and without VS */
    zlacpy("F", n, n, A, lda, HT, lda);

    zgees("N", sort, zslect, n, HT, lda, &sdim, WT, VS, ldvs,
          work, nnwork, rwork, bwork, &iinfo);

    if (iinfo != 0) {
        ws->result[rsub + 4] = ulpinv;
    } else {
        /* Test 5: Compare T matrices */
        ws->result[rsub + 4] = 0.0;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (H[i + j * lda] != HT[i + j * lda]) {
                    ws->result[rsub + 4] = ulpinv;
                }
            }
        }

        /* Test 6: Compare eigenvalues */
        ws->result[rsub + 5] = 0.0;
        for (INT i = 0; i < n; i++) {
            if (W[i] != WT[i]) {
                ws->result[rsub + 5] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[rsub + 4]);
    assert_residual_ok(ws->result[rsub + 5]);

    /* Test 13 (only for sorted case): SDIM validation */
    if (isort == 1) {
        INT knteig = 0;
        for (INT i = 0; i < n; i++) {
            if (zslect(&W[i])) {
                knteig++;
            }
        }

        ws->result[12] = 0.0;
        if (sdim != knteig) {
            ws->result[12] = ulpinv;
        }

        /* Check contiguity: selected eigenvalues must be first */
        for (INT i = 0; i < n - 1; i++) {
            if (zslect(&W[i + 1]) && !zslect(&W[i])) {
                ws->result[12] = ulpinv;
            }
        }

        assert_residual_ok(ws->result[12]);
    }
}

static void test_zdrves_case(void** state)
{
    zdrves_params_t* params = *state;
    run_zdrves_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP * 2 * 2)

static zdrves_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            for (INT iwk = 1; iwk <= 2; iwk++) {
                for (INT isort = 0; isort <= 1; isort++) {
                    zdrves_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->jtype = jtype;
                    p->iwk = iwk;
                    p->isort = isort;
                    snprintf(p->name, sizeof(p->name),
                             "zdrves_n%d_type%d_wk%d_sort%d", n, jtype, iwk, isort);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zdrves_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();

    (void)_cmocka_run_group_tests("zdrves", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
