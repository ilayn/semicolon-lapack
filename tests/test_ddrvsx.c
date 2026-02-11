/**
 * @file test_ddrvsx.c
 * @brief Non-symmetric eigenvalue Schur form expert driver test - port of
 *        LAPACK TESTING/EIG/ddrvsx.f
 *
 * Tests the nonsymmetric eigenvalue problem expert driver DGEESX.
 *
 * Each (n, jtype, iwk) combination is registered as a separate CMocka test
 * for random matrices (tests 1-15). An additional 26 precomputed matrices
 * test condition number accuracy (tests 16-17).
 *
 * Test ratios (17 total):
 *   (1)  0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *   (2)  | A - VS T VS' | / ( n |A| ulp ) (no sorting)
 *   (3)  | I - VS VS' | / ( n ulp ) (no sorting)
 *   (4)  0 if WR+sqrt(-1)*WI are eigenvalues of T (no sorting)
 *   (5)  0 if T(with VS) = T(without VS) (no sorting)
 *   (6)  0 if eigenvalues(with VS) = eigenvalues(without VS) (no sorting)
 *   (7)  0 if T is in Schur form (with sorting)
 *   (8)  | A - VS T VS' | / ( n |A| ulp ) (with sorting)
 *   (9)  | I - VS VS' | / ( n ulp ) (with sorting)
 *  (10)  0 if WR+sqrt(-1)*WI are eigenvalues of T (with sorting)
 *  (11)  0 if T(with VS) = T(without VS) (with sorting)
 *  (12)  0 if eigenvalues(with VS) = eigenvalues(without VS) (with sorting)
 *  (13)  0 if sorting successful
 *  (14)  0 if RCONDE same no matter what else computed
 *  (15)  0 if RCONDV same no matter what else computed
 *  (16)  |RCONDE - RCONDE(precomputed)| / cond(RCONDE)
 *  (17)  |RCONDV - RCONDV(precomputed)| / cond(RCONDV)
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via DLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via DLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via DLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via DLATMR)
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include "testutils/dsx_testdata.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold from ded.in */
#define THRESH 20.0

/* Maximum matrix type to test */
#define MAXTYP 21

/* Test dimensions from ded.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* External function declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta, double* A, const int lda);

/* Test parameters for a single test case */
typedef struct {
    int n;
    int jtype;    /* Matrix type (1-21 for random, 22 for precomputed) */
    int iwk;      /* Workspace variant (1=minimal, 2=generous) */
    int precomp_idx; /* Index into DSX_PRECOMPUTED (-1 for random) */
    char name[96];
} ddrvsx_params_t;

/* Workspace structure for all tests */
typedef struct {
    int nmax;

    /* Matrices (all nmax x nmax) */
    double* A;      /* Original matrix */
    double* H;      /* Copy modified by DGEESX */
    double* HT;     /* Copy for comparison */
    double* VS;     /* Schur vectors */
    double* VS1;    /* Schur vectors (backup) */

    /* Eigenvalues */
    double* wr;     /* Real parts */
    double* wi;     /* Imaginary parts */
    double* wrt;    /* Real parts (temp) */
    double* wit;    /* Imaginary parts (temp) */
    double* wrtmp;  /* Real parts (comparison) */
    double* witmp;  /* Imaginary parts (comparison) */

    /* Work arrays */
    double* work;
    int* iwork;
    int* bwork;
    int lwork;

    /* Test results */
    double result[17];

    /* RNG state */
    uint64_t rng_state[4];
} ddrvsx_workspace_t;

/* Global workspace pointer */
static ddrvsx_workspace_t* g_ws = NULL;

/* Matrix type parameters (from ddrvsx.f DATA statements lines 519-524) */
static const int KTYPE[MAXTYP]  = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
static const int KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
static const int KMODE[MAXTYP]  = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
static const int KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};

/**
 * Group setup: allocate shared workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvsx_workspace_t));
    if (!g_ws) return -1;

    /* 12 is the largest dimension in precomputed input */
    g_ws->nmax = 12;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(double));
    g_ws->H   = malloc(n2 * sizeof(double));
    g_ws->HT  = malloc(n2 * sizeof(double));
    g_ws->VS  = malloc(n2 * sizeof(double));
    g_ws->VS1 = malloc(n2 * sizeof(double));
    g_ws->wr     = malloc(nmax * sizeof(double));
    g_ws->wi     = malloc(nmax * sizeof(double));
    g_ws->wrt    = malloc(nmax * sizeof(double));
    g_ws->wit    = malloc(nmax * sizeof(double));
    g_ws->wrtmp  = malloc(nmax * sizeof(double));
    g_ws->witmp  = malloc(nmax * sizeof(double));

    /* Workspace: max(3*N, 2*N^2) (ddrvsx.f line 569) */
    g_ws->lwork = 2 * n2;
    if (g_ws->lwork < 3 * nmax) g_ws->lwork = 3 * nmax;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work  = malloc(g_ws->lwork * sizeof(double));

    /* IWORK dimension: N*N (dget24.c line 144: liwork = n*n) */
    g_ws->iwork = malloc(n2 * sizeof(int));

    /* BWORK dimension: N */
    g_ws->bwork = malloc(nmax * sizeof(int));

    if (!g_ws->A || !g_ws->H || !g_ws->HT || !g_ws->VS || !g_ws->VS1 ||
        !g_ws->wr || !g_ws->wi || !g_ws->wrt || !g_ws->wit ||
        !g_ws->wrtmp || !g_ws->witmp ||
        !g_ws->work || !g_ws->iwork || !g_ws->bwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

/**
 * Group teardown: free shared workspace.
 */
static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->H);
        free(g_ws->HT);
        free(g_ws->VS);
        free(g_ws->VS1);
        free(g_ws->wr);
        free(g_ws->wi);
        free(g_ws->wrt);
        free(g_ws->wit);
        free(g_ws->wrtmp);
        free(g_ws->witmp);
        free(g_ws->work);
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
 * Based on ddrvsx.f lines 630-765.
 */
static int generate_matrix(int n, int jtype, double* A, int lda,
                           double* work, int* iwork, uint64_t state[static 4])
{
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    double anorm, cond, conds;
    int iinfo = 0;

    double ulp = dlamch("P");
    double unfl = dlamch("S");
    double ovfl = 1.0 / unfl;
    double ulpinv = 1.0 / ulp;
    double rtulp = sqrt(ulp);
    double rtulpi = 1.0 / rtulp;

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0;
    }

    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (int jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 3) {
        for (int jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
            if (jcol > 0) {
                A[jcol + (jcol - 1) * lda] = 1.0;
            }
        }

    } else if (itype == 4) {
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0;
        }

        char ei[2] = " ";
        dlatme(n, "S", work, imode, cond, 1.0,
               ei, "T", "T", "T", work + n, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 7) {
        int idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        int idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        int idumma[1] = {1};
        dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

        if (n >= 4) {
            dlaset("F", 2, n, 0.0, 0.0, A, lda);
            dlaset("F", n - 3, 1, 0.0, 0.0, A + 2, lda);
            dlaset("F", n - 3, 2, 0.0, 0.0, A + 2 + (n - 2) * lda, lda);
            dlaset("F", 1, n, 0.0, 0.0, A + n - 1, lda);
        }

    } else if (itype == 10) {
        int idumma[1] = {1};
        dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tests for a single random (n, jtype, iwk) combination.
 *
 * Based on ddrvsx.f lines 776-826.
 */
static void run_ddrvsx_random(ddrvsx_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;
    int iwk = params->iwk;

    ddrvsx_workspace_t* ws = g_ws;
    int lda = ws->nmax;
    int ldvs = ws->nmax;

    double* A = ws->A;
    double* H = ws->H;
    double* HT = ws->HT;
    double* VS = ws->VS;
    double* VS1 = ws->VS1;
    double* wr = ws->wr;
    double* wi = ws->wi;
    double* wrt = ws->wrt;
    double* wit = ws->wit;
    double* wrtmp = ws->wrtmp;
    double* witmp = ws->witmp;
    double* work = ws->work;
    int* iwork = ws->iwork;
    int* bwork = ws->bwork;
    double* result = ws->result;

    double ulpinv = 1.0 / dlamch("P");

    for (int j = 0; j < 17; j++) {
        result[j] = -1.0;
    }

    if (n == 0) {
        return;
    }

    /* Generate matrix */
    int iinfo = generate_matrix(n, jtype, A, lda, work, iwork, ws->rng_state);
    if (iinfo != 0) {
        result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Determine workspace size (ddrvsx.f lines 778-784) */
    int nnwork;
    if (iwk == 1) {
        nnwork = 3 * n;
    } else {
        nnwork = 3 * n;
        if (2 * n * n > nnwork) nnwork = 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    /* Call dget24 with COMP=0 (ddrvsx.f lines 786-790) */
    int info = 0;
    int any_fail = 0;

    dget24(0, jtype, THRESH, n, A, lda, H, HT,
           wr, wi, wrt, wit, wrtmp, witmp,
           VS, ldvs, VS1,
           0.0, 0.0,
           0, NULL,
           result, work, nnwork, iwork, bwork, &info);

    /* Check for RESULT(j) > THRESH (ddrvsx.f lines 796-820) */
    for (int j = 0; j < 15; j++) {
        if (result[j] >= 0.0 && result[j] >= THRESH) {
            print_message("N=%d, IWK=%d, type %d, test(%d)=%g\n",
                          n, iwk, jtype, j + 1, result[j]);
            any_fail = 1;
        }
    }

    assert_int_equal(any_fail, 0);
}

/**
 * Run tests for a single precomputed matrix.
 *
 * Based on ddrvsx.f lines 834-883.
 */
static void run_ddrvsx_precomp(ddrvsx_params_t* params)
{
    int idx = params->precomp_idx;

    ddrvsx_workspace_t* ws = g_ws;
    int lda = ws->nmax;
    int ldvs = ws->nmax;

    double* A = ws->A;
    double* H = ws->H;
    double* HT = ws->HT;
    double* VS = ws->VS;
    double* VS1 = ws->VS1;
    double* wr = ws->wr;
    double* wi = ws->wi;
    double* wrt = ws->wrt;
    double* wit = ws->wit;
    double* wrtmp = ws->wrtmp;
    double* witmp = ws->witmp;
    double* work = ws->work;
    int* iwork = ws->iwork;
    int* bwork = ws->bwork;
    double* result = ws->result;

    const dsx_precomputed_t* pc = &DSX_PRECOMPUTED[idx];
    int n = pc->n;

    for (int j = 0; j < 17; j++) {
        result[j] = -1.0;
    }

    /* Copy precomputed matrix into workspace A with proper leading dimension */
    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            A[row + col * lda] = pc->A[row + col * n];
        }
    }

    int info = 0;

    /* Call dget24 with COMP=1 (ddrvsx.f lines 848-851) */
    dget24(1, 22, THRESH, n, A, lda, H, HT,
           wr, wi, wrt, wit, wrtmp, witmp,
           VS, ldvs, VS1,
           pc->rcdein, pc->rcdvin,
           pc->nslct, pc->islct,
           result, work, ws->lwork, iwork, bwork, &info);

    /* Check for RESULT(j) > THRESH (ddrvsx.f lines 855-879) */
    int any_fail = 0;
    for (int j = 0; j < 17; j++) {
        if (result[j] >= 0.0 && result[j] >= THRESH) {
            print_message("N=%d, input example=%d, test(%d)=%g\n",
                          n, idx + 1, j + 1, result[j]);
            any_fail = 1;
        }
    }

    assert_int_equal(any_fail, 0);
}

/**
 * Test function wrappers.
 */
static void test_ddrvsx_random_case(void** state)
{
    ddrvsx_params_t* params = *state;
    run_ddrvsx_random(params);
}

static void test_ddrvsx_precomp_case(void** state)
{
    ddrvsx_params_t* params = *state;
    run_ddrvsx_precomp(params);
}

/*
 * Generate all parameter combinations.
 * Random: NNVAL * MAXTYP * 2 = 7 * 21 * 2 = 294 tests
 * Precomputed: DSX_NUM_PRECOMPUTED = 26 tests
 * Total: 320 tests
 */

#define MAX_RANDOM_TESTS (NNVAL * MAXTYP * 2)
#define MAX_TESTS (MAX_RANDOM_TESTS + DSX_NUM_PRECOMPUTED)

static ddrvsx_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    /* Random tests: n x jtype x iwk */
    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            for (int iwk = 1; iwk <= 2; iwk++) {
                ddrvsx_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->jtype = jtype;
                p->iwk = iwk;
                p->precomp_idx = -1;
                snprintf(p->name, sizeof(p->name),
                         "ddrvsx_n%d_type%d_wk%d", n, jtype, iwk);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvsx_random_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }

    /* Precomputed tests */
    for (int idx = 0; idx < DSX_NUM_PRECOMPUTED; idx++) {
        ddrvsx_params_t* p = &g_params[g_num_tests];
        p->n = DSX_PRECOMPUTED[idx].n;
        p->jtype = 22;
        p->iwk = 2;
        p->precomp_idx = idx;
        snprintf(p->name, sizeof(p->name),
                 "ddrvsx_precomp_%d_n%d", idx + 1, p->n);

        g_tests[g_num_tests].name = p->name;
        g_tests[g_num_tests].test_func = test_ddrvsx_precomp_case;
        g_tests[g_num_tests].setup_func = NULL;
        g_tests[g_num_tests].teardown_func = NULL;
        g_tests[g_num_tests].initial_state = p;

        g_num_tests++;
    }
}

/* ===== Main ===== */

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("ddrvsx", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
