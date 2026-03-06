/**
 * @file test_cdrvgt.c
 * @brief ZDRVGT tests the driver routines CGTSV and CGTSVX.
 *
 * Port of LAPACK TESTING/LIN/zdrvgt.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include "semicolon_cblas.h"
#include <math.h>

/* Test parameters - matching LAPACK zchkaa.f defaults */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  12
#define NTESTS  6
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

typedef struct {
    INT n;
    INT imat;
    INT ifact;      /* 0='F', 1='N' */
    INT itran;      /* 0='N', 1='T', 2='C' */
    char name[64];
} zdrvgt_params_t;

typedef struct {
    c64* A;      /* Tridiagonal storage: DL, D, DU = 3*N */
    c64* AF;     /* Factored: DLF, DF, DUF, DU2 = 4*N */
    c64* B;
    c64* X;
    c64* XACT;
    c64* WORK;
    f32* RWORK;
    INT* IWORK;
} zdrvgt_workspace_t;

static zdrvgt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvgt_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    INT lwork = nmax * nmax + 4 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->A = calloc(3 * nmax, sizeof(c64));
    g_workspace->AF = calloc(4 * nmax, sizeof(c64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->WORK = calloc(lwork, sizeof(c64));
    g_workspace->RWORK = calloc(2 * NRHS + nmax, sizeof(f32));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(INT));

    if (!g_workspace->A || !g_workspace->AF || !g_workspace->B ||
        !g_workspace->X || !g_workspace->XACT || !g_workspace->WORK ||
        !g_workspace->RWORK || !g_workspace->IWORK) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AF);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zdrvgt_single(INT n, INT imat, INT ifact, INT itran)
{
    static const char* FACTS[] = {"F", "N"};
    static const char* TRANSS[] = {"N", "T", "C"};

    zdrvgt_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];
    const char* trans = TRANSS[itran];

    INT m = (n > 1) ? n - 1 : 0;
    INT lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    c64* DL = ws->A;
    c64* D = ws->A + m;
    c64* DU = ws->A + m + n;

    c64* DLF = ws->AF;
    c64* DF = ws->AF + m;
    c64* DUF = ws->AF + m + n;
    c64* DU2 = ws->AF + m + 2 * n;

    INT zerot = (imat >= 8 && imat <= 10);
    INT izero = 0;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    clatb4("CGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t seed;
    if (imat <= 6) {
        seed = 1988 + n * 1000 + imat * 100;
    } else {
        seed = 1988 + n * 1000 + 7 * 100;
    }
    uint64_t rng_state[4];
    rng_seed(rng_state, seed);
    INT info;

    if (imat <= 6) {
        INT nmax1 = (1 > n) ? 1 : n;
        INT koff = (2 - ku > 3 - nmax1) ? 2 - ku : 3 - nmax1;
        koff -= 1;

        clatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
               anorm, kl, ku, "Z", &ws->AF[koff], 3, ws->WORK, &info, rng_state);
        if (info != 0) {
            fail_msg("CLATMS info=%d", info);
            return;
        }
        izero = 0;

        for (INT i = 0; i < n; i++) {
            D[i] = ws->AF[1 + 3 * i];
        }
        if (n > 1) {
            for (INT i = 0; i < n - 1; i++) {
                DL[i] = ws->AF[3 + 3 * i];
            }
            for (INT i = 0; i < n - 1; i++) {
                DU[i] = ws->AF[2 + 3 * i];
            }
        }
    } else {
        if (!zerot) {
            for (INT i = 0; i < n + 2 * m; i++) {
                ws->A[i] = CMPLXF(2.0f * rng_uniform_f32(rng_state) - 1.0f,
                                  2.0f * rng_uniform_f32(rng_state) - 1.0f);
            }
            if (anorm != 1.0f) {
                cblas_csscal(n + 2 * m, anorm, ws->A, 1);
            }
        }

        if (imat == 8) {
            izero = 1;
            D[0] = CMPLXF(0.0f, 0.0f);
            if (n > 1) DL[0] = CMPLXF(0.0f, 0.0f);
        } else if (imat == 9) {
            izero = n;
            if (n > 1) DU[n - 2] = CMPLXF(0.0f, 0.0f);
            D[n - 1] = CMPLXF(0.0f, 0.0f);
        } else if (imat == 10) {
            izero = (n + 1) / 2;
            for (INT i = izero - 1; i < n; i++) {
                if (i < n - 1) DU[i] = CMPLXF(0.0f, 0.0f);
                D[i] = CMPLXF(0.0f, 0.0f);
                if (i > 0) DL[i - 1] = CMPLXF(0.0f, 0.0f);
            }
        } else {
            izero = 0;
        }
    }

    if (zerot && ifact == 0) {
        return;
    }

    f32 rcondo = 0.0f, rcondi = 0.0f;

    if (zerot) {
        rcondo = 0.0f;
        rcondi = 0.0f;
    } else if (n == 0) {
        rcondo = 1.0f / cndnum;
        rcondi = 1.0f / cndnum;
    } else {
        cblas_ccopy(n + 2 * m, ws->A, 1, ws->AF, 1);

        f32 anormo = clangt("1", n, DL, D, DU);
        f32 anormi = clangt("I", n, DL, D, DU);

        cgttrf(n, DLF, DF, DUF, DU2, ws->IWORK, &info);

        f32 ainvnm = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) ws->X[j] = CMPLXF(0.0f, 0.0f);
            ws->X[i] = CMPLXF(1.0f, 0.0f);
            cgttrs("N", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            f32 colsum = cblas_scasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormo <= 0.0f || ainvnm <= 0.0f) {
            rcondo = 1.0f;
        } else {
            rcondo = (1.0f / anormo) / ainvnm;
        }

        ainvnm = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) ws->X[j] = CMPLXF(0.0f, 0.0f);
            ws->X[i] = CMPLXF(1.0f, 0.0f);
            cgttrs("C", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            f32 colsum = cblas_scasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormi <= 0.0f || ainvnm <= 0.0f) {
            rcondi = 1.0f;
        } else {
            rcondi = (1.0f / anormi) / ainvnm;
        }
    }

    f32 rcondc = (itran == 0) ? rcondo : rcondi;

    rng_seed(rng_state, seed + (uint64_t)itran);
    for (INT j = 0; j < NRHS; j++) {
        for (INT i = 0; i < n; i++) {
            ws->XACT[j * lda + i] = CMPLXF(2.0f * rng_uniform_f32(rng_state) - 1.0f,
                                            2.0f * rng_uniform_f32(rng_state) - 1.0f);
        }
    }

    clagtm(trans, n, NRHS, 1.0f, DL, D, DU, ws->XACT, lda, 0.0f, ws->B, lda);

    /* --- Test CGTSV --- */
    if (ifact == 1 && itran == 0) {
        cblas_ccopy(n + 2 * m, ws->A, 1, ws->AF, 1);
        clacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        cgtsv(n, NRHS, DLF, DF, DUF, ws->X, lda, &info);

        if (zerot) {
            if (info <= 0) {
                fail_msg("CGTSV: expected INFO > 0 for singular matrix, got %d", info);
                return;
            }
        } else if (info != izero) {
            fail_msg("CGTSV info=%d expected=%d", info, izero);
            return;
        }

        INT nt = 1;
        if (izero == 0 && info == 0) {
            clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            cgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (INT k = 1; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("CGTSV test %d failed: result=%e >= thresh=%e",
                         k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test CGTSVX --- */
    if (ifact > 0) {
        for (INT i = 0; i < 3 * n - 2; i++) ws->AF[i] = CMPLXF(0.0f, 0.0f);
    }
    claset("Full", n, NRHS, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), ws->X, lda);

    f32 rcond;
    cgtsvx(fact, trans, n, NRHS, DL, D, DU, DLF, DF, DUF, DU2,
           ws->IWORK, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->RWORK[2 * NRHS], &info);

    if (zerot) {
        if (info <= 0) {
            fail_msg("CGTSVX: expected INFO > 0 for singular matrix, got %d", info);
            return;
        }
    } else if (info != izero) {
        fail_msg("CGTSVX info=%d expected=%d", info, izero);
        return;
    }

    INT k1;
    INT nt = 5;
    if (ifact >= 1) {
        cgtt01(n, DL, D, DU, DLF, DF, DUF, DU2, ws->IWORK,
               ws->WORK, lda, &result[0]);
        k1 = 1;
    } else {
        k1 = 2;
    }

    if (info == 0) {
        clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        cgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

        cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        cgtt05(trans, n, NRHS, DL, D, DU, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
        nt = 5;
    }

    result[5] = sget06(rcond, rcondc);

    for (INT k = k1 - 1; k < nt; k++) {
        if (result[k] >= THRESH) {
            fail_msg("CGTSVX FACT=%s TRANS=%s test %d: result=%e >= thresh=%e",
                     fact, trans, k + 1, (double)result[k], (double)THRESH);
        }
    }
    if (result[5] >= THRESH) {
        fail_msg("CGTSVX FACT=%s TRANS=%s test 6: result=%e >= thresh=%e",
                 fact, trans, (double)result[5], (double)THRESH);
    }
}

static void test_zdrvgt_case(void** state)
{
    zdrvgt_params_t* p = *state;
    run_zdrvgt_single(p->n, p->imat, p->ifact, p->itran);
}

#define MAX_TESTS 3000

static zdrvgt_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    static const char* FACTS[] = {"F", "N"};
    static const char* TRANSS[] = {"N", "T", "C"};

    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nimat = (n <= 0) ? 1 : NTYPES;

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 8 && imat <= 10);

            for (INT ifact = 0; ifact < 2; ifact++) {
                if (zerot && ifact == 0) continue;

                for (INT itran = 0; itran < 3; itran++) {
                    zdrvgt_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->ifact = ifact;
                    p->itran = itran;
                    snprintf(p->name, sizeof(p->name),
                             "n%d_t%d_%s_%s",
                             n, imat, FACTS[ifact], TRANSS[itran]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zdrvgt_case;
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
    (void)_cmocka_run_group_tests("zdrvgt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
