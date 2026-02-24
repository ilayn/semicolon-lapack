/**
 * @file test_sdrvgt.c
 * @brief DDRVGT tests the driver routines SGTSV and SGTSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvgt.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include "semicolon_cblas.h"
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  12
#define NTESTS  6
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

/* Routines under test */
/* Supporting routines */
/* Verification routines */
/* Matrix generation */
/* Utilities */
typedef struct {
    INT n;
    INT imat;
    INT ifact;      /* 0='F', 1='N' */
    INT itran;      /* 0='N', 1='T', 2='C' */
    char name[64];
} ddrvgt_params_t;

typedef struct {
    f32* A;      /* Tridiagonal storage: DL, D, DU = 3*N */
    f32* AF;     /* Factored: DLF, DF, DUF, DU2 = 4*N */
    f32* B;
    f32* X;
    f32* XACT;
    f32* WORK;
    f32* RWORK;
    INT* IWORK;
} ddrvgt_workspace_t;

static ddrvgt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvgt_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    /* WORK needs to be large enough for slatms which uses n*n for full matrix
     * plus additional workspace for slagge/slagsy (roughly 2*n more) */
    INT lwork = nmax * nmax + 4 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->A = calloc(3 * nmax, sizeof(f32));
    g_workspace->AF = calloc(4 * nmax, sizeof(f32));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(nmax > 2 * NRHS ? nmax : 2 * NRHS, sizeof(f32));
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

static void run_ddrvgt_single(INT n, INT imat, INT ifact, INT itran)
{
    static const char* FACTS[] = {"F", "N"};
    static const char* TRANSS[] = {"N", "T", "C"};

    ddrvgt_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];
    const char* trans = TRANSS[itran];

    INT m = (n > 1) ? n - 1 : 0;
    INT lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    /* Pointers to tridiagonal components in A:
     * A[0..m-1] = DL (subdiagonal)
     * A[m..m+n-1] = D (diagonal)
     * A[m+n..m+2n-2] = DU (superdiagonal) */
    f32* DL = ws->A;
    f32* D = ws->A + m;
    f32* DU = ws->A + m + n;

    /* Pointers to factored components in AF */
    f32* DLF = ws->AF;
    f32* DF = ws->AF + m;
    f32* DUF = ws->AF + m + n;
    f32* DU2 = ws->AF + m + 2 * n;

    INT zerot = (imat >= 8 && imat <= 10);
    INT izero = 0;

    /* Set up parameters with SLATB4 */
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix.
     * For types 1-6, each type has a unique seed.
     * For types 7-12, they share the same base seed because types 8-10
     * are singular versions of the type 7 matrix (with zeroed columns),
     * and types 11-12 are scaled versions of type 7. */
    uint64_t seed;
    if (imat <= 6) {
        seed = 1988 + n * 1000 + imat * 100;
    } else {
        seed = 1988 + n * 1000 + 7 * 100;  /* Types 7-12 share type 7's seed */
    }
    uint64_t rng_state[4];
    rng_seed(rng_state, seed);
    INT info;

    if (imat <= 6) {
        /* Types 1-6: generate matrices of known condition number.
         * Generate in band storage with LDA=3, using KOFF offset. */
        INT nmax1 = (1 > n) ? 1 : n;
        INT koff = (2 - ku > 3 - nmax1) ? 2 - ku : 3 - nmax1;
        koff -= 1;  /* convert to 0-based */

        slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
               anorm, kl, ku, "Z", &ws->AF[koff], 3, ws->WORK, &info, rng_state);
        if (info != 0) {
            fail_msg("SLATMS info=%d", info);
            return;
        }
        izero = 0;

        /* Extract tridiagonal from AF with stride 3.
         * AF[1 + 3*i] = diagonal, AF[3 + 3*i] = subdiag, AF[2 + 3*i] = superdiag */
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
        /* Types 7-12: generate tridiagonal matrices with unknown condition */

        if (!zerot) {
            /* Generate matrix with elements from [-1,1] */
            for (INT i = 0; i < n + 2 * m; i++) {
                ws->A[i] = 2.0f * rng_uniform_f32(rng_state) - 1.0f;
            }
            if (anorm != 1.0f) {
                cblas_sscal(n + 2 * m, anorm, ws->A, 1);
            }
        }

        /* Zero out elements for singular matrix types 8-10.
         * LAPACK uses flat array A(1:3N-2) = [DL(1:M), D(1:N), DU(1:M)]
         * imat=8: Zero first column → A(N) = 0 (D[0]) and A(1) = 0 (DL[0])
         * imat=9: Zero last column → A(3N-2) = 0 (DU[M-1]) and A(2N-1) = 0 (D[N-1])
         * imat=10: Zero middle columns → Zero elements from IZERO to N
         */
        if (imat == 8) {
            izero = 1;
            D[0] = 0.0f;
            if (n > 1) DL[0] = 0.0f;
        } else if (imat == 9) {
            izero = n;
            if (n > 1) DU[n - 2] = 0.0f;
            D[n - 1] = 0.0f;
        } else if (imat == 10) {
            izero = (n + 1) / 2;
            /* Zero from column izero to n (1-indexed), i.e., indices izero-1 to n-1 (0-indexed).
             * For N=10, IZERO=5: zero columns 5-10 (1-indexed) = indices 4-9 (0-indexed). */
            for (INT i = izero - 1; i < n; i++) {
                if (i < n - 1) DU[i] = 0.0f;
                D[i] = 0.0f;
                if (i > 0) DL[i - 1] = 0.0f;
            }
        } else {
            izero = 0;
        }
    }

    /* Skip FACT='F' for singular matrices */
    if (zerot && ifact == 0) {
        return;
    }

    /*
     * Compute condition numbers RCONDO and RCONDI.
     * For FACT='N', reuse values from FACT='F' iteration.
     * Since tests are independent, we always compute them.
     */
    f32 rcondo = 0.0f, rcondi = 0.0f;

    if (zerot) {
        rcondo = 0.0f;
        rcondi = 0.0f;
    } else if (n == 0) {
        rcondo = 1.0f / cndnum;
        rcondi = 1.0f / cndnum;
    } else {
        /* Copy tridiagonal to AF and factor */
        cblas_scopy(n + 2 * m, ws->A, 1, ws->AF, 1);

        f32 anormo = slangt("1", n, DL, D, DU);
        f32 anormi = slangt("I", n, DL, D, DU);

        sgttrf(n, DLF, DF, DUF, DU2, ws->IWORK, &info);

        /* Compute inverse norm using SGTTRS */
        f32 ainvnm = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) ws->X[j] = 0.0f;
            ws->X[i] = 1.0f;
            sgttrs("N", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            f32 colsum = cblas_sasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormo <= 0.0f || ainvnm <= 0.0f) {
            rcondo = 1.0f;
        } else {
            rcondo = (1.0f / anormo) / ainvnm;
        }

        /* Compute infinity-norm condition number */
        ainvnm = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) ws->X[j] = 0.0f;
            ws->X[i] = 1.0f;
            sgttrs("T", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            f32 colsum = cblas_sasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormi <= 0.0f || ainvnm <= 0.0f) {
            rcondi = 1.0f;
        } else {
            rcondi = (1.0f / anormi) / ainvnm;
        }
    }

    f32 rcondc = (itran == 0) ? rcondo : rcondi;

    /* Generate NRHS random solution vectors */
    rng_seed(rng_state, seed + (uint64_t)itran);
    for (INT j = 0; j < NRHS; j++) {
        for (INT i = 0; i < n; i++) {
            ws->XACT[j * lda + i] = 2.0f * rng_uniform_f32(rng_state) - 1.0f;
        }
    }

    /* Set the right hand side using SLAGTM */
    slagtm(trans, n, NRHS, 1.0f, DL, D, DU, ws->XACT, lda, 0.0f, ws->B, lda);

    /* --- Test SGTSV --- */
    if (ifact == 1 && itran == 0) {
        cblas_scopy(n + 2 * m, ws->A, 1, ws->AF, 1);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        sgtsv(n, NRHS, DLF, DF, DUF, ws->X, lda, &info);

        if (zerot) {
            if (info <= 0) {
                fail_msg("SGTSV: expected INFO > 0 for singular matrix, got %d", info);
                return;
            }
        } else if (info != izero) {
            fail_msg("SGTSV info=%d expected=%d", info, izero);
            return;
        }

        INT nt = 1;
        if (izero == 0 && info == 0) {
            /* TEST 2: Check residual of computed solution */
            slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            sgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (INT k = 1; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SGTSV test %d failed: result=%e >= thresh=%e",
                         k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test SGTSVX --- */
    if (ifact > 0) {
        /* Initialize AF to zero */
        for (INT i = 0; i < 3 * n - 2; i++) ws->AF[i] = 0.0f;
    }
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    f32 rcond;
    sgtsvx(fact, trans, n, NRHS, DL, D, DU, DLF, DF, DUF, DU2,
           ws->IWORK, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->IWORK[n], &info);

    if (zerot) {
        if (info <= 0) {
            fail_msg("SGTSVX: expected INFO > 0 for singular matrix, got %d", info);
            return;
        }
    } else if (info != izero) {
        fail_msg("SGTSVX info=%d expected=%d", info, izero);
        return;
    }

    INT k1;
    INT nt = 5;
    if (ifact >= 1) {
        /* TEST 1: Reconstruct matrix from factors */
        sgtt01(n, DL, D, DU, DLF, DF, DUF, DU2, ws->IWORK,
               ws->WORK, lda, &result[0]);
        k1 = 1;
    } else {
        k1 = 2;
    }

    if (info == 0) {
        /* TEST 2: Check residual of computed solution */
        slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        sgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        sgtt05(trans, n, NRHS, DL, D, DU, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
        nt = 5;
    }

    /* TEST 6: Compare RCOND from SGTSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    for (INT k = k1 - 1; k < nt; k++) {
        if (result[k] >= THRESH) {
            fail_msg("SGTSVX FACT=%s TRANS=%s test %d: result=%e >= thresh=%e",
                     fact, trans, k + 1, (double)result[k], (double)THRESH);
        }
    }
    if (result[5] >= THRESH) {
        fail_msg("SGTSVX FACT=%s TRANS=%s test 6: result=%e >= thresh=%e",
                 fact, trans, (double)result[5], (double)THRESH);
    }
}

static void test_ddrvgt_case(void** state)
{
    ddrvgt_params_t* p = *state;
    run_ddrvgt_single(p->n, p->imat, p->ifact, p->itran);
}

#define MAX_TESTS 3000

static ddrvgt_params_t g_params[MAX_TESTS];
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
                    ddrvgt_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->ifact = ifact;
                    p->itran = itran;
                    snprintf(p->name, sizeof(p->name),
                             "n%d_t%d_%s_%s",
                             n, imat, FACTS[ifact], TRANSS[itran]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_ddrvgt_case;
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
    return _cmocka_run_group_tests("ddrvgt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
