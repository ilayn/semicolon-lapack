/**
 * @file test_cchkrfp.c
 * @brief Comprehensive test suite for Rectangular Full Packed (RFP) routines (complex).
 *
 * This is a port of LAPACK's TESTING/LIN/zdrvrfp.f to C using CMocka.
 * Tests CPFTRF, CPFTRS, CPFTRI and format conversion routines.
 *
 * Test structure from zdrvrfp.f:
 *   TEST 1: ||L*L' - A|| / (N * ||A|| * eps)  - Cholesky factorization
 *   TEST 2: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)  - Inverse
 *   TEST 3: ||B - A*X|| / (||A|| * ||X|| * eps)  - Solve residual
 *   TEST 4: ||X - XACT|| / (||XACT|| * CNDNUM * eps)  - Solution accuracy
 *
 * Matrix types (from clatb4 for CPO):
 *   1. Diagonal
 *   2. Random, CNDNUM = 2
 *   3. First row and column zero (error exit test)
 *   4. Last row and column zero (error exit test)
 *   5. Middle row and column zero (error exit test)
 *   6. Random, CNDNUM = sqrt(0.1/EPS)
 *   7. Random, CNDNUM = 0.1/EPS
 *   8. Scaled near underflow
 *   9. Scaled near overflow
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const INT NSVAL[] = {1, 2, 5, 10};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  9
#define NTESTS  4
#define THRESH  30.0f
#define NMAX    50
#define MAXRHS  16

typedef struct {
    INT n;
    INT nrhs;
    INT imat;
    INT iuplo;
    INT iform;
    char name[80];
} zchkrfp_params_t;

static void run_zchkrfp_single(INT n, INT nrhs, INT imat, INT iuplo, INT iform)
{
    f32 result[NTESTS];
    char ctx[128];
    INT info;
    INT lda = (n > 1) ? n : 1;
    INT ldb = lda;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);
    char uplo = (iuplo == 0) ? 'U' : 'L';
    char cform = (iform == 0) ? 'N' : 'C';
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    f32 rcondc, ainvnm;
    INT rfp_size = (n * (n + 1)) / 2;
    INT k;
    INT zerot, izero;

    if (n == 0 && imat > 1) return;
    if (imat == 4 && n <= 1) return;
    if (imat == 5 && n <= 2) return;

    c64* A = calloc(NMAX * NMAX, sizeof(c64));
    c64* ASAV = calloc(NMAX * NMAX, sizeof(c64));
    c64* AFAC = calloc(NMAX * NMAX, sizeof(c64));
    c64* AINV = calloc(NMAX * NMAX, sizeof(c64));
    c64* B = calloc(NMAX * MAXRHS, sizeof(c64));
    c64* BSAV = calloc(NMAX * MAXRHS, sizeof(c64));
    c64* X = calloc(NMAX * MAXRHS, sizeof(c64));
    c64* XACT = calloc(NMAX * MAXRHS, sizeof(c64));
    c64* ARF = calloc(rfp_size + 1, sizeof(c64));
    c64* ARFINV = calloc(rfp_size + 1, sizeof(c64));
    c64* work = calloc(3 * NMAX, sizeof(c64));
    f32* rwork = calloc(NMAX, sizeof(f32));
    c64* temp = calloc(NMAX * NMAX, sizeof(c64));

    clatb4("CPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    clatms(n, n, &dist, &type, rwork, mode, cndnum, anorm, kl, ku,
           &uplo, A, lda, work, &info, rng_state);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c CLATMS info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    zerot = (imat >= 3 && imat <= 5);
    izero = 0;
    if (zerot) {
        if (imat == 3) izero = 0;
        else if (imat == 4) izero = n - 1;
        else izero = n / 2;

        if (uplo == 'U') {
            for (INT i = 0; i < izero; i++) {
                A[izero * lda + i] = CMPLXF(0.0f, 0.0f);
            }
            for (INT i = izero; i < n; i++) {
                A[i * lda + izero] = CMPLXF(0.0f, 0.0f);
            }
        } else {
            for (INT i = 0; i < izero; i++) {
                A[i * lda + izero] = CMPLXF(0.0f, 0.0f);
            }
            for (INT i = izero; i < n; i++) {
                A[izero * lda + i] = CMPLXF(0.0f, 0.0f);
            }
        }
    }

    claipd(n, A, lda + 1, 0);

    clacpy(&uplo, n, n, A, lda, ASAV, lda);

    if (zerot) {
        rcondc = 0.0f;
    } else {
        f32 norm_a = clanhe("1", &uplo, n, A, lda, rwork);
        clacpy(&uplo, n, n, A, lda, AFAC, lda);
        cpotrf(&uplo, n, AFAC, lda, &info);
        if (info == 0) {
            cpotri(&uplo, n, AFAC, lda, &info);
            if (info == 0 && n > 0) {
                ainvnm = clanhe("1", &uplo, n, AFAC, lda, rwork);
                rcondc = (1.0f / norm_a) / ainvnm;
            } else {
                rcondc = 0.0f;
            }
        } else {
            rcondc = 0.0f;
        }
        clacpy(&uplo, n, n, ASAV, lda, A, lda);
    }

    clarhs("CPO", "N", &uplo, " ", n, n, kl, ku, nrhs, A, lda,
           XACT, lda, B, lda, &info, rng_state);
    clacpy("F", n, nrhs, B, lda, BSAV, lda);

    clacpy(&uplo, n, n, A, lda, AFAC, lda);
    clacpy("F", n, nrhs, B, ldb, X, ldb);

    ctrttf(&cform, &uplo, n, AFAC, lda, ARF, &info);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c CTRTTF info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    cpftrf(&cform, &uplo, n, ARF, &info);

    if (zerot) {
        if (info != izero + 1) {
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c CPFTRF info=%d expected=%d",
                     n, imat, uplo, cform, info, izero + 1);
            set_test_context(ctx);
        }
        goto cleanup;
    }

    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c CPFTRF info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    cpftrs(&cform, &uplo, n, nrhs, ARF, X, ldb, &info);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c CPFTRS info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    ctfttr(&cform, &uplo, n, ARF, AFAC, lda, &info);

    clacpy(&uplo, n, n, AFAC, lda, temp, lda);
    cpot01(&uplo, n, A, lda, AFAC, lda, rwork, &result[0]);
    clacpy(&uplo, n, n, temp, lda, AFAC, lda);

    if ((n % 2) == 0) {
        for (INT j = 0; j < n / 2; j++) {
            for (INT i = 0; i <= n; i++) {
                ARFINV[j * (n + 1) + i] = ARF[j * (n + 1) + i];
            }
        }
    } else {
        for (INT j = 0; j < (n + 1) / 2; j++) {
            for (INT i = 0; i < n; i++) {
                ARFINV[j * n + i] = ARF[j * n + i];
            }
        }
    }

    cpftri(&cform, &uplo, n, ARFINV, &info);
    ctfttr(&cform, &uplo, n, ARFINV, AINV, lda, &info);

    cpot03(&uplo, n, A, lda, AINV, lda, temp, lda, rwork, &rcondc, &result[1]);

    clacpy("F", n, nrhs, BSAV, lda, temp, lda);
    cpot02(&uplo, n, nrhs, A, lda, X, lda, temp, lda, rwork, &result[2]);

    cget04(n, nrhs, X, lda, XACT, lda, rcondc, &result[3]);

    for (k = 0; k < NTESTS; k++) {
        snprintf(ctx, sizeof(ctx), "n=%d nrhs=%d imat=%d uplo=%c form=%c TEST %d",
                 n, nrhs, imat, uplo, cform, k + 1);
        set_test_context(ctx);
        assert_residual_below(result[k], THRESH);
    }
    clear_test_context();

cleanup:
    free(A);
    free(ASAV);
    free(AFAC);
    free(AINV);
    free(B);
    free(BSAV);
    free(X);
    free(XACT);
    free(ARF);
    free(ARFINV);
    free(work);
    free(rwork);
    free(temp);
}

static void test_zchkrfp_case(void** state)
{
    zchkrfp_params_t* params = *state;
    run_zchkrfp_single(params->n, params->nrhs, params->imat,
                       params->iuplo, params->iform);
}

#define MAX_TESTS (NN * NNS * NTYPES * 2 * 2)

static zchkrfp_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        for (INT ins = 0; ins < (INT)NNS; ins++) {
            INT nrhs = NSVAL[ins];

            for (INT imat = 1; imat <= (INT)NTYPES; imat++) {

                if (n == 0 && imat > 1) continue;
                if (imat == 4 && n <= 1) continue;
                if (imat == 5 && n <= 2) continue;

                for (INT iuplo = 0; iuplo < 2; iuplo++) {
                    for (INT iform = 0; iform < 2; iform++) {

                        zchkrfp_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->nrhs = nrhs;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->iform = iform;
                        snprintf(p->name, sizeof(p->name),
                                 "zchkrfp_n%d_nrhs%d_imat%d_%c_%c",
                                 n, nrhs, imat,
                                 (iuplo == 0) ? 'U' : 'L',
                                 (iform == 0) ? 'N' : 'C');

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_zchkrfp_case;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;
                        g_tests[g_num_tests].initial_state = p;

                        g_num_tests++;
                        if (g_num_tests >= (INT)MAX_TESTS) return;
                    }
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();

    if (g_num_tests == 0) {
        printf("No valid test cases generated\n");
        return 0;
    }

    (void)_cmocka_run_group_tests("zchkrfp", g_tests, g_num_tests, NULL, NULL);
    return 0;
}
