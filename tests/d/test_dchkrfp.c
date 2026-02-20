/**
 * @file test_dchkrfp.c
 * @brief Comprehensive test suite for Rectangular Full Packed (RFP) routines.
 *
 * This is a port of LAPACK's TESTING/LIN/dchkrfp.f and ddrvrfp.f to C using CMocka.
 * Tests DPFTRF, DPFTRS, DPFTRI and format conversion routines.
 *
 * Test structure from ddrvrfp.f:
 *   TEST 1: ||L*L' - A|| / (N * ||A|| * eps)  - Cholesky factorization
 *   TEST 2: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)  - Inverse
 *   TEST 3: ||B - A*X|| / (||A|| * ||X|| * eps)  - Solve residual
 *   TEST 4: ||X - XACT|| / (||XACT|| * CNDNUM * eps)  - Solution accuracy
 *
 * Matrix types (from dlatb4 for DPO):
 *   1. Diagonal
 *   2. Random, CNDNUM = 2
 *   3. First row and column zero (error exit test)
 *   4. Last row and column zero (error exit test)
 *   5. Middle row and column zero (error exit test)
 *   6. Random, CNDNUM = sqrt(0.1/EPS)
 *   7. Random, CNDNUM = 0.1/EPS
 *   8. Scaled near underflow
 *   9. Scaled near overflow
 *
 * Parameters:
 *   N values: 0, 1, 2, 3, 5, 10, 16, 50
 *   NRHS values: 1, 2, 5, 10
 *   UPLO: 'U', 'L'
 *   TRANS (RFP format): 'N', 'T'
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const int NSVAL[] = {1, 2, 5, 10};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  9
#define NTESTS  4
#define THRESH  30.0
#define NMAX    50
#define MAXRHS  16

extern f64 dlamch(const char* cmach);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* const restrict A, const int lda,
                   f64* const restrict B, const int ldb);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* a, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   f64* X, const int ldx, f64* B, const int ldb,
                   int* info, uint64_t state[static 4]);
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dpftrf(const char* transr, const char* uplo, const int n,
                   f64* arf, int* info);
extern void dpftrs(const char* transr, const char* uplo, const int n,
                   const int nrhs, const f64* arf, f64* b, const int ldb,
                   int* info);
extern void dpftri(const char* transr, const char* uplo, const int n,
                   f64* arf, int* info);
extern void dpotrf(const char* uplo, const int n, f64* a, const int lda,
                   int* info);
extern void dpotri(const char* uplo, const int n, f64* a, const int lda,
                   int* info);
extern void dtrttf(const char* transr, const char* uplo, const int n,
                   const f64* a, const int lda, f64* arf, int* info);
extern void dtfttr(const char* transr, const char* uplo, const int n,
                   const f64* arf, f64* a, const int lda, int* info);
void dpot01(const char* uplo, int n, const f64* A, int lda,
            f64* AFAC, int ldafac, f64* rwork, f64* resid);
void dpot02(const char* uplo, int n, int nrhs, const f64* A, int lda,
            const f64* X, int ldx, f64* B, int ldb,
            f64* rwork, f64* resid);
void dpot03(const char* uplo, int n, const f64* A, int lda,
            const f64* AINV, int ldainv, f64* work, int ldwork,
            f64* rwork, f64* rcond, f64* resid);
void dget04(int n, int nrhs, const f64* X, int ldx,
            const f64* XACT, int ldxact, f64 rcond, f64* resid);

typedef struct {
    int n;
    int nrhs;
    int imat;
    int iuplo;
    int iform;
    char name[80];
} dchkrfp_params_t;

static void run_dchkrfp_single(int n, int nrhs, int imat, int iuplo, int iform)
{
    f64 result[NTESTS];
    char ctx[128];
    int info;
    int lda = (n > 1) ? n : 1;
    int ldb = lda;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);
    char uplo = (iuplo == 0) ? 'U' : 'L';
    char cform = (iform == 0) ? 'N' : 'T';
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    f64 rcondc, ainvnm;
    int rfp_size = (n * (n + 1)) / 2;
    int k;
    int zerot, izero;

    if (n == 0 && imat > 1) return;
    if (imat == 4 && n <= 1) return;
    if (imat == 5 && n <= 2) return;

    f64* A = calloc(NMAX * NMAX, sizeof(f64));
    f64* ASAV = calloc(NMAX * NMAX, sizeof(f64));
    f64* AFAC = calloc(NMAX * NMAX, sizeof(f64));
    f64* AINV = calloc(NMAX * NMAX, sizeof(f64));
    f64* B = calloc(NMAX * MAXRHS, sizeof(f64));
    f64* BSAV = calloc(NMAX * MAXRHS, sizeof(f64));
    f64* X = calloc(NMAX * MAXRHS, sizeof(f64));
    f64* XACT = calloc(NMAX * MAXRHS, sizeof(f64));
    f64* ARF = calloc(rfp_size + 1, sizeof(f64));
    f64* ARFINV = calloc(rfp_size + 1, sizeof(f64));
    f64* work = calloc(3 * NMAX, sizeof(f64));
    f64* rwork = calloc(NMAX, sizeof(f64));
    f64* temp = calloc(NMAX * NMAX, sizeof(f64));

    dlatb4("DPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    dlatms(n, n, &dist, &type, work, mode, cndnum, anorm, kl, ku,
           &uplo, A, lda, work, &info, rng_state);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c DLATMS info=%d",
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
            for (int i = 0; i < izero; i++) {
                A[izero * lda + i] = 0.0;
            }
            for (int i = izero; i < n; i++) {
                A[i * lda + izero] = 0.0;
            }
        } else {
            for (int i = 0; i < izero; i++) {
                A[i * lda + izero] = 0.0;
            }
            for (int i = izero; i < n; i++) {
                A[izero * lda + i] = 0.0;
            }
        }
    }

    dlacpy(&uplo, n, n, A, lda, ASAV, lda);

    if (zerot) {
        rcondc = 0.0;
    } else {
        f64 norm_a = dlansy("1", &uplo, n, A, lda, rwork);
        dlacpy(&uplo, n, n, A, lda, AFAC, lda);
        dpotrf(&uplo, n, AFAC, lda, &info);
        if (info == 0) {
            dpotri(&uplo, n, AFAC, lda, &info);
            if (info == 0 && n > 0) {
                ainvnm = dlansy("1", &uplo, n, AFAC, lda, rwork);
                rcondc = (1.0 / norm_a) / ainvnm;
            } else {
                rcondc = 0.0;
            }
        } else {
            rcondc = 0.0;
        }
        dlacpy(&uplo, n, n, ASAV, lda, A, lda);
    }

    dlarhs("DPO", "N", &uplo, " ", n, n, kl, ku, nrhs, A, lda,
           XACT, lda, B, lda, &info, rng_state);
    dlacpy("F", n, nrhs, B, lda, BSAV, lda);

    dlacpy(&uplo, n, n, A, lda, AFAC, lda);
    dlacpy("F", n, nrhs, B, ldb, X, ldb);

    dtrttf(&cform, &uplo, n, AFAC, lda, ARF, &info);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c DTRTTF info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    dpftrf(&cform, &uplo, n, ARF, &info);

    if (zerot) {
        if (info != izero + 1) {
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c DPFTRF info=%d expected=%d",
                     n, imat, uplo, cform, info, izero + 1);
            set_test_context(ctx);
        }
        goto cleanup;
    }

    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c DPFTRF info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    dpftrs(&cform, &uplo, n, nrhs, ARF, X, ldb, &info);
    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d uplo=%c form=%c DPFTRS info=%d",
                 n, imat, uplo, cform, info);
        set_test_context(ctx);
        goto cleanup;
    }

    dtfttr(&cform, &uplo, n, ARF, AFAC, lda, &info);

    dlacpy(&uplo, n, n, AFAC, lda, temp, lda);
    dpot01(&uplo, n, A, lda, AFAC, lda, rwork, &result[0]);
    dlacpy(&uplo, n, n, temp, lda, AFAC, lda);

    if ((n % 2) == 0) {
        for (int j = 0; j < n / 2; j++) {
            for (int i = 0; i <= n; i++) {
                ARFINV[j * (n + 1) + i] = ARF[j * (n + 1) + i];
            }
        }
    } else {
        for (int j = 0; j < (n + 1) / 2; j++) {
            for (int i = 0; i < n; i++) {
                ARFINV[j * n + i] = ARF[j * n + i];
            }
        }
    }

    dpftri(&cform, &uplo, n, ARFINV, &info);
    dtfttr(&cform, &uplo, n, ARFINV, AINV, lda, &info);

    dpot03(&uplo, n, A, lda, AINV, lda, temp, lda, rwork, &rcondc, &result[1]);

    dlacpy("F", n, nrhs, BSAV, lda, temp, lda);
    dpot02(&uplo, n, nrhs, A, lda, X, lda, temp, lda, rwork, &result[2]);

    dget04(n, nrhs, X, lda, XACT, lda, rcondc, &result[3]);

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

static void test_dchkrfp_case(void** state)
{
    dchkrfp_params_t* params = *state;
    run_dchkrfp_single(params->n, params->nrhs, params->imat,
                       params->iuplo, params->iform);
}

#define MAX_TESTS (NN * NNS * NTYPES * 2 * 2)

static dchkrfp_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        for (int ins = 0; ins < (int)NNS; ins++) {
            int nrhs = NSVAL[ins];

            for (int imat = 1; imat <= (int)NTYPES; imat++) {

                if (n == 0 && imat > 1) continue;
                if (imat == 4 && n <= 1) continue;
                if (imat == 5 && n <= 2) continue;

                for (int iuplo = 0; iuplo < 2; iuplo++) {
                    for (int iform = 0; iform < 2; iform++) {

                        dchkrfp_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->nrhs = nrhs;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->iform = iform;
                        snprintf(p->name, sizeof(p->name),
                                 "dchkrfp_n%d_nrhs%d_imat%d_%c_%c",
                                 n, nrhs, imat,
                                 (iuplo == 0) ? 'U' : 'L',
                                 (iform == 0) ? 'N' : 'T');

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_dchkrfp_case;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;
                        g_tests[g_num_tests].initial_state = p;

                        g_num_tests++;
                        if (g_num_tests >= (int)MAX_TESTS) return;
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

    return _cmocka_run_group_tests("dchkrfp", g_tests, g_num_tests, NULL, NULL);
}
