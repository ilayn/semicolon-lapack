/**
 * @file test_ddrgev.c
 * @brief Generalized nonsymmetric eigenvalue driver test - port of
 *        LAPACK TESTING/EIG/ddrgev.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem driver DGGEV.
 *
 * DGGEV computes for a pair of n-by-n nonsymmetric matrices (A,B) the
 * generalized eigenvalues and, optionally, the left and/or right
 * generalized eigenvectors.
 *
 * Test ratios (7 total):
 *
 * (1)  max over all left eigenvalue/-vector pairs (alpha/beta,l) of
 *      | VL**H * (beta A - alpha B) | / ( ulp max(|beta A|, |alpha B|) )
 *
 * (2)  | |VL(i)| - 1 | / ulp and whether largest component real
 *
 * (3)  max over all right eigenvalue/-vector pairs (alpha/beta,r) of
 *      | (beta A - alpha B) * VR | / ( ulp max(|beta A|, |alpha B|) )
 *
 * (4)  | |VR(i)| - 1 | / ulp and whether largest component real
 *
 * (5)  W(full) = W(partial)
 *      W(full) denotes eigenvalues computed when both l and r are also
 *      computed, and W(partial) denotes eigenvalues computed when only
 *      eigenvalues are computed (JOBVL='N', JOBVR='N').
 *
 * (6)  VL(full) = VL(partial)
 *      VL(full) denotes left eigenvectors computed when both l and r
 *      are computed, and VL(partial) denotes the result when only l
 *      is computed (JOBVL='V', JOBVR='N').
 *
 * (7)  VR(full) = VR(partial)
 *      VR(full) denotes right eigenvectors computed when both l and r
 *      are also computed, and VR(partial) denotes the result when only
 *      r is computed (JOBVL='N', JOBVR='V').
 *
 * Matrix types: 26 types generated via DLATM4 with optional
 * random orthogonal transforms (DLARFG + DORM2R).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 20.0
#define MAXTYP 26

/* Test dimensions from dgg.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 16};
#define NSIZES ((int)(sizeof(NVAL) / sizeof(NVAL[0])))
#define NMAX 16   /* max(NVAL) */

/* External declarations */
/* Parameters for each test case */
typedef struct {
    INT jsize;   /* index into NVAL[] */
    INT jtype;   /* matrix type 1..26 */
    char name[64];
} ddrgev_params_t;

/* Shared workspace */
typedef struct {
    f64* A;
    f64* B;
    f64* S;
    f64* T;
    f64* Q;       /* left eigenvectors from V/V call */
    f64* Z;       /* right eigenvectors from V/V call */
    f64* QE;      /* eigenvectors from partial calls */
    f64* alphar;
    f64* alphai;
    f64* beta;
    f64* alphr1;
    f64* alphi1;
    f64* beta1;
    f64* work;
    INT lwork;
} ddrgev_workspace_t;

static ddrgev_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(ddrgev_workspace_t));
    if (!g_ws) return -1;

    const INT lda = NMAX;
    const INT n2 = lda * NMAX;

    g_ws->A      = malloc(n2 * sizeof(f64));
    g_ws->B      = malloc(n2 * sizeof(f64));
    g_ws->S      = malloc(n2 * sizeof(f64));
    g_ws->T      = malloc(n2 * sizeof(f64));
    g_ws->Q      = malloc(n2 * sizeof(f64));
    g_ws->Z      = malloc(n2 * sizeof(f64));
    g_ws->QE     = malloc(n2 * sizeof(f64));
    g_ws->alphar = malloc(NMAX * sizeof(f64));
    g_ws->alphai = malloc(NMAX * sizeof(f64));
    g_ws->beta   = malloc(NMAX * sizeof(f64));
    g_ws->alphr1 = malloc(NMAX * sizeof(f64));
    g_ws->alphi1 = malloc(NMAX * sizeof(f64));
    g_ws->beta1  = malloc(NMAX * sizeof(f64));

    /* Workspace: MAX( 8*N, N*(N+1) ) from the Fortran */
    INT minwrk = 8 * NMAX;
    INT tmp = NMAX * (NMAX + 1);
    if (tmp > minwrk) minwrk = tmp;
    g_ws->lwork = minwrk;
    g_ws->work = malloc(g_ws->lwork * sizeof(f64));

    if (!g_ws->A || !g_ws->B || !g_ws->S || !g_ws->T ||
        !g_ws->Q || !g_ws->Z || !g_ws->QE ||
        !g_ws->alphar || !g_ws->alphai || !g_ws->beta ||
        !g_ws->alphr1 || !g_ws->alphi1 || !g_ws->beta1 ||
        !g_ws->work) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->S);
        free(g_ws->T);
        free(g_ws->Q);
        free(g_ws->Z);
        free(g_ws->QE);
        free(g_ws->alphar);
        free(g_ws->alphai);
        free(g_ws->beta);
        free(g_ws->alphr1);
        free(g_ws->alphi1);
        free(g_ws->beta1);
        free(g_ws->work);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* DATA arrays from ddrgev.f (converted to 0-based) */
static const INT kclass[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3
};
static const INT kz1[6]  = {0, 1, 2, 1, 3, 3};
static const INT kz2[6]  = {0, 0, 1, 2, 1, 1};
static const INT kadd[6] = {0, 0, 0, 0, 3, 2};
static const INT katype[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4,
    4, 4, 2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
};
static const INT kbtype[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4,
    1, 1, -4, 2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
};
static const INT kazero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 1, 3,
    5, 5, 5, 5, 3, 3, 3, 3, 1
};
static const INT kbzero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4, 1, 4,
    6, 6, 6, 6, 4, 4, 4, 4, 1
};
static const INT kamagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    2, 3, 3, 2, 1
};
static const INT kbmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    3, 2, 3, 2, 1
};
static const INT ktrian[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static const INT iasign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0,
    2, 2, 2, 2, 2, 0
};
static const INT ibsign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0
};

static void test_ddrgev(void** state)
{
    ddrgev_params_t* params = (ddrgev_params_t*)(*state);
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;  /* 1-based */
    const INT lda = NMAX;
    const INT ldq = NMAX;
    const INT ldqe = NMAX;

    /* Quick return for n=0 */
    if (n == 0) return;

    const f64 safmin = dlamch("S");
    const f64 ulp = dlamch("P");
    const f64 safmax = 1.0 / safmin;
    const f64 ulpinv = 1.0 / ulp;

    /* RMAGN(0:3) */
    const INT n1 = (n > 1) ? n : 1;
    f64 rmagn[4];
    rmagn[0] = 0.0;
    rmagn[1] = 1.0;
    rmagn[2] = safmax * ulp / (f64)n1;
    rmagn[3] = safmin * ulpinv * (f64)n1;

    /* RNG state (seeded from size and type for reproducibility) */
    uint64_t rng_state[4];
    rng_seed(rng_state, (uint64_t)(params->jsize * 1000 + jtype));

    f64 result[7];
    for (INT i = 0; i < 7; i++) result[i] = -1.0;

    INT iinfo = 0;
    INT jt = jtype - 1;  /* 0-based index into DATA arrays */

    /* Generate test matrices A and B */
    if (kclass[jt] < 3) {

        /* Generate A (w/o rotation) */
        INT in = n;
        if (abs(katype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                dlaset("Full", n, n, 0.0, 0.0, g_ws->A, lda);
        }
        dlatm4(katype[jt], in, kz1[kazero[jt] - 1],
                kz2[kazero[jt] - 1], iasign[jt],
                rmagn[kamagn[jt]], ulp,
                rmagn[ktrian[jt] * kamagn[jt]], 2,
                g_ws->A, lda, rng_state);
        INT iadd = kadd[kazero[jt] - 1];
        if (iadd > 0 && iadd <= n)
            g_ws->A[(iadd - 1) + (iadd - 1) * lda] = 1.0;

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(kbtype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                dlaset("Full", n, n, 0.0, 0.0, g_ws->B, lda);
        }
        dlatm4(kbtype[jt], in, kz1[kbzero[jt] - 1],
                kz2[kbzero[jt] - 1], ibsign[jt],
                rmagn[kbmagn[jt]], 1.0,
                rmagn[ktrian[jt] * kbmagn[jt]], 2,
                g_ws->B, lda, rng_state);
        iadd = kadd[kbzero[jt] - 1];
        if (iadd != 0 && iadd <= n)
            g_ws->B[(iadd - 1) + (iadd - 1) * lda] = 1.0;

        if (kclass[jt] == 2 && n > 0) {
            /*
             * Include rotations
             *
             * Generate Q, Z as Householder transformations times
             * a diagonal matrix.
             */
            for (INT jc = 0; jc < n - 1; jc++) {
                for (INT jr = jc; jr < n; jr++) {
                    g_ws->Q[jr + jc * ldq] = rng_dist(rng_state, 3);
                    g_ws->Z[jr + jc * ldq] = rng_dist(rng_state, 3);
                }
                dlarfg(n - jc, &g_ws->Q[jc + jc * ldq],
                       &g_ws->Q[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[jc]);
                g_ws->work[2 * n + jc] = copysign(1.0, g_ws->Q[jc + jc * ldq]);
                g_ws->Q[jc + jc * ldq] = 1.0;
                dlarfg(n - jc, &g_ws->Z[jc + jc * ldq],
                       &g_ws->Z[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[n + jc]);
                g_ws->work[3 * n + jc] = copysign(1.0, g_ws->Z[jc + jc * ldq]);
                g_ws->Z[jc + jc * ldq] = 1.0;
            }
            g_ws->Q[(n - 1) + (n - 1) * ldq] = 1.0;
            g_ws->work[n - 1] = 0.0;
            g_ws->work[3 * n - 1] = copysign(1.0, rng_dist(rng_state, 2));
            g_ws->Z[(n - 1) + (n - 1) * ldq] = 1.0;
            g_ws->work[2 * n - 1] = 0.0;
            g_ws->work[4 * n - 1] = copysign(1.0, rng_dist(rng_state, 2));

            /* Apply the diagonal matrices */
            for (INT jc = 0; jc < n; jc++) {
                for (INT jr = 0; jr < n; jr++) {
                    g_ws->A[jr + jc * lda] = g_ws->work[2 * n + jr] *
                                              g_ws->work[3 * n + jc] *
                                              g_ws->A[jr + jc * lda];
                    g_ws->B[jr + jc * lda] = g_ws->work[2 * n + jr] *
                                              g_ws->work[3 * n + jc] *
                                              g_ws->B[jr + jc * lda];
                }
            }
            dorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            dorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            dorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            dorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
        }
    } else {
        /* Random matrices (class 3, type 26) */
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr < n; jr++) {
                g_ws->A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist(rng_state, 2);
                g_ws->B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                          rng_dist(rng_state, 2);
            }
        }
    }

    goto gen_ok;
gen_error:
    print_message("Generator returned INFO=%d for N=%d JTYPE=%d\n",
                  iinfo, n, jtype);
    assert_int_equal(iinfo, 0);
    return;
gen_ok:

    for (INT i = 0; i < 7; i++) result[i] = -1.0;

    /*
     * Call DGGEV to compute eigenvalues and eigenvectors.
     * First call: JOBVL='V', JOBVR='V' (full)
     */
    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    dggev("V", "V", n, g_ws->S, lda, g_ws->T, lda,
          g_ws->alphar, g_ws->alphai, g_ws->beta,
          g_ws->Q, ldq, g_ws->Z, ldq,
          g_ws->work, g_ws->lwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV1 returned INFO=%d for N=%d JTYPE=%d\n",
                      iinfo, n, jtype);
        goto check_results;
    }

    /*
     * Do tests (1) and (2)
     * Left eigenvector check
     */
    dget52(1, n, g_ws->A, lda, g_ws->B, lda, g_ws->Q, ldq,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->work, result);
    if (result[1] > THRESH) {
        print_message("Left eigenvectors from DGGEV1 incorrectly "
                      "normalized. Bits of error=%g, N=%d JTYPE=%d\n",
                      result[1], n, jtype);
    }

    /*
     * Do tests (3) and (4)
     * Right eigenvector check
     */
    dget52(0, n, g_ws->A, lda, g_ws->B, lda, g_ws->Z, ldq,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->work, &result[2]);
    if (result[3] > THRESH) {
        print_message("Right eigenvectors from DGGEV1 incorrectly "
                      "normalized. Bits of error=%g, N=%d JTYPE=%d\n",
                      result[3], n, jtype);
    }

    /*
     * Do test (5)
     * Compare eigenvalues from JOBVL='N', JOBVR='N' with full
     */
    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    dggev("N", "N", n, g_ws->S, lda, g_ws->T, lda,
          g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
          NULL, ldqe, g_ws->QE, ldqe,
          g_ws->work, g_ws->lwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV2 returned INFO=%d for N=%d JTYPE=%d\n",
                      iinfo, n, jtype);
        goto check_results;
    }

    for (INT j = 0; j < n; j++) {
        if (g_ws->alphar[j] != g_ws->alphr1[j] ||
            g_ws->alphai[j] != g_ws->alphi1[j] ||
            g_ws->beta[j]   != g_ws->beta1[j]) {
            result[4] = ulpinv;
        }
    }

    /*
     * Do test (6): Compute eigenvalues and left eigenvectors,
     * and test them
     */
    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    dggev("V", "N", n, g_ws->S, lda, g_ws->T, lda,
          g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
          g_ws->QE, ldqe, g_ws->Z, ldq,
          g_ws->work, g_ws->lwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV3 returned INFO=%d for N=%d JTYPE=%d\n",
                      iinfo, n, jtype);
        goto check_results;
    }

    for (INT j = 0; j < n; j++) {
        if (g_ws->alphar[j] != g_ws->alphr1[j] ||
            g_ws->alphai[j] != g_ws->alphi1[j] ||
            g_ws->beta[j]   != g_ws->beta1[j]) {
            result[5] = ulpinv;
        }
    }

    for (INT j = 0; j < n; j++) {
        for (INT jc = 0; jc < n; jc++) {
            if (g_ws->Q[j + jc * ldq] != g_ws->QE[j + jc * ldqe])
                result[5] = ulpinv;
        }
    }

    /*
     * Do test (7): Compute eigenvalues and right eigenvectors,
     * and test them
     */
    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    dggev("N", "V", n, g_ws->S, lda, g_ws->T, lda,
          g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
          g_ws->Q, ldq, g_ws->QE, ldqe,
          g_ws->work, g_ws->lwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV4 returned INFO=%d for N=%d JTYPE=%d\n",
                      iinfo, n, jtype);
        goto check_results;
    }

    for (INT j = 0; j < n; j++) {
        if (g_ws->alphar[j] != g_ws->alphr1[j] ||
            g_ws->alphai[j] != g_ws->alphi1[j] ||
            g_ws->beta[j]   != g_ws->beta1[j]) {
            result[6] = ulpinv;
        }
    }

    for (INT j = 0; j < n; j++) {
        for (INT jc = 0; jc < n; jc++) {
            if (g_ws->Z[j + jc * ldq] != g_ws->QE[j + jc * ldqe])
                result[6] = ulpinv;
        }
    }

check_results:
    ;
    /* Check results against threshold */
    INT any_fail = 0;
    for (INT jr = 0; jr < 7; jr++) {
        if (result[jr] >= THRESH) {
            print_message("N=%d JTYPE=%d test(%d)=%g\n",
                          n, jtype, jr + 1, result[jr]);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

int main(void)
{
    /* Total: NSIZES * MAXTYP test cases (7 * 26 = 182) */
    static ddrgev_params_t all_params[NSIZES * MAXTYP];
    static struct CMUnitTest all_tests[NSIZES * MAXTYP];
    INT idx = 0;

    for (INT js = 0; js < NSIZES; js++) {
        for (INT jt = 1; jt <= MAXTYP; jt++) {
            ddrgev_params_t* p = &all_params[idx];
            p->jsize = js;
            p->jtype = jt;
            snprintf(p->name, sizeof(p->name),
                     "N=%d_type=%d", NVAL[js], jt);

            all_tests[idx].name = p->name;
            all_tests[idx].test_func = test_ddrgev;
            all_tests[idx].setup_func = NULL;
            all_tests[idx].teardown_func = NULL;
            all_tests[idx].initial_state = p;
            idx++;
        }
    }

    return cmocka_run_group_tests_name("ddrgev", all_tests, group_setup,
                                       group_teardown);
}
