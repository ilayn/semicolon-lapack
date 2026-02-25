/**
 * @file test_ddrgev3.c
 * @brief Generalized nonsymmetric eigenvalue driver test - port of
 *        LAPACK TESTING/EIG/ddrgev3.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem driver DGGEV3.
 *
 * DGGEV3 computes for a pair of n-by-n nonsymmetric matrices (A,B) the
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
 * Matrix types: 27 types generated via DLATM4 with optional
 * random orthogonal transforms (DLARFG + DORM2R).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 20.0
#define MAXTYP 27

/* Test dimensions from dgg.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 16};
#define NSIZES ((int)(sizeof(NVAL) / sizeof(NVAL[0])))
#define NMAX 16   /* max(NVAL) */

/* External declarations */
/* Parameters for each test case */
typedef struct {
    INT jsize;   /* index into NVAL[] */
    INT jtype;   /* matrix type 1..27 */
    char name[64];
} ddrgev3_params_t;

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
} ddrgev3_workspace_t;

static ddrgev3_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(ddrgev3_workspace_t));
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

/* DATA arrays from ddrgev3.f (converted to 0-based) */
static const INT kclass[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 4
};
static const INT kz1[6]  = {0, 1, 2, 1, 3, 3};
static const INT kz2[6]  = {0, 0, 1, 2, 1, 1};
static const INT kadd[6] = {0, 0, 0, 0, 3, 2};
static const INT katype[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4,
    4, 4, 2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0, 0
};
static const INT kbtype[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4,
    1, 1, -4, 2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0
};
static const INT kazero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 1, 3,
    5, 5, 5, 5, 3, 3, 3, 3, 1, 1
};
static const INT kbzero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4, 1, 4,
    6, 6, 6, 6, 4, 4, 4, 4, 1, 1
};
static const INT kamagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    2, 3, 3, 2, 1, 3
};
static const INT kbmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    3, 2, 3, 2, 1, 3
};
static const INT ktrian[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static const INT iasign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0,
    2, 2, 2, 2, 2, 0, 0
};
static const INT ibsign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0
};

static void test_ddrgev3(void** state)
{
    ddrgev3_params_t* params = (ddrgev3_params_t*)(*state);
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
    } else if (kclass[jt] == 3) {
        /* Random matrices (class 3, type 26) */
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr < n; jr++) {
                g_ws->A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist(rng_state, 2);
                g_ws->B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                          rng_dist(rng_state, 2);
            }
        }
    } else {
        /* Random upper Hessenberg pencil with singular B (class 4, type 27) */
        for (INT jc = 0; jc < n; jc++) {
            INT jrmax = jc + 1;
            if (jrmax > n - 1) jrmax = n - 1;
            for (INT jr = 0; jr <= jrmax; jr++) {
                g_ws->A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist(rng_state, 2);
            }
            for (INT jr = jc + 2; jr < n; jr++) {
                g_ws->A[jr + jc * lda] = 0.0;
            }
        }
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr <= jc; jr++) {
                g_ws->B[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist(rng_state, 2);
            }
            for (INT jr = jc + 1; jr < n; jr++) {
                g_ws->B[jr + jc * lda] = 0.0;
            }
        }
        for (INT jc = 0; jc < n; jc += 4) {
            g_ws->B[jc + jc * lda] = 0.0;
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

    INT any_mismatch = 0;

    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    INT iinfo_vv = 0;
    dggev3("V", "V", n, g_ws->S, lda, g_ws->T, lda,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->Q, ldq, g_ws->Z, ldq,
           g_ws->work, g_ws->lwork, &iinfo_vv);
    if (iinfo_vv != 0 && iinfo_vv != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV3(V,V) returned INFO=%lld for N=%lld JTYPE=%lld\n",
                      (long long)iinfo_vv, (long long)n, (long long)jtype);
        goto check_results;
    }

    dget52(1, n, g_ws->A, lda, g_ws->B, lda, g_ws->Q, ldq,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->work, result);
    if (result[1] > THRESH) {
        print_message("Left eigenvectors from DGGEV3(V,V) incorrectly "
                      "normalized. Bits of error=%g, N=%lld JTYPE=%lld\n",
                      result[1], (long long)n, (long long)jtype);
    }

    dget52(0, n, g_ws->A, lda, g_ws->B, lda, g_ws->Z, ldq,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->work, &result[2]);
    if (result[3] > THRESH) {
        print_message("Right eigenvectors from DGGEV3(V,V) incorrectly "
                      "normalized. Bits of error=%g, N=%lld JTYPE=%lld\n",
                      result[3], (long long)n, (long long)jtype);
    }

    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    INT iinfo_nn = 0;
    dggev3("N", "N", n, g_ws->S, lda, g_ws->T, lda,
           g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
           NULL, ldqe, g_ws->QE, ldqe,
           g_ws->work, g_ws->lwork, &iinfo_nn);
    if (iinfo_nn != 0 && iinfo_nn != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV3(N,N) returned INFO=%lld for N=%lld JTYPE=%lld\n",
                      (long long)iinfo_nn, (long long)n, (long long)jtype);
        goto check_results;
    }

    {
        INT nmismatch5 = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j]) {
                nmismatch5++;
            }
        }
        if (nmismatch5 > 0) {
            result[4] = ulpinv;
            any_mismatch = 1;
            print_message("\n=== ddrgev3 test(5) MISMATCH N=%lld JTYPE=%lld "
                          "(%lld of %lld eigenvalues differ) ===\n",
                          (long long)n, (long long)jtype,
                          (long long)nmismatch5, (long long)n);
            print_message("  DGGEV3(V,V) info=%lld   DGGEV3(N,N) info=%lld\n",
                          (long long)iinfo_vv, (long long)iinfo_nn);
            for (INT j = 0; j < n; j++) {
                int ar_eq = (g_ws->alphar[j] == g_ws->alphr1[j]);
                int ai_eq = (g_ws->alphai[j] == g_ws->alphi1[j]);
                int b_eq  = (g_ws->beta[j]   == g_ws->beta1[j]);
                print_message("  j=%2lld: %s\n", (long long)j,
                              (ar_eq && ai_eq && b_eq) ? "MATCH" : "DIFF");
                print_message("    alphar(V,V)=%+.17e  (N,N)=%+.17e  diff=%.3e  %s\n",
                              g_ws->alphar[j], g_ws->alphr1[j],
                              fabs(g_ws->alphar[j] - g_ws->alphr1[j]),
                              ar_eq ? "" : "***");
                print_message("    alphai(V,V)=%+.17e  (N,N)=%+.17e  diff=%.3e  %s\n",
                              g_ws->alphai[j], g_ws->alphi1[j],
                              fabs(g_ws->alphai[j] - g_ws->alphi1[j]),
                              ai_eq ? "" : "***");
                print_message("    beta  (V,V)=%+.17e  (N,N)=%+.17e  diff=%.3e  %s\n",
                              g_ws->beta[j], g_ws->beta1[j],
                              fabs(g_ws->beta[j] - g_ws->beta1[j]),
                              b_eq ? "" : "***");
            }
        }
    }

    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    INT iinfo_vn = 0;
    dggev3("V", "N", n, g_ws->S, lda, g_ws->T, lda,
           g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
           g_ws->QE, ldqe, g_ws->Z, ldq,
           g_ws->work, g_ws->lwork, &iinfo_vn);
    if (iinfo_vn != 0 && iinfo_vn != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV3(V,N) returned INFO=%lld for N=%lld JTYPE=%lld\n",
                      (long long)iinfo_vn, (long long)n, (long long)jtype);
        goto check_results;
    }

    {
        INT nmismatch6_eig = 0, nmismatch6_vec = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j]) {
                nmismatch6_eig++;
                result[5] = ulpinv;
            }
        }
        for (INT j = 0; j < n; j++) {
            for (INT jc = 0; jc < n; jc++) {
                if (g_ws->Q[j + jc * ldq] != g_ws->QE[j + jc * ldqe]) {
                    nmismatch6_vec++;
                    result[5] = ulpinv;
                }
            }
        }
        if (nmismatch6_eig > 0 || nmismatch6_vec > 0) {
            any_mismatch = 1;
            print_message("  test(6) DGGEV3(V,N) info=%lld: "
                          "%lld eigenvalue mismatches, %lld VL element mismatches\n",
                          (long long)iinfo_vn,
                          (long long)nmismatch6_eig, (long long)nmismatch6_vec);
        }
    }

    dlacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    dlacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    INT iinfo_nv = 0;
    dggev3("N", "V", n, g_ws->S, lda, g_ws->T, lda,
           g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
           g_ws->Q, ldq, g_ws->QE, ldqe,
           g_ws->work, g_ws->lwork, &iinfo_nv);
    if (iinfo_nv != 0 && iinfo_nv != n + 1) {
        result[0] = ulpinv;
        print_message("DGGEV3(N,V) returned INFO=%lld for N=%lld JTYPE=%lld\n",
                      (long long)iinfo_nv, (long long)n, (long long)jtype);
        goto check_results;
    }

    {
        INT nmismatch7_eig = 0, nmismatch7_vec = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j]) {
                nmismatch7_eig++;
                result[6] = ulpinv;
            }
        }
        for (INT j = 0; j < n; j++) {
            for (INT jc = 0; jc < n; jc++) {
                if (g_ws->Z[j + jc * ldq] != g_ws->QE[j + jc * ldqe]) {
                    nmismatch7_vec++;
                    result[6] = ulpinv;
                }
            }
        }
        if (nmismatch7_eig > 0 || nmismatch7_vec > 0) {
            any_mismatch = 1;
            print_message("  test(7) DGGEV3(N,V) info=%lld: "
                          "%lld eigenvalue mismatches, %lld VR element mismatches\n",
                          (long long)iinfo_nv,
                          (long long)nmismatch7_eig, (long long)nmismatch7_vec);
        }
    }

    if (any_mismatch) {
        print_message("  result[0..6] = %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
                      result[0], result[1], result[2], result[3],
                      result[4], result[5], result[6]);
    }

check_results:
    for (INT jr = 0; jr < 7; jr++) {
        if (result[jr] >= 0.0) {
            char ctx[128];
            snprintf(ctx, sizeof(ctx), "ddrgev3 N=%lld JTYPE=%lld test(%lld)",
                     (long long)n, (long long)jtype, (long long)(jr + 1));
            set_test_context(ctx);
            assert_residual_below(result[jr], THRESH);
        }
    }
    clear_test_context();
}

int main(void)
{
    /* Total: NSIZES * MAXTYP test cases (7 * 27 = 189) */
    static ddrgev3_params_t all_params[NSIZES * MAXTYP];
    static struct CMUnitTest all_tests[NSIZES * MAXTYP];
    INT idx = 0;

    for (INT js = 0; js < NSIZES; js++) {
        for (INT jt = 1; jt <= MAXTYP; jt++) {
            ddrgev3_params_t* p = &all_params[idx];
            p->jsize = js;
            p->jtype = jt;
            snprintf(p->name, sizeof(p->name),
                     "N=%d_type=%d", NVAL[js], jt);

            all_tests[idx].name = p->name;
            all_tests[idx].test_func = test_ddrgev3;
            all_tests[idx].setup_func = NULL;
            all_tests[idx].teardown_func = NULL;
            all_tests[idx].initial_state = p;
            idx++;
        }
    }

    return cmocka_run_group_tests_name("ddrgev3", all_tests, group_setup,
                                       group_teardown);
}
