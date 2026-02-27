/**
 * @file test_sdrgev.c
 * @brief Generalized nonsymmetric eigenvalue driver test - port of
 *        LAPACK TESTING/EIG/ddrgev.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem driver SGGEV.
 *
 * SGGEV computes for a pair of n-by-n nonsymmetric matrices (A,B) the
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
 * Matrix types: 26 types generated via SLATM4 with optional
 * random orthogonal transforms (SLARFG + SORM2R).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 20.0f
#define MAXTYP 26

/* Test dimensions from dgg.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 16};
#define NSIZES ((int)(sizeof(NVAL) / sizeof(NVAL[0])))
#define NMAX 16   /* max(NVAL) */

typedef struct {
    INT jsize;     /* index into NVAL[] */
    INT jtype;     /* matrix type 1..26 */
    INT test_num;  /* sub-test 1..7 */
    char name[80];
} sdrgev_params_t;

typedef struct {
    f32* A;
    f32* B;
    f32* S;
    f32* T;
    f32* Q;       /* left eigenvectors from V/V call */
    f32* Z;       /* right eigenvectors from V/V call */
    f32* QE;      /* eigenvectors from partial calls */
    f32* alphar;
    f32* alphai;
    f32* beta;
    f32* alphr1;
    f32* alphi1;
    f32* beta1;
    f32* work;
    INT lwork;
} sdrgev_workspace_t;

static sdrgev_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(sdrgev_workspace_t));
    if (!g_ws) return -1;

    const INT lda = NMAX;
    const INT n2 = lda * NMAX;

    g_ws->A      = malloc(n2 * sizeof(f32));
    g_ws->B      = malloc(n2 * sizeof(f32));
    g_ws->S      = malloc(n2 * sizeof(f32));
    g_ws->T      = malloc(n2 * sizeof(f32));
    g_ws->Q      = malloc(n2 * sizeof(f32));
    g_ws->Z      = malloc(n2 * sizeof(f32));
    g_ws->QE     = malloc(n2 * sizeof(f32));
    g_ws->alphar = malloc(NMAX * sizeof(f32));
    g_ws->alphai = malloc(NMAX * sizeof(f32));
    g_ws->beta   = malloc(NMAX * sizeof(f32));
    g_ws->alphr1 = malloc(NMAX * sizeof(f32));
    g_ws->alphi1 = malloc(NMAX * sizeof(f32));
    g_ws->beta1  = malloc(NMAX * sizeof(f32));

    /* Workspace: MAX( 8*N, N*(N+1) ) from the Fortran */
    INT minwrk = 8 * NMAX;
    INT tmp = NMAX * (NMAX + 1);
    if (tmp > minwrk) minwrk = tmp;
    g_ws->lwork = minwrk;
    g_ws->work = malloc(g_ws->lwork * sizeof(f32));

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

/**
 * Generate test matrices A and B for given (jsize, jtype).
 * Returns 0 on success, nonzero on failure.
 */
static INT generate_matrices(INT jsize, INT jtype)
{
    const INT n = NVAL[jsize];
    const INT lda = NMAX;
    const INT ldq = NMAX;
    const INT jt = jtype - 1;

    const f32 safmin = slamch("S");
    const f32 ulp = slamch("P");
    const f32 safmax = 1.0f / safmin;
    const f32 ulpinv = 1.0f / ulp;
    const INT n1 = (n > 1) ? n : 1;

    f32 rmagn[4];
    rmagn[0] = 0.0f;
    rmagn[1] = 1.0f;
    rmagn[2] = safmax * ulp / (f32)n1;
    rmagn[3] = safmin * ulpinv * (f32)n1;

    uint64_t rng_state[4];
    rng_seed(rng_state, (uint64_t)(jsize * 1000 + jtype));

    INT iinfo = 0;

    if (kclass[jt] < 3) {

        /* Generate A (w/o rotation) */
        INT in = n;
        if (abs(katype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, g_ws->A, lda);
        }
        slatm4(katype[jt], in, kz1[kazero[jt] - 1],
                kz2[kazero[jt] - 1], iasign[jt],
                rmagn[kamagn[jt]], ulp,
                rmagn[ktrian[jt] * kamagn[jt]], 2,
                g_ws->A, lda, rng_state);
        INT iadd = kadd[kazero[jt] - 1];
        if (iadd > 0 && iadd <= n)
            g_ws->A[(iadd - 1) + (iadd - 1) * lda] = 1.0f;

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(kbtype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, g_ws->B, lda);
        }
        slatm4(kbtype[jt], in, kz1[kbzero[jt] - 1],
                kz2[kbzero[jt] - 1], ibsign[jt],
                rmagn[kbmagn[jt]], 1.0f,
                rmagn[ktrian[jt] * kbmagn[jt]], 2,
                g_ws->B, lda, rng_state);
        iadd = kadd[kbzero[jt] - 1];
        if (iadd != 0 && iadd <= n)
            g_ws->B[(iadd - 1) + (iadd - 1) * lda] = 1.0f;

        if (kclass[jt] == 2 && n > 0) {
            /*
             * Include rotations
             *
             * Generate Q, Z as Householder transformations times
             * a diagonal matrix.
             */
            for (INT jc = 0; jc < n - 1; jc++) {
                for (INT jr = jc; jr < n; jr++) {
                    g_ws->Q[jr + jc * ldq] = rng_dist_f32(rng_state, 3);
                    g_ws->Z[jr + jc * ldq] = rng_dist_f32(rng_state, 3);
                }
                slarfg(n - jc, &g_ws->Q[jc + jc * ldq],
                       &g_ws->Q[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[jc]);
                g_ws->work[2 * n + jc] = copysignf(1.0f, g_ws->Q[jc + jc * ldq]);
                g_ws->Q[jc + jc * ldq] = 1.0f;
                slarfg(n - jc, &g_ws->Z[jc + jc * ldq],
                       &g_ws->Z[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[n + jc]);
                g_ws->work[3 * n + jc] = copysignf(1.0f, g_ws->Z[jc + jc * ldq]);
                g_ws->Z[jc + jc * ldq] = 1.0f;
            }
            g_ws->Q[(n - 1) + (n - 1) * ldq] = 1.0f;
            g_ws->work[n - 1] = 0.0f;
            g_ws->work[3 * n - 1] = copysignf(1.0f, rng_dist_f32(rng_state, 2));
            g_ws->Z[(n - 1) + (n - 1) * ldq] = 1.0f;
            g_ws->work[2 * n - 1] = 0.0f;
            g_ws->work[4 * n - 1] = copysignf(1.0f, rng_dist_f32(rng_state, 2));

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
            sorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            sorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            sorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            sorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
        }
    } else {
        /* Random matrices (class 3, type 26) */
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr < n; jr++) {
                g_ws->A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist_f32(rng_state, 2);
                g_ws->B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                          rng_dist_f32(rng_state, 2);
            }
        }
    }
    return 0;
}

static void test_sdrgev_case(void** state)
{
    sdrgev_params_t* params = (sdrgev_params_t*)(*state);
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;
    const INT test_num = params->test_num;
    const INT lda = NMAX;
    const INT ldq = NMAX;
    const INT ldqe = NMAX;

    /* Quick return for n=0 */
    if (n == 0) return;

    const f32 ulpinv = 1.0f / slamch("P");

    /* Generate test matrices */
    INT iinfo = generate_matrices(params->jsize, jtype);
    if (iinfo != 0) {
        fail_msg("Generator returned INFO=%lld for N=%lld JTYPE=%lld",
                 (long long)iinfo, (long long)n, (long long)jtype);
        return;
    }

    /* V,V call - needed by all tests */
    slacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
    slacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
    INT iinfo_vv = 0;
    sggev("V", "V", n, g_ws->S, lda, g_ws->T, lda,
          g_ws->alphar, g_ws->alphai, g_ws->beta,
          g_ws->Q, ldq, g_ws->Z, ldq,
          g_ws->work, g_ws->lwork, &iinfo_vv);
    if (iinfo_vv != 0 && iinfo_vv != n + 1) {
        fail_msg("SGGEV(V,V) returned INFO=%lld for N=%lld JTYPE=%lld",
                 (long long)iinfo_vv, (long long)n, (long long)jtype);
        return;
    }

    if (test_num <= 2) {
        /* Tests 1-2: left eigenvector check */
        f32 result[2];
        sget52(1, n, g_ws->A, lda, g_ws->B, lda, g_ws->Q, ldq,
               g_ws->alphar, g_ws->alphai, g_ws->beta,
               g_ws->work, result);
        assert_residual_below(result[test_num - 1], THRESH);

    } else if (test_num <= 4) {
        /* Tests 3-4: right eigenvector check */
        f32 result[2];
        sget52(0, n, g_ws->A, lda, g_ws->B, lda, g_ws->Z, ldq,
               g_ws->alphar, g_ws->alphai, g_ws->beta,
               g_ws->work, result);
        assert_residual_below(result[test_num - 3], THRESH);

    } else if (test_num == 5) {
        /* Test 5: eigenvalue consistency V,V vs N,N */
        slacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
        slacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
        INT iinfo_nn = 0;
        sggev("N", "N", n, g_ws->S, lda, g_ws->T, lda,
              g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
              NULL, ldqe, g_ws->QE, ldqe,
              g_ws->work, g_ws->lwork, &iinfo_nn);
        if (iinfo_nn != 0 && iinfo_nn != n + 1) {
            fail_msg("SGGEV(N,N) returned INFO=%lld for N=%lld JTYPE=%lld",
                     (long long)iinfo_nn, (long long)n, (long long)jtype);
            return;
        }
        INT nmismatch = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j]) {
                nmismatch++;
            }
        }
        if (nmismatch > 0) {
            print_message("SGGEV(V,V) vs (N,N): %lld of %lld eigenvalues "
                          "differ, info(V,V)=%lld info(N,N)=%lld\n",
                          (long long)nmismatch, (long long)n,
                          (long long)iinfo_vv, (long long)iinfo_nn);
            for (INT j = 0; j < n; j++) {
                int ar_eq = (g_ws->alphar[j] == g_ws->alphr1[j]);
                int ai_eq = (g_ws->alphai[j] == g_ws->alphi1[j]);
                int b_eq  = (g_ws->beta[j]   == g_ws->beta1[j]);
                if (!(ar_eq && ai_eq && b_eq)) {
                    print_message("  j=%lld: alphar diff=%.3e  alphai diff=%.3e"
                                  "  beta diff=%.3e\n", (long long)j,
                                  (double)fabsf(g_ws->alphar[j] - g_ws->alphr1[j]),
                                  (double)fabsf(g_ws->alphai[j] - g_ws->alphi1[j]),
                                  (double)fabsf(g_ws->beta[j] - g_ws->beta1[j]));
                }
            }
            assert_residual_below(ulpinv, THRESH);
        }

    } else if (test_num == 6) {
        /* Test 6: left eigenvector consistency V,V vs V,N */
        slacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
        slacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
        INT iinfo_vn = 0;
        sggev("V", "N", n, g_ws->S, lda, g_ws->T, lda,
              g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
              g_ws->QE, ldqe, g_ws->Z, ldq,
              g_ws->work, g_ws->lwork, &iinfo_vn);
        if (iinfo_vn != 0 && iinfo_vn != n + 1) {
            fail_msg("SGGEV(V,N) returned INFO=%lld for N=%lld JTYPE=%lld",
                     (long long)iinfo_vn, (long long)n, (long long)jtype);
            return;
        }
        INT nmismatch_eig = 0, nmismatch_vec = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j])
                nmismatch_eig++;
        }
        for (INT j = 0; j < n; j++) {
            for (INT jc = 0; jc < n; jc++) {
                if (g_ws->Q[j + jc * ldq] != g_ws->QE[j + jc * ldqe])
                    nmismatch_vec++;
            }
        }
        if (nmismatch_eig > 0 || nmismatch_vec > 0) {
            print_message("SGGEV(V,V) vs (V,N): %lld eigenvalue mismatches, "
                          "%lld VL element mismatches\n",
                          (long long)nmismatch_eig, (long long)nmismatch_vec);
            assert_residual_below(ulpinv, THRESH);
        }

    } else {
        /* Test 7: right eigenvector consistency V,V vs N,V */
        slacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
        slacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
        INT iinfo_nv = 0;
        sggev("N", "V", n, g_ws->S, lda, g_ws->T, lda,
              g_ws->alphr1, g_ws->alphi1, g_ws->beta1,
              g_ws->Q, ldq, g_ws->QE, ldqe,
              g_ws->work, g_ws->lwork, &iinfo_nv);
        if (iinfo_nv != 0 && iinfo_nv != n + 1) {
            fail_msg("SGGEV(N,V) returned INFO=%lld for N=%lld JTYPE=%lld",
                     (long long)iinfo_nv, (long long)n, (long long)jtype);
            return;
        }
        INT nmismatch_eig = 0, nmismatch_vec = 0;
        for (INT j = 0; j < n; j++) {
            if (g_ws->alphar[j] != g_ws->alphr1[j] ||
                g_ws->alphai[j] != g_ws->alphi1[j] ||
                g_ws->beta[j]   != g_ws->beta1[j])
                nmismatch_eig++;
        }
        for (INT j = 0; j < n; j++) {
            for (INT jc = 0; jc < n; jc++) {
                if (g_ws->Z[j + jc * ldq] != g_ws->QE[j + jc * ldqe])
                    nmismatch_vec++;
            }
        }
        if (nmismatch_eig > 0 || nmismatch_vec > 0) {
            print_message("SGGEV(V,V) vs (N,V): %lld eigenvalue mismatches, "
                          "%lld VR element mismatches\n",
                          (long long)nmismatch_eig, (long long)nmismatch_vec);
            assert_residual_below(ulpinv, THRESH);
        }
    }
}

int main(void)
{
    static sdrgev_params_t all_params[NSIZES * MAXTYP * 7];
    static struct CMUnitTest all_tests[NSIZES * MAXTYP * 7];
    INT idx = 0;

    for (INT js = 0; js < NSIZES; js++) {
        for (INT jt = 1; jt <= MAXTYP; jt++) {
            for (INT tn = 1; tn <= 7; tn++) {
                sdrgev_params_t* p = &all_params[idx];
                p->jsize = js;
                p->jtype = jt;
                p->test_num = tn;
                snprintf(p->name, sizeof(p->name),
                         "N=%d_type=%d_test%d", NVAL[js], jt, tn);

                all_tests[idx].name = p->name;
                all_tests[idx].test_func = test_sdrgev_case;
                all_tests[idx].setup_func = NULL;
                all_tests[idx].teardown_func = NULL;
                all_tests[idx].initial_state = p;
                idx++;
            }
        }
    }

    return _cmocka_run_group_tests("sdrgev", all_tests, idx,
                                   group_setup, group_teardown);
}
