/**
 * @file test_schkgg.c
 * @brief Generalized nonsymmetric eigenvalue (Hessenberg) test driver - port of
 *        LAPACK TESTING/EIG/dchkgg.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem routines.
 *                                T          T
 * SGGHRD factors A and B as U H V  and U T V , where T means
 * transpose, H is Hessenberg, T is triangular and U and V are orthogonal.
 *                                 T          T
 * SHGEQZ factors H and T as  Q S Z  and Q P Z , where P is upper
 * triangular, S is in generalized Schur form (block upper triangular,
 * with 1x1 and 2x2 blocks on the diagonal, the 2x2 blocks corresponding
 * to complex conjugate pairs of generalized eigenvalues), and Q and Z are
 * orthogonal.
 *
 * STGEVC computes the matrix L of left eigenvectors and the matrix R
 * of right eigenvectors for the matrix pair (S, P).
 *
 * Test ratios (15 total):
 *                  T
 *  (1)  | A - U H V  | / ( |A| n ulp )
 *                  T
 *  (2)  | B - U T V  | / ( |B| n ulp )
 *               T
 *  (3)  | I - UU  | / ( n ulp )
 *               T
 *  (4)  | I - VV  | / ( n ulp )
 *                  T
 *  (5)  | H - Q S Z  | / ( |H| n ulp )
 *                  T
 *  (6)  | T - Q P Z  | / ( |T| n ulp )
 *               T
 *  (7)  | I - QQ  | / ( n ulp )
 *               T
 *  (8)  | I - ZZ  | / ( n ulp )
 *                          T
 *  (9)  max | ( b S - a P ) l | / const.
 *                          T
 *  (10) max | ( b H - a T ) l | / const.  (backtransformed)
 *
 *  (11) max | ( b S - a P ) r | / const.
 *
 *  (12) max | ( b H - a T ) r | / const.  (backtransformed)
 *
 *  (13) | S1 - Q' S2 Z | / ( |S1| n ulp )  (Schur consistency)
 *
 *  (14) | P1 - Q' P2 Z | / ( |P1| n ulp )  (Schur consistency)
 *
 *  (15) max eigenvalue discrepancy / ulp
 *
 * Matrix types: 26 types from SLATM4 (see dchkgg.f documentation)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 30.0f
#define THRSHN 10.0f
#define MAXTYP 26
#define NTEST  15

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* External function declarations */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void sgeqr2(const int m, const int n, f32* A, const int lda,
                   f32* tau, f32* work, int* info);
extern void sorm2r(const char* side, const char* trans, const int m,
                   const int n, const int k, const f32* A, const int lda,
                   const f32* tau, f32* C, const int ldc, f32* work,
                   int* info);
extern void sgghrd(const char* compq, const char* compz, const int n,
                   const int ilo, const int ihi, f32* A, const int lda,
                   f32* B, const int ldb, f32* Q, const int ldq,
                   f32* Z, const int ldz, int* info);
extern void shgeqz(const char* job, const char* compq, const char* compz,
                   const int n, const int ilo, const int ihi,
                   f32* H, const int ldh, f32* T, const int ldt,
                   f32* alphar, f32* alphai, f32* beta,
                   f32* Q, const int ldq, f32* Z, const int ldz,
                   f32* work, const int lwork, int* info);
extern void stgevc(const char* side, const char* howmny,
                   const int* select, const int n,
                   const f32* S, const int lds, const f32* P, const int ldp,
                   f32* VL, const int ldvl, f32* VR, const int ldvr,
                   const int mm, int* m, f32* work, int* info);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta, f32* A, const int lda);
extern void slarfg(const int n, f32* alpha, f32* x, const int incx, f32* tau);

/* ===================================================================== */
/* DATA arrays from dchkgg.f lines 573-592                               */
/* ===================================================================== */

static const int kclass[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3
};

static const int katype[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4, 4, 4,
    2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
};

static const int kbtype[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4, 1, 1, -4,
    2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
};

static const int kazero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3,
    1, 3, 5, 5, 5, 5, 3, 3, 3, 3, 1
};

static const int kbzero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4,
    1, 4, 6, 6, 6, 6, 4, 4, 4, 4, 1
};

static const int kamagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1
};

static const int kbmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1
};

static const int ktrian[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static const int iasign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2,
    0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0
};

static const int ibsign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2,
    0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const int kz1[6] = {0, 1, 2, 1, 3, 3};
static const int kz2[6] = {0, 0, 1, 2, 1, 1};
static const int kadd[6] = {0, 0, 0, 0, 3, 2};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    int jsize;    /* index into NVAL[] */
    int jtype;    /* matrix type 1..26 */
    char name[96];
} dchkgg_params_t;

typedef struct {
    int nmax;
    f32* A;
    f32* B;
    f32* H;
    f32* T;
    f32* S1;
    f32* P1;
    f32* S2;
    f32* P2;
    f32* U;
    f32* V;
    f32* Q;
    f32* Z;
    f32* evectl;
    f32* evectr;
    f32* alphr1;
    f32* alphi1;
    f32* beta1;
    f32* alphr3;
    f32* alphi3;
    f32* beta3;
    f32* work;
    int* llwork;
    int lwork;
    uint64_t rng_state[4];
} dchkgg_workspace_t;

static dchkgg_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchkgg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

    g_ws->A      = malloc(n2 * sizeof(f32));
    g_ws->B      = malloc(n2 * sizeof(f32));
    g_ws->H      = malloc(n2 * sizeof(f32));
    g_ws->T      = malloc(n2 * sizeof(f32));
    g_ws->S1     = malloc(n2 * sizeof(f32));
    g_ws->P1     = malloc(n2 * sizeof(f32));
    g_ws->S2     = malloc(n2 * sizeof(f32));
    g_ws->P2     = malloc(n2 * sizeof(f32));
    g_ws->U      = malloc(n2 * sizeof(f32));
    g_ws->V      = malloc(n2 * sizeof(f32));
    g_ws->Q      = malloc(n2 * sizeof(f32));
    g_ws->Z      = malloc(n2 * sizeof(f32));
    g_ws->evectl = malloc(n2 * sizeof(f32));
    g_ws->evectr = malloc(n2 * sizeof(f32));

    g_ws->alphr1 = malloc(nmax * sizeof(f32));
    g_ws->alphi1 = malloc(nmax * sizeof(f32));
    g_ws->beta1  = malloc(nmax * sizeof(f32));
    g_ws->alphr3 = malloc(nmax * sizeof(f32));
    g_ws->alphi3 = malloc(nmax * sizeof(f32));
    g_ws->beta3  = malloc(nmax * sizeof(f32));

    int lwk1 = 2 * n2;
    int lwk2 = 6 * nmax;
    g_ws->lwork = (lwk1 > lwk2) ? lwk1 : lwk2;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work   = malloc(g_ws->lwork * sizeof(f32));
    g_ws->llwork = malloc(nmax * sizeof(int));

    if (!g_ws->A || !g_ws->B || !g_ws->H || !g_ws->T ||
        !g_ws->S1 || !g_ws->P1 || !g_ws->S2 || !g_ws->P2 ||
        !g_ws->U || !g_ws->V || !g_ws->Q || !g_ws->Z ||
        !g_ws->evectl || !g_ws->evectr ||
        !g_ws->alphr1 || !g_ws->alphi1 || !g_ws->beta1 ||
        !g_ws->alphr3 || !g_ws->alphi3 || !g_ws->beta3 ||
        !g_ws->work || !g_ws->llwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDCEE6601ULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->H);
        free(g_ws->T);
        free(g_ws->S1);
        free(g_ws->P1);
        free(g_ws->S2);
        free(g_ws->P2);
        free(g_ws->U);
        free(g_ws->V);
        free(g_ws->Q);
        free(g_ws->Z);
        free(g_ws->evectl);
        free(g_ws->evectr);
        free(g_ws->alphr1);
        free(g_ws->alphi1);
        free(g_ws->beta1);
        free(g_ws->alphr3);
        free(g_ws->alphi3);
        free(g_ws->beta3);
        free(g_ws->work);
        free(g_ws->llwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_dchkgg_single(dchkgg_params_t* params)
{
    const int n = NVAL[params->jsize];
    const int jtype = params->jtype;
    const int jt = jtype - 1;

    if (n == 0) {
        /* n=0 is trivially correct */
        return;
    }

    const int lda = g_ws->nmax;
    const int ldu = g_ws->nmax;
    const int lwork = g_ws->lwork;

    f32* A      = g_ws->A;
    f32* B      = g_ws->B;
    f32* H      = g_ws->H;
    f32* T      = g_ws->T;
    f32* S1     = g_ws->S1;
    f32* P1     = g_ws->P1;
    f32* S2     = g_ws->S2;
    f32* P2     = g_ws->P2;
    f32* U      = g_ws->U;
    f32* V      = g_ws->V;
    f32* Q      = g_ws->Q;
    f32* Z      = g_ws->Z;
    f32* evectl = g_ws->evectl;
    f32* evectr = g_ws->evectr;
    f32* alphr1 = g_ws->alphr1;
    f32* alphi1 = g_ws->alphi1;
    f32* beta1  = g_ws->beta1;
    f32* alphr3 = g_ws->alphr3;
    f32* alphi3 = g_ws->alphi3;
    f32* beta3  = g_ws->beta3;
    f32* work   = g_ws->work;
    int* llwork = g_ws->llwork;

    uint64_t* rng = g_ws->rng_state;

    const f32 ulp = slamch("P");
    const f32 safmin = slamch("S") / ulp;
    const f32 safmax = 1.0f / safmin;
    const f32 ulpinv = 1.0f / ulp;

    const int n1 = (n > 1) ? n : 1;

    f32 rmagn[4];
    rmagn[0] = 0.0f;
    rmagn[1] = 1.0f;
    rmagn[2] = safmax * ulp / (f32)n1;
    rmagn[3] = safmin * ulpinv * (f32)n1;

    f32 result[NTEST];
    for (int j = 0; j < NTEST; j++)
        result[j] = 0.0f;

    int iinfo = 0;
    f32 anorm, bnorm;
    char ctx[128];

    /* ================================================================ */
    /* Generate A and B                                                 */
    /* ================================================================ */

    if (kclass[jt] < 3) {
        /* Generate A (w/o rotation) */
        int in = n;
        if (abs(katype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, A, lda);
        }
        slatm4(katype[jt], in, kz1[kazero[jt] - 1], kz2[kazero[jt] - 1],
               iasign[jt], rmagn[kamagn[jt]], ulp,
               rmagn[ktrian[jt] * kamagn[jt]], 2, A, lda, rng);
        int iadd = kadd[kazero[jt] - 1];
        if (iadd > 0 && iadd <= n)
            A[(iadd - 1) + (iadd - 1) * lda] = rmagn[kamagn[jt]];

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(kbtype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, B, lda);
        }
        slatm4(kbtype[jt], in, kz1[kbzero[jt] - 1], kz2[kbzero[jt] - 1],
               ibsign[jt], rmagn[kbmagn[jt]], 1.0f,
               rmagn[ktrian[jt] * kbmagn[jt]], 2, B, lda, rng);
        int iadd_b = kadd[kbzero[jt] - 1];
        if (iadd_b != 0 && iadd_b <= n)
            B[(iadd_b - 1) + (iadd_b - 1) * lda] = rmagn[kbmagn[jt]];

        if (kclass[jt] == 2 && n > 0) {
            /* Include rotations:
             * Generate U, V as Householder transformations times
             * a diagonal matrix. */
            for (int jc = 0; jc < n - 1; jc++) {
                for (int jr = jc; jr < n; jr++) {
                    U[jr + jc * ldu] = rng_normal_f32(rng);
                    V[jr + jc * ldu] = rng_normal_f32(rng);
                }
                slarfg(n - jc, &U[jc + jc * ldu], &U[jc + 1 + jc * ldu], 1,
                       &work[jc]);
                work[2 * n + jc] = (U[jc + jc * ldu] >= 0.0f) ? 1.0f : -1.0f;
                U[jc + jc * ldu] = 1.0f;
                slarfg(n - jc, &V[jc + jc * ldu], &V[jc + 1 + jc * ldu], 1,
                       &work[n + jc]);
                work[3 * n + jc] = (V[jc + jc * ldu] >= 0.0f) ? 1.0f : -1.0f;
                V[jc + jc * ldu] = 1.0f;
            }
            U[(n - 1) + (n - 1) * ldu] = 1.0f;
            work[n - 1] = 0.0f;
            work[3 * n - 1] = (rng_uniform_symmetric_f32(rng) >= 0.0f) ? 1.0f : -1.0f;
            V[(n - 1) + (n - 1) * ldu] = 1.0f;
            work[2 * n - 1] = 0.0f;
            work[4 * n - 1] = (rng_uniform_symmetric_f32(rng) >= 0.0f) ? 1.0f : -1.0f;

            /* Apply the diagonal matrices */
            for (int jc = 0; jc < n; jc++) {
                for (int jr = 0; jr < n; jr++) {
                    A[jr + jc * lda] = work[2 * n + jr] * work[3 * n + jc] *
                                       A[jr + jc * lda];
                    B[jr + jc * lda] = work[2 * n + jr] * work[3 * n + jc] *
                                       B[jr + jc * lda];
                }
            }
            sorm2r("L", "N", n, n, n - 1, U, ldu, work, A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("R", "T", n, n, n - 1, V, ldu, &work[n], A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("L", "N", n, n, n - 1, U, ldu, work, B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("R", "T", n, n, n - 1, V, ldu, &work[n], B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
        }
    } else {
        /* Random matrices */
        for (int jc = 0; jc < n; jc++) {
            for (int jr = 0; jr < n; jr++) {
                A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                   rng_uniform_symmetric_f32(rng);
                B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                   rng_uniform_symmetric_f32(rng);
            }
        }
    }

    anorm = slange("1", n, n, A, lda, work);
    bnorm = slange("1", n, n, B, lda, work);

    goto gen_done;
gen_error:
    snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: generator error info=%d",
             n, jtype, iinfo);
    set_test_context(ctx);
    assert_info_success(iinfo);
    return;
gen_done:

    /* ================================================================ */
    /* Call SGEQR2, SORM2R, and SGGHRD to compute H, T, U, and V       */
    /* ================================================================ */

    slacpy(" ", n, n, A, lda, H, lda);
    slacpy(" ", n, n, B, lda, T, lda);

    result[0] = ulpinv;

    sgeqr2(n, n, T, lda, work, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SGEQR2 info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    sorm2r("L", "T", n, n, n, T, lda, work, H, lda, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SORM2R info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    slaset("Full", n, n, 0.0f, 1.0f, U, ldu);
    sorm2r("R", "N", n, n, n, T, lda, work, U, ldu, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SORM2R info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* 0-based: ilo=0, ihi=n-1 */
    sgghrd("V", "I", n, 0, n - 1, H, lda, T, lda, U, ldu, V, ldu, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SGGHRD info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* ================================================================ */
    /* Tests 1-4: SGGHRD results                                       */
    /* ================================================================ */

    sget51(1, n, A, lda, H, lda, U, ldu, V, ldu, work, &result[0]);
    sget51(1, n, B, lda, T, lda, U, ldu, V, ldu, work, &result[1]);
    sget51(3, n, B, lda, T, lda, U, ldu, U, ldu, work, &result[2]);
    sget51(3, n, B, lda, T, lda, V, ldu, V, ldu, work, &result[3]);

    /* ================================================================ */
    /* Call SHGEQZ to compute S1, P1, S2, P2, Q, and Z                 */
    /* ================================================================ */

    /* Eigenvalues only */
    slacpy(" ", n, n, H, lda, S2, lda);
    slacpy(" ", n, n, T, lda, P2, lda);
    result[4] = ulpinv;

    /* 0-based: ilo=0, ihi=n-1 */
    shgeqz("E", "N", "N", n, 0, n - 1, S2, lda, P2, lda,
           alphr3, alphi3, beta3, Q, ldu, Z, ldu, work, lwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SHGEQZ(E) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues and Full Schur Form */
    slacpy(" ", n, n, H, lda, S2, lda);
    slacpy(" ", n, n, T, lda, P2, lda);

    shgeqz("S", "N", "N", n, 0, n - 1, S2, lda, P2, lda,
           alphr1, alphi1, beta1, Q, ldu, Z, ldu, work, lwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SHGEQZ(S) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues, Schur Form, and Schur Vectors */
    slacpy(" ", n, n, H, lda, S1, lda);
    slacpy(" ", n, n, T, lda, P1, lda);

    shgeqz("S", "I", "I", n, 0, n - 1, S1, lda, P1, lda,
           alphr1, alphi1, beta1, Q, ldu, Z, ldu, work, lwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: SHGEQZ(V) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* ================================================================ */
    /* Tests 5-8: SHGEQZ results                                       */
    /* ================================================================ */

    sget51(1, n, H, lda, S1, lda, Q, ldu, Z, ldu, work, &result[4]);
    sget51(1, n, T, lda, P1, lda, Q, ldu, Z, ldu, work, &result[5]);
    sget51(3, n, T, lda, P1, lda, Q, ldu, Q, ldu, work, &result[6]);
    sget51(3, n, T, lda, P1, lda, Z, ldu, Z, ldu, work, &result[7]);

    /* ================================================================ */
    /* Compute the Left and Right Eigenvectors of (S1, P1)              */
    /* ================================================================ */

    /* 9: Left eigenvectors without back transforming */
    result[8] = ulpinv;

    /* To test "SELECT" option, compute half of the eigenvectors
     * in one call, and half in another */
    int i1 = n / 2;
    for (int j = 0; j < i1; j++)
        llwork[j] = 1;
    for (int j = i1; j < n; j++)
        llwork[j] = 0;

    f32 dumma[4];
    int in_out;

    stgevc("L", "S", llwork, n, S1, lda, P1, lda, evectl, ldu,
           dumma, ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(L,S1) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    i1 = in_out;
    for (int j = 0; j < i1; j++)
        llwork[j] = 0;
    for (int j = i1; j < n; j++)
        llwork[j] = 1;

    stgevc("L", "S", llwork, n, S1, lda, P1, lda,
           &evectl[i1 * ldu], ldu, dumma, ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(L,S2) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    sget52(1, n, S1, lda, P1, lda, evectl, ldu,
           alphr1, alphi1, beta1, work, dumma);
    result[8] = dumma[0];

    /* 10: Left eigenvectors with back transforming */
    result[9] = ulpinv;
    slacpy("F", n, n, Q, ldu, evectl, ldu);
    stgevc("L", "B", llwork, n, S1, lda, P1, lda, evectl, ldu,
           dumma, ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(L,B) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    sget52(1, n, H, lda, T, lda, evectl, ldu,
           alphr1, alphi1, beta1, work, dumma);
    result[9] = dumma[0];

    /* 11: Right eigenvectors without back transforming */
    result[10] = ulpinv;

    i1 = n / 2;
    for (int j = 0; j < i1; j++)
        llwork[j] = 1;
    for (int j = i1; j < n; j++)
        llwork[j] = 0;

    stgevc("R", "S", llwork, n, S1, lda, P1, lda, dumma, ldu,
           evectr, ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(R,S1) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    i1 = in_out;
    for (int j = 0; j < i1; j++)
        llwork[j] = 0;
    for (int j = i1; j < n; j++)
        llwork[j] = 1;

    stgevc("R", "S", llwork, n, S1, lda, P1, lda, dumma, ldu,
           &evectr[i1 * ldu], ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(R,S2) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    sget52(0, n, S1, lda, P1, lda, evectr, ldu,
           alphr1, alphi1, beta1, work, dumma);
    result[10] = dumma[0];

    /* 12: Right eigenvectors with back transforming */
    result[11] = ulpinv;
    slacpy("F", n, n, Z, ldu, evectr, ldu);
    stgevc("R", "B", llwork, n, S1, lda, P1, lda, dumma, ldu,
           evectr, ldu, n, &in_out, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d: STGEVC(R,B) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    sget52(0, n, H, lda, T, lda, evectr, ldu,
           alphr1, alphi1, beta1, work, dumma);
    result[11] = dumma[0];

    /* ================================================================ */
    /* Tests 13--15: Consistency tests                                  */
    /* ================================================================ */

    sget51(2, n, S1, lda, S2, lda, Q, ldu, Z, ldu, work, &result[12]);
    sget51(2, n, P1, lda, P2, lda, Q, ldu, Z, ldu, work, &result[13]);

    /* Test 15: eigenvalue consistency */
    {
        f32 temp1 = 0.0f;
        f32 temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            f32 d1 = fabsf(alphr1[j] - alphr3[j]) +
                     fabsf(alphi1[j] - alphi3[j]);
            if (d1 > temp1) temp1 = d1;
            f32 d2 = fabsf(beta1[j] - beta3[j]);
            if (d2 > temp2) temp2 = d2;
        }

        f32 denom1 = ulp * ((temp1 > anorm) ? temp1 : anorm);
        if (denom1 < safmin) denom1 = safmin;
        temp1 = temp1 / denom1;

        f32 denom2 = ulp * ((temp2 > bnorm) ? temp2 : bnorm);
        if (denom2 < safmin) denom2 = safmin;
        temp2 = temp2 / denom2;

        result[14] = (temp1 > temp2) ? temp1 : temp2;
    }

    /* ================================================================ */
    /* Check results against threshold                                  */
    /* ================================================================ */

    for (int jr = 0; jr < NTEST; jr++) {
        snprintf(ctx, sizeof(ctx), "dchkgg n=%d type=%d TEST %d",
                 n, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_ok(result[jr]);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_dchkgg_case(void** state)
{
    dchkgg_params_t* params = *state;
    run_dchkgg_single(params);
}

/* ===================================================================== */
/* Build test array and main                                             */
/* ===================================================================== */

#define MAX_TESTS (NNVAL * MAXTYP)

static dchkgg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];
        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            dchkgg_params_t* p = &g_params[g_num_tests];
            p->jsize = (int)in;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "dchkgg_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_dchkgg_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("dchkgg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
