/**
 * @file test_cchkgg.c
 * @brief Generalized nonsymmetric eigenvalue (Hessenberg) test driver - port of
 *        LAPACK TESTING/EIG/zchkgg.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem routines.
 *                                H          H        H
 * CGGHRD factors A and B as U H V  and U T V , where   means conjugate
 * transpose, H is Hessenberg, T is triangular and U and V are unitary.
 *                                 H          H
 * CHGEQZ factors H and T as  Q S Z  and Q P Z , where P and S are upper
 * triangular and Q and Z are unitary.
 *
 * CTGEVC computes the matrix L of left eigenvectors and the matrix R
 * of right eigenvectors for the matrix pair (S, P).
 *
 * Test ratios (15 total):
 *                  H
 *  (1)  | A - U H V  | / ( |A| n ulp )
 *                  H
 *  (2)  | B - U T V  | / ( |B| n ulp )
 *               H
 *  (3)  | I - UU  | / ( n ulp )
 *               H
 *  (4)  | I - VV  | / ( n ulp )
 *                  H
 *  (5)  | H - Q S Z  | / ( |H| n ulp )
 *                  H
 *  (6)  | T - Q P Z  | / ( |T| n ulp )
 *               H
 *  (7)  | I - QQ  | / ( n ulp )
 *               H
 *  (8)  | I - ZZ  | / ( n ulp )
 *                          H
 *  (9)  max | ( b S - a P ) l | / const.
 *                          H
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
 * Matrix types: 26 types from CLATM4 (see zchkgg.f documentation)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f
#define THRSHN 10.0f
#define MAXTYP 26
#define NTEST  15

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* ===================================================================== */
/* DATA arrays from zchkgg.f lines 571-593                               */
/* ===================================================================== */

static const INT kclass[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3
};

static const INT katype[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4, 4, 4,
    2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
};

static const INT kbtype[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4, 1, 1, -4,
    2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
};

static const INT kazero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3,
    1, 3, 5, 5, 5, 5, 3, 3, 3, 3, 1
};

static const INT kbzero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4,
    1, 4, 6, 6, 6, 6, 4, 4, 4, 4, 1
};

static const INT kamagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1
};

static const INT kbmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1
};

static const INT ktrian[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static const INT lasign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0
};

static const INT lbsign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const INT kz1[6] = {0, 1, 2, 1, 3, 3};
static const INT kz2[6] = {0, 0, 1, 2, 1, 1};
static const INT kadd[6] = {0, 0, 0, 0, 3, 2};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT jsize;    /* index into NVAL[] */
    INT jtype;    /* matrix type 1..26 */
    char name[96];
} zchkgg_params_t;

typedef struct {
    INT nmax;
    c64* A;
    c64* B;
    c64* H;
    c64* T;
    c64* S1;
    c64* P1;
    c64* S2;
    c64* P2;
    c64* U;
    c64* V;
    c64* Q;
    c64* Z;
    c64* evectl;
    c64* evectr;
    c64* alpha1;
    c64* beta1;
    c64* alpha3;
    c64* beta3;
    c64* work;
    f32*  rwork;
    INT*  llwork;
    INT   lwork;
    uint64_t rng_state[4];
} zchkgg_workspace_t;

static zchkgg_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(zchkgg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A      = malloc(n2 * sizeof(c64));
    g_ws->B      = malloc(n2 * sizeof(c64));
    g_ws->H      = malloc(n2 * sizeof(c64));
    g_ws->T      = malloc(n2 * sizeof(c64));
    g_ws->S1     = malloc(n2 * sizeof(c64));
    g_ws->P1     = malloc(n2 * sizeof(c64));
    g_ws->S2     = malloc(n2 * sizeof(c64));
    g_ws->P2     = malloc(n2 * sizeof(c64));
    g_ws->U      = malloc(n2 * sizeof(c64));
    g_ws->V      = malloc(n2 * sizeof(c64));
    g_ws->Q      = malloc(n2 * sizeof(c64));
    g_ws->Z      = malloc(n2 * sizeof(c64));
    g_ws->evectl = malloc(n2 * sizeof(c64));
    g_ws->evectr = malloc(n2 * sizeof(c64));

    g_ws->alpha1 = malloc(nmax * sizeof(c64));
    g_ws->beta1  = malloc(nmax * sizeof(c64));
    g_ws->alpha3 = malloc(nmax * sizeof(c64));
    g_ws->beta3  = malloc(nmax * sizeof(c64));

    INT lwk1 = 2 * n2;
    INT lwk2 = 4 * nmax;
    g_ws->lwork = (lwk1 > lwk2) ? lwk1 : lwk2;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work   = malloc(g_ws->lwork * sizeof(c64));
    g_ws->rwork  = malloc(2 * nmax * sizeof(f32));
    g_ws->llwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->B || !g_ws->H || !g_ws->T ||
        !g_ws->S1 || !g_ws->P1 || !g_ws->S2 || !g_ws->P2 ||
        !g_ws->U || !g_ws->V || !g_ws->Q || !g_ws->Z ||
        !g_ws->evectl || !g_ws->evectr ||
        !g_ws->alpha1 || !g_ws->beta1 ||
        !g_ws->alpha3 || !g_ws->beta3 ||
        !g_ws->work || !g_ws->rwork || !g_ws->llwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0x2CEE6601ULL);
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
        free(g_ws->alpha1);
        free(g_ws->beta1);
        free(g_ws->alpha3);
        free(g_ws->beta3);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->llwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_zchkgg_single(zchkgg_params_t* params)
{
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    if (n == 0) {
        return;
    }

    const INT lda = g_ws->nmax;
    const INT ldu = g_ws->nmax;
    const INT lwork = g_ws->lwork;

    c64* A      = g_ws->A;
    c64* B      = g_ws->B;
    c64* H      = g_ws->H;
    c64* T      = g_ws->T;
    c64* S1     = g_ws->S1;
    c64* P1     = g_ws->P1;
    c64* S2     = g_ws->S2;
    c64* P2     = g_ws->P2;
    c64* U      = g_ws->U;
    c64* V      = g_ws->V;
    c64* Q      = g_ws->Q;
    c64* Z      = g_ws->Z;
    c64* evectl = g_ws->evectl;
    c64* evectr = g_ws->evectr;
    c64* alpha1 = g_ws->alpha1;
    c64* beta1  = g_ws->beta1;
    c64* alpha3 = g_ws->alpha3;
    c64* beta3  = g_ws->beta3;
    c64* work   = g_ws->work;
    f32*  rwork  = g_ws->rwork;
    INT*  llwork = g_ws->llwork;

    uint64_t* rng = g_ws->rng_state;

    const f32 ulp = slamch("P");
    const f32 safmin = slamch("S") / ulp;
    const f32 safmax = 1.0f / safmin;
    const f32 ulpinv = 1.0f / ulp;

    const INT n1 = (n > 1) ? n : 1;

    f32 rmagn[4];
    rmagn[0] = 0.0f;
    rmagn[1] = 1.0f;
    rmagn[2] = safmax * ulp / (f32)n1;
    rmagn[3] = safmin * ulpinv * (f32)n1;

    f32 result[NTEST];
    for (INT j = 0; j < NTEST; j++)
        result[j] = 0.0f;

    INT iinfo = 0;
    f32 anorm, bnorm;
    char ctx[128];

    /* ================================================================ */
    /* Generate A and B                                                 */
    /* ================================================================ */

    if (kclass[jt] < 3) {
        /* Generate A (w/o rotation) */
        INT in = n;
        if (abs(katype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                claset("Full", n, n, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), A, lda);
        }
        clatm4(katype[jt], in, kz1[kazero[jt] - 1], kz2[kazero[jt] - 1],
               lasign[jt], rmagn[kamagn[jt]], ulp,
               rmagn[ktrian[jt] * kamagn[jt]], 4, A, lda, rng);
        INT iadd = kadd[kazero[jt] - 1];
        if (iadd > 0 && iadd <= n)
            A[(iadd - 1) + (iadd - 1) * lda] = CMPLXF(rmagn[kamagn[jt]], 0.0f);

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(kbtype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                claset("Full", n, n, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), B, lda);
        }
        clatm4(kbtype[jt], in, kz1[kbzero[jt] - 1], kz2[kbzero[jt] - 1],
               lbsign[jt], rmagn[kbmagn[jt]], 1.0f,
               rmagn[ktrian[jt] * kbmagn[jt]], 4, B, lda, rng);
        INT iadd_b = kadd[kbzero[jt] - 1];
        if (iadd_b != 0 && iadd_b <= n)
            B[(iadd_b - 1) + (iadd_b - 1) * lda] = CMPLXF(rmagn[kbmagn[jt]], 0.0f);

        if (kclass[jt] == 2 && n > 0) {
            /* Include rotations:
             * Generate U, V as Householder transformations times
             * a diagonal matrix.  (Note that CLARFG makes U(j,j) and
             * V(j,j) real.) */
            for (INT jc = 0; jc < n - 1; jc++) {
                for (INT jr = jc; jr < n; jr++) {
                    U[jr + jc * ldu] = clarnd_rng(3, rng);
                    V[jr + jc * ldu] = clarnd_rng(3, rng);
                }
                clarfg(n - jc, &U[jc + jc * ldu], &U[jc + 1 + jc * ldu], 1,
                       &work[jc]);
                work[2 * n + jc] = CMPLXF(copysignf(1.0f, crealf(U[jc + jc * ldu])), 0.0f);
                U[jc + jc * ldu] = CMPLXF(1.0f, 0.0f);
                clarfg(n - jc, &V[jc + jc * ldu], &V[jc + 1 + jc * ldu], 1,
                       &work[n + jc]);
                work[3 * n + jc] = CMPLXF(copysignf(1.0f, crealf(V[jc + jc * ldu])), 0.0f);
                V[jc + jc * ldu] = CMPLXF(1.0f, 0.0f);
            }
            c64 ctemp = clarnd_rng(3, rng);
            U[(n - 1) + (n - 1) * ldu] = CMPLXF(1.0f, 0.0f);
            work[n - 1] = CMPLXF(0.0f, 0.0f);
            work[3 * n - 1] = ctemp / cabsf(ctemp);
            ctemp = clarnd_rng(3, rng);
            V[(n - 1) + (n - 1) * ldu] = CMPLXF(1.0f, 0.0f);
            work[2 * n - 1] = CMPLXF(0.0f, 0.0f);
            work[4 * n - 1] = ctemp / cabsf(ctemp);

            /* Apply the diagonal matrices */
            for (INT jc = 0; jc < n; jc++) {
                for (INT jr = 0; jr < n; jr++) {
                    A[jr + jc * lda] = work[2 * n + jr] *
                                       conjf(work[3 * n + jc]) *
                                       A[jr + jc * lda];
                    B[jr + jc * lda] = work[2 * n + jr] *
                                       conjf(work[3 * n + jc]) *
                                       B[jr + jc * lda];
                }
            }
            cunm2r("L", "N", n, n, n - 1, U, ldu, work, A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            cunm2r("R", "C", n, n, n - 1, V, ldu, &work[n], A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            cunm2r("L", "N", n, n, n - 1, U, ldu, work, B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            cunm2r("R", "C", n, n, n - 1, V, ldu, &work[n], B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
        }
    } else {
        /* Random matrices */
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr < n; jr++) {
                A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                   clarnd_rng(4, rng);
                B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                   clarnd_rng(4, rng);
            }
        }
    }

    anorm = clange("1", n, n, A, lda, rwork);
    bnorm = clange("1", n, n, B, lda, rwork);

    goto gen_done;
gen_error:
    snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: generator error info=%d",
             n, jtype, iinfo);
    set_test_context(ctx);
    assert_info_success(iinfo);
    return;
gen_done:

    /* ================================================================ */
    /* Call CGEQR2, CUNM2R, and CGGHRD to compute H, T, U, and V       */
    /* ================================================================ */

    clacpy(" ", n, n, A, lda, H, lda);
    clacpy(" ", n, n, B, lda, T, lda);

    result[0] = ulpinv;

    cgeqr2(n, n, T, lda, work, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CGEQR2 info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    cunm2r("L", "C", n, n, n, T, lda, work, H, lda, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CUNM2R info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    claset("Full", n, n, CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), U, ldu);
    cunm2r("R", "N", n, n, n, T, lda, work, U, ldu, &work[n], &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CUNM2R info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* 0-based: ilo=0, ihi=n-1 */
    cgghrd("V", "I", n, 0, n - 1, H, lda, T, lda, U, ldu, V, ldu, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CGGHRD info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* ================================================================ */
    /* Tests 1-4: CGGHRD results                                       */
    /* ================================================================ */

    cget51(1, n, A, lda, H, lda, U, ldu, V, ldu, work, rwork, &result[0]);
    cget51(1, n, B, lda, T, lda, U, ldu, V, ldu, work, rwork, &result[1]);
    cget51(3, n, B, lda, T, lda, U, ldu, U, ldu, work, rwork, &result[2]);
    cget51(3, n, B, lda, T, lda, V, ldu, V, ldu, work, rwork, &result[3]);

    /* ================================================================ */
    /* Call CHGEQZ to compute S1, P1, S2, P2, Q, and Z                 */
    /* ================================================================ */

    /* Eigenvalues only */
    clacpy(" ", n, n, H, lda, S2, lda);
    clacpy(" ", n, n, T, lda, P2, lda);
    result[4] = ulpinv;

    /* 0-based: ilo=0, ihi=n-1 */
    chgeqz("E", "N", "N", n, 0, n - 1, S2, lda, P2, lda,
           alpha3, beta3, Q, ldu, Z, ldu, work, lwork, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CHGEQZ(E) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues and Full Schur Form */
    clacpy(" ", n, n, H, lda, S2, lda);
    clacpy(" ", n, n, T, lda, P2, lda);

    chgeqz("S", "N", "N", n, 0, n - 1, S2, lda, P2, lda,
           alpha1, beta1, Q, ldu, Z, ldu, work, lwork, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CHGEQZ(S) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues, Schur Form, and Schur Vectors */
    clacpy(" ", n, n, H, lda, S1, lda);
    clacpy(" ", n, n, T, lda, P1, lda);

    chgeqz("S", "I", "I", n, 0, n - 1, S1, lda, P1, lda,
           alpha1, beta1, Q, ldu, Z, ldu, work, lwork, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CHGEQZ(V) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* ================================================================ */
    /* Tests 5-8: CHGEQZ results                                       */
    /* ================================================================ */

    cget51(1, n, H, lda, S1, lda, Q, ldu, Z, ldu, work, rwork, &result[4]);
    cget51(1, n, T, lda, P1, lda, Q, ldu, Z, ldu, work, rwork, &result[5]);
    cget51(3, n, T, lda, P1, lda, Q, ldu, Q, ldu, work, rwork, &result[6]);
    cget51(3, n, T, lda, P1, lda, Z, ldu, Z, ldu, work, rwork, &result[7]);

    /* ================================================================ */
    /* Compute the Left and Right Eigenvectors of (S1, P1)              */
    /* ================================================================ */

    /* 9: Left eigenvectors without back transforming */
    result[8] = ulpinv;

    /* To test "SELECT" option, compute half of the eigenvectors
     * in one call, and half in another */
    INT i1 = n / 2;
    for (INT j = 0; j < i1; j++)
        llwork[j] = 1;
    for (INT j = i1; j < n; j++)
        llwork[j] = 0;

    f32 dumma[4];
    c64 cdumma[4];
    INT in_out;

    ctgevc("L", "S", llwork, n, S1, lda, P1, lda, evectl, ldu,
           cdumma, ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(L,S1) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    i1 = in_out;
    for (INT j = 0; j < i1; j++)
        llwork[j] = 0;
    for (INT j = i1; j < n; j++)
        llwork[j] = 1;

    ctgevc("L", "S", llwork, n, S1, lda, P1, lda,
           &evectl[i1 * ldu], ldu, cdumma, ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(L,S2) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    cget52(1, n, S1, lda, P1, lda, evectl, ldu,
           alpha1, beta1, work, rwork, dumma);
    result[8] = dumma[0];

    /* 10: Left eigenvectors with back transforming */
    result[9] = ulpinv;
    clacpy("F", n, n, Q, ldu, evectl, ldu);
    ctgevc("L", "B", llwork, n, S1, lda, P1, lda, evectl, ldu,
           cdumma, ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(L,B) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    cget52(1, n, H, lda, T, lda, evectl, ldu, alpha1, beta1,
           work, rwork, dumma);
    result[9] = dumma[0];

    /* 11: Right eigenvectors without back transforming */
    result[10] = ulpinv;

    i1 = n / 2;
    for (INT j = 0; j < i1; j++)
        llwork[j] = 1;
    for (INT j = i1; j < n; j++)
        llwork[j] = 0;

    ctgevc("R", "S", llwork, n, S1, lda, P1, lda, cdumma, ldu,
           evectr, ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(R,S1) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    i1 = in_out;
    for (INT j = 0; j < i1; j++)
        llwork[j] = 0;
    for (INT j = i1; j < n; j++)
        llwork[j] = 1;

    ctgevc("R", "S", llwork, n, S1, lda, P1, lda, cdumma, ldu,
           &evectr[i1 * ldu], ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(R,S2) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    cget52(0, n, S1, lda, P1, lda, evectr, ldu,
           alpha1, beta1, work, rwork, dumma);
    result[10] = dumma[0];

    /* 12: Right eigenvectors with back transforming */
    result[11] = ulpinv;
    clacpy("F", n, n, Z, ldu, evectr, ldu);
    ctgevc("R", "B", llwork, n, S1, lda, P1, lda, cdumma, ldu,
           evectr, ldu, n, &in_out, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d: CTGEVC(R,B) info=%d",
                 n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    cget52(0, n, H, lda, T, lda, evectr, ldu,
           alpha1, beta1, work, rwork, dumma);
    result[11] = dumma[0];

    /* ================================================================ */
    /* Tests 13--15: Consistency tests                                  */
    /* ================================================================ */

    cget51(2, n, S1, lda, S2, lda, Q, ldu, Z, ldu, work, rwork, &result[12]);
    cget51(2, n, P1, lda, P2, lda, Q, ldu, Z, ldu, work, rwork, &result[13]);

    /* Test 15: eigenvalue consistency */
    {
        f32 temp1 = 0.0f;
        f32 temp2 = 0.0f;
        for (INT j = 0; j < n; j++) {
            f32 d1 = cabsf(alpha1[j] - alpha3[j]);
            if (d1 > temp1) temp1 = d1;
            f32 d2 = cabsf(beta1[j] - beta3[j]);
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

    for (INT jr = 0; jr < NTEST; jr++) {
        snprintf(ctx, sizeof(ctx), "zchkgg n=%d type=%d TEST %d",
                 n, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_ok(result[jr]);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_zchkgg_case(void** state)
{
    zchkgg_params_t* params = *state;
    run_zchkgg_single(params);
}

/* ===================================================================== */
/* Build test array and main                                             */
/* ===================================================================== */

#define MAX_TESTS (NNVAL * MAXTYP)

static zchkgg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n_val = NVAL[in];
        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zchkgg_params_t* p = &g_params[g_num_tests];
            p->jsize = (INT)in;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zchkgg_n%d_type%d", n_val, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zchkgg_case;
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
    (void)_cmocka_run_group_tests("zchkgg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
