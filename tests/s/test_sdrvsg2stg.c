/**
 * @file test_sdrvsg2stg.c
 * @brief DDRVSG2STG checks the real symmetric generalized eigenproblem drivers
 *        including SSYGV_2STAGE.
 *
 * Port of LAPACK TESTING/EIG/ddrvsg2stg.f
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 50.0f
#define MAXTYP 21

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Generalized symmetric eigenvalue routines */
extern void ssygv(const int itype, const char* jobz, const char* uplo,
                  const int n, f32* A, const int lda, f32* B, const int ldb,
                  f32* W, f32* work, const int lwork, int* info);
extern void ssygv_2stage(const int itype, const char* jobz, const char* uplo,
                         const int n, f32* A, const int lda, f32* B, const int ldb,
                         f32* W, f32* work, const int lwork, int* info);
extern void ssygvd(const int itype, const char* jobz, const char* uplo,
                   const int n, f32* A, const int lda, f32* B, const int ldb,
                   f32* W, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void ssygvx(const int itype, const char* jobz, const char* range,
                   const char* uplo, const int n, f32* A, const int lda,
                   f32* B, const int ldb, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol,
                   int* m, f32* W, f32* Z, const int ldz,
                   f32* work, const int lwork, int* iwork,
                   int* ifail, int* info);

/* Packed storage variants */
extern void sspgv(const int itype, const char* jobz, const char* uplo,
                  const int n, f32* AP, f32* BP, f32* W,
                  f32* Z, const int ldz, f32* work, int* info);
extern void sspgvd(const int itype, const char* jobz, const char* uplo,
                   const int n, f32* AP, f32* BP, f32* W,
                   f32* Z, const int ldz, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void sspgvx(const int itype, const char* jobz, const char* range,
                   const char* uplo, const int n, f32* AP, f32* BP,
                   const f32 vl, const f32 vu, const int il, const int iu,
                   const f32 abstol, int* m, f32* W, f32* Z, const int ldz,
                   f32* work, int* iwork, int* ifail, int* info);

/* Banded storage variants */
extern void ssbgv(const char* jobz, const char* uplo, const int n,
                  const int ka, const int kb, f32* AB, const int ldab,
                  f32* BB, const int ldbb, f32* W, f32* Z, const int ldz,
                  f32* work, int* info);
extern void ssbgvd(const char* jobz, const char* uplo, const int n,
                   const int ka, const int kb, f32* AB, const int ldab,
                   f32* BB, const int ldbb, f32* W, f32* Z, const int ldz,
                   f32* work, const int lwork, int* iwork, const int liwork,
                   int* info);
extern void ssbgvx(const char* jobz, const char* range, const char* uplo,
                   const int n, const int ka, const int kb,
                   f32* AB, const int ldab, f32* BB, const int ldbb,
                   f32* Q, const int ldq, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol,
                   int* m, f32* W, f32* Z, const int ldz,
                   f32* work, int* iwork, int* ifail, int* info);

/* Utility routines */
extern f32 slamch(const char* cmach);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta, f32* A, const int lda);

typedef struct {
    int n;
    int jtype;
    char name[96];
} ddrvsg2stg_params_t;

typedef struct {
    int nmax;

    f32* A;
    f32* B;
    f32* Z;
    f32* AB;
    f32* BB;
    f32* AP;
    f32* BP;
    f32* D;
    f32* D2;

    f32* work;
    int* iwork;
    int nwork;
    int liwork;

    f32 result[80];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
} ddrvsg2stg_workspace_t;

static ddrvsg2stg_workspace_t* g_ws = NULL;

/*
 * ka9/kb9 cycling state for banded types (16-21).
 * Reset per size, advanced per jtype==9 (KTYPE==9).
 */
static int g_ka9 = 0;
static int g_kb9 = 0;
static int g_last_n = -1;

static const int KTYPE[MAXTYP] = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 9};
static const int KMAGN[MAXTYP] = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1};
static const int KMODE[MAXTYP] = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvsg2stg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

    /* Workspace sizes from ddrvsg2stg.f docstring:
     * NWORK >= 1 + 5*N + 2*N*lg(N) + 3*N^2
     * LIWORK >= 6*N
     * where lg(N) = smallest integer k such that 2^k >= N. */
    int lgn = 0;
    if (nmax > 0) {
        lgn = (int)(logf((f32)nmax) / logf(2.0f));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }
    g_ws->nwork = 1 + 5 * nmax + 2 * nmax * lgn + 3 * n2;
    if (g_ws->nwork < 18) g_ws->nwork = 18;
    g_ws->liwork = 6 * nmax + 6 + 5 * nmax * lgn;
    if (g_ws->liwork < 18) g_ws->liwork = 18;

    /* SSYGV_2STAGE workspace: lwmin = 2*n + lhtrd + lwtrd where
     * lhtrd = 4*n (JOBZ='N'), lwtrd = n*kd + n*max(kd+1,nb) + 2*kd^2 + (kd+1)*n
     * with kd=32, nb=32 (iparam2stage, single-threaded real): 104*n + 2048 */
    {
        int nwork_2stg = 104 * nmax + 2048;
        if (nwork_2stg > g_ws->nwork) {
            g_ws->nwork = nwork_2stg;
        }
    }

    g_ws->A  = malloc(n2 * sizeof(f32));
    g_ws->B  = malloc(n2 * sizeof(f32));
    g_ws->Z  = malloc(n2 * sizeof(f32));
    g_ws->AB = malloc(n2 * sizeof(f32));
    g_ws->BB = malloc(n2 * sizeof(f32));
    /* AP/BP need n^2: AP for packed storage, BP doubles as Q workspace for SSBGVX */
    g_ws->AP = malloc(n2 * sizeof(f32));
    g_ws->BP = malloc(n2 * sizeof(f32));
    g_ws->D  = malloc(nmax * sizeof(f32));
    g_ws->D2 = malloc(nmax * sizeof(f32));

    g_ws->work  = malloc(g_ws->nwork * sizeof(f32));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(int));

    if (!g_ws->A || !g_ws->B || !g_ws->Z || !g_ws->AB || !g_ws->BB ||
        !g_ws->AP || !g_ws->BP || !g_ws->D || !g_ws->D2 ||
        !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xB47C2D935F81A06EULL);
    rng_seed(g_ws->rng_state2, 0xB47C2D935F81A06EULL);

    g_ka9 = 0;
    g_kb9 = 0;
    g_last_n = -1;

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->Z);
        free(g_ws->AB);
        free(g_ws->BB);
        free(g_ws->AP);
        free(g_ws->BP);
        free(g_ws->D);
        free(g_ws->D2);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/*
 * Pack upper or lower triangle of dense A into packed storage AP.
 */
static void pack_symmetric(int n, const char* uplo, const f32* A, int lda,
                           f32* AP)
{
    int ij = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                AP[ij] = A[i + j * lda];
                ij++;
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                AP[ij] = A[i + j * lda];
                ij++;
            }
        }
    }
}

/*
 * Convert dense matrix to LAPACK band storage.
 */
static void dense_to_band(int n, int kd, const char* uplo, const f32* A,
                          int lda, f32* AB, int ldab)
{
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int j = 0; j < n; j++) {
            int imin = (j - kd > 0) ? j - kd : 0;
            for (int i = imin; i <= j; i++) {
                AB[(kd + i - j) + j * ldab] = A[i + j * lda];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            int imax = (j + kd < n - 1) ? j + kd : n - 1;
            for (int i = j; i <= imax; i++) {
                AB[(i - j) + j * ldab] = A[i + j * lda];
            }
        }
    }
}

/*
 * Generate matrix A for the given jtype.
 * Port of ddrvsg2stg.f lines 544-632.
 * Returns 0 on success, nonzero on failure.
 * Sets *ka_out and *kb_out to the half-bandwidths of A and B.
 */
static int generate_A(int n, int jtype,
                      f32* A, int lda,
                      f32* work, int* iwork,
                      uint64_t rng[static 4],
                      int* ka_out, int* kb_out)
{
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    int iinfo = 0;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);
    f32 rtovfl = sqrtf(ovfl);
    f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);

    f32 anorm;
    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0f; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0f;
    }

    f32 cond = ulpinv;
    int ka, kb;

    slaset("F", lda, n, 0.0f, 0.0f, A, lda);

    if (itype == 1) {
        ka = 0;
        kb = 0;

    } else if (itype == 2) {
        ka = 0;
        kb = 0;
        for (int jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 4) {
        ka = 0;
        kb = 0;
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, rng);

    } else if (itype == 5) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, rng);

    } else if (itype == 7) {
        ka = 0;
        kb = 0;
        int idumma[1] = {1};
        slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, 0, 0, 0.0f, anorm, "NO",
               A, lda, iwork, &iinfo, rng);

    } else if (itype == 8) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        int idumma[1] = {1};
        slatmr(n, n, "S", "H", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, n, n, 0.0f, anorm, "NO",
               A, lda, iwork, &iinfo, rng);

    } else if (itype == 9) {
        g_kb9 = g_kb9 + 1;
        if (g_kb9 > g_ka9) {
            g_ka9 = g_ka9 + 1;
            g_kb9 = 1;
        }
        ka = (n - 1 > 0) ? ((g_ka9 < n - 1) ? g_ka9 : n - 1) : 0;
        kb = (n - 1 > 0) ? ((g_kb9 < n - 1) ? g_kb9 : n - 1) : 0;
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               ka, ka, "N", A, lda, work + n, &iinfo, rng);

    } else {
        iinfo = 1;
        ka = 0;
        kb = 0;
    }

    *ka_out = ka;
    *kb_out = kb;
    return iinfo;
}

/*
 * Run all tests for a single (n, jtype) combination.
 * Port of ddrvsg2stg.f lines 478-1295.
 */
static void run_ddrvsg2stg_single(ddrvsg2stg_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;

    ddrvsg2stg_workspace_t* ws = g_ws;
    int nmax = ws->nmax;
    int lda = (nmax > 1) ? nmax : 1;

    f32* A    = ws->A;
    f32* B    = ws->B;
    f32* Z    = ws->Z;
    f32* AB   = ws->AB;
    f32* BB   = ws->BB;
    f32* AP   = ws->AP;
    f32* BP   = ws->BP;
    f32* D    = ws->D;
    f32* D2   = ws->D2;
    f32* work = ws->work;
    int* iwork = ws->iwork;
    int nwork = ws->nwork;
    int liwork = ws->liwork;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);
    f32 rtovfl = sqrtf(ovfl);
    (void)rtunfl; (void)rtovfl; (void)ovfl;

    int ntest = 0;
    int iinfo;
    int m;
    f32 vl = 0.0f, vu = 0.0f;
    f32 anorm;

    /* Reset ka9/kb9 cycling per size */
    if (n != g_last_n) {
        g_ka9 = 0;
        g_kb9 = 0;
        g_last_n = n;
    }

    if (n == 0) {
        return;
    }

    /* Compute anorm for VL/VU bounds in RANGE='V' tests */
    {
        f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);
        switch (KMAGN[jtype - 1]) {
            case 1: anorm = 1.0f; break;
            case 2: anorm = (rtovfl * ulp) * aninv; break;
            case 3: anorm = rtunfl * n * ulpinv; break;
            default: anorm = 1.0f;
        }
    }

    /* Generate A */
    int ka, kb;
    iinfo = generate_A(n, jtype, A, lda, work, iwork, ws->rng_state,
                       &ka, &kb);
    if (iinfo != 0) {
        print_message("  Generator returned info=%d for n=%d jtype=%d\n",
                      iinfo, n, jtype);
        fail_msg("Matrix generator failed");
        return;
    }

    /* Compute IL, IU (0-based) */
    f32 abstol = unfl + unfl;
    int il, iu;
    if (n <= 1) {
        il = 0;
        iu = n - 1;
    } else {
        il = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        iu = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        if (il > iu) {
            int itemp = il;
            il = iu;
            iu = itemp;
        }
    }

    memset(ws->result, 0, sizeof(ws->result));

    for (int ibtype = 1; ibtype <= 3; ibtype++) {
        for (int ibuplo = 0; ibuplo < 2; ibuplo++) {
            const char* uplo = (ibuplo == 0) ? "U" : "L";

            /* Generate B: well-conditioned positive definite */
            iinfo = 0;
            slatms(n, n, "U", "P", work, 5, 10.0f, 1.0f,
                   kb, kb, uplo, B, lda, work + n, &iinfo, ws->rng_state);
            if (iinfo != 0) {
                print_message("  B generator returned info=%d\n", iinfo);
                fail_msg("B matrix generator failed");
                return;
            }

            /* ============ Test SSYGV ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, Z, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            ssygv(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                  work, nwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSYGV_2STAGE ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, Z, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            ssygv_2stage(ibtype, "N", uplo, n, Z, lda, BB, lda, D2,
                         work, nwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGV_2STAGE(N,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGV_2STAGE returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            /* | D1 - D2 | / ( |D1| ulp )
             * D1 computed using the standard 1-stage reduction as reference
             * D2 computed using the 2-stage reduction */
            {
                f32 temp1 = 0.0f, temp2 = 0.0f;
                for (int j = 0; j < n; j++) {
                    temp1 = fmaxf(temp1, fmaxf(fabsf(D[j]), fabsf(D2[j])));
                    temp2 = fmaxf(temp2, fabsf(D[j] - D2[j]));
                }
                ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));
            }

            /* ============ Test SSYGVD ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, Z, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            ssygvd(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                   work, nwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSYGVX (RANGE='A') ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, AB, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            ssygvx(ibtype, "V", "A", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSYGVX (RANGE='V') ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, AB, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            vl = 0.0f;
            vu = anorm;
            ssygvx(ibtype, "V", "V", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSYGVX (RANGE='I') ============ */
            ntest++;

            slacpy(" ", n, n, A, lda, AB, lda);
            slacpy(uplo, n, n, B, lda, BB, lda);

            ssygvx(ibtype, "V", "I", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSYGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

L100:
            /* ============ Test SSPGV ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            sspgv(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                  work, &iinfo);
            if (iinfo != 0) {
                print_message("  SSPGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSPGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSPGVD ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            sspgvd(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                   work, nwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSPGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSPGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSPGVX (RANGE='A') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            sspgvx(ibtype, "V", "A", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSPGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSPGVX (RANGE='V') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            vl = 0.0f;
            vu = anorm;
            sspgvx(ibtype, "V", "V", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSPGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test SSPGVX (RANGE='I') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            sspgvx(ibtype, "V", "I", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  SSPGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("SSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

L310:
            /* ============ Banded tests (IBTYPE=1 only) ============ */

            if (ibtype == 1) {

                /* ============ Test SSBGV ============ */
                ntest++;

                /* Fortran uses LDA as band leading dimension */
                slaset("F", lda, n, 0.0f, 0.0f, AB, lda);
                slaset("F", lda, n, 0.0f, 0.0f, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                ssbgv("V", uplo, n, ka, kb, AB, lda, BB, lda,
                      D, Z, lda, work, &iinfo);
                if (iinfo != 0) {
                    print_message("  SSBGV(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("SSBGV returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test SSBGVD ============ */
                ntest++;

                slaset("F", lda, n, 0.0f, 0.0f, AB, lda);
                slaset("F", lda, n, 0.0f, 0.0f, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                ssbgvd("V", uplo, n, ka, kb, AB, lda, BB, lda,
                       D, Z, lda, work, nwork, iwork, liwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  SSBGVD(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("SSBGVD returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                ssgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test SSBGVX (RANGE='A') ============ */
                ntest++;

                slaset("F", lda, n, 0.0f, 0.0f, AB, lda);
                slaset("F", lda, n, 0.0f, 0.0f, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                int ldq = (n > 1) ? n : 1;
                ssbgvx("V", "A", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  SSBGVX(V,A,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("SSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test SSBGVX (RANGE='V') ============ */
                ntest++;

                slaset("F", lda, n, 0.0f, 0.0f, AB, lda);
                slaset("F", lda, n, 0.0f, 0.0f, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                vl = 0.0f;
                vu = anorm;
                ssbgvx("V", "V", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  SSBGVX(V,V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("SSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test SSBGVX (RANGE='I') ============ */
                ntest++;

                slaset("F", lda, n, 0.0f, 0.0f, AB, lda);
                slaset("F", lda, n, 0.0f, 0.0f, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                ssbgvx("V", "I", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  SSBGVX(V,I,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("SSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                ssgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

            } /* end if ibtype == 1 */

L620:
            ; /* continue to next ibuplo */
        } /* end ibuplo loop */
    } /* end ibtype loop */

    /* Check results against threshold */
    for (int j = 0; j < ntest; j++) {
        if (ws->result[j] >= THRESH) {
            print_message("  Test %d: ratio = %.6e (THRESH=%.1f) n=%d jtype=%d ka=%d kb=%d\n",
                          j + 1, (double)ws->result[j], (double)THRESH, n, jtype, ka, kb);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

static void test_ddrvsg2stg_case(void** state)
{
    ddrvsg2stg_params_t* params = *state;
    run_ddrvsg2stg_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static ddrvsg2stg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            ddrvsg2stg_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "ddrvsg2stg_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_ddrvsg2stg_case;
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

    return _cmocka_run_group_tests("ddrvsg2stg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
