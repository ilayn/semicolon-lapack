/**
 * @file test_ddrvsg2stg.c
 * @brief DDRVSG2STG checks the real symmetric generalized eigenproblem drivers
 *        including DSYGV_2STAGE.
 *
 * Port of LAPACK TESTING/EIG/ddrvsg2stg.f
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 50.0
#define MAXTYP 21

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Generalized symmetric eigenvalue routines */
extern void dsygv(const int itype, const char* jobz, const char* uplo,
                  const int n, f64* A, const int lda, f64* B, const int ldb,
                  f64* W, f64* work, const int lwork, int* info);
extern void dsygv_2stage(const int itype, const char* jobz, const char* uplo,
                         const int n, f64* A, const int lda, f64* B, const int ldb,
                         f64* W, f64* work, const int lwork, int* info);
extern void dsygvd(const int itype, const char* jobz, const char* uplo,
                   const int n, f64* A, const int lda, f64* B, const int ldb,
                   f64* W, f64* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void dsygvx(const int itype, const char* jobz, const char* range,
                   const char* uplo, const int n, f64* A, const int lda,
                   f64* B, const int ldb, const f64 vl, const f64 vu,
                   const int il, const int iu, const f64 abstol,
                   int* m, f64* W, f64* Z, const int ldz,
                   f64* work, const int lwork, int* iwork,
                   int* ifail, int* info);

/* Packed storage variants */
extern void dspgv(const int itype, const char* jobz, const char* uplo,
                  const int n, f64* AP, f64* BP, f64* W,
                  f64* Z, const int ldz, f64* work, int* info);
extern void dspgvd(const int itype, const char* jobz, const char* uplo,
                   const int n, f64* AP, f64* BP, f64* W,
                   f64* Z, const int ldz, f64* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void dspgvx(const int itype, const char* jobz, const char* range,
                   const char* uplo, const int n, f64* AP, f64* BP,
                   const f64 vl, const f64 vu, const int il, const int iu,
                   const f64 abstol, int* m, f64* W, f64* Z, const int ldz,
                   f64* work, int* iwork, int* ifail, int* info);

/* Banded storage variants */
extern void dsbgv(const char* jobz, const char* uplo, const int n,
                  const int ka, const int kb, f64* AB, const int ldab,
                  f64* BB, const int ldbb, f64* W, f64* Z, const int ldz,
                  f64* work, int* info);
extern void dsbgvd(const char* jobz, const char* uplo, const int n,
                   const int ka, const int kb, f64* AB, const int ldab,
                   f64* BB, const int ldbb, f64* W, f64* Z, const int ldz,
                   f64* work, const int lwork, int* iwork, const int liwork,
                   int* info);
extern void dsbgvx(const char* jobz, const char* range, const char* uplo,
                   const int n, const int ka, const int kb,
                   f64* AB, const int ldab, f64* BB, const int ldbb,
                   f64* Q, const int ldq, const f64 vl, const f64 vu,
                   const int il, const int iu, const f64 abstol,
                   int* m, f64* W, f64* Z, const int ldz,
                   f64* work, int* iwork, int* ifail, int* info);

/* Utility routines */
extern f64 dlamch(const char* cmach);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta, f64* A, const int lda);

typedef struct {
    int n;
    int jtype;
    char name[96];
} ddrvsg2stg_params_t;

typedef struct {
    int nmax;

    f64* A;
    f64* B;
    f64* Z;
    f64* AB;
    f64* BB;
    f64* AP;
    f64* BP;
    f64* D;
    f64* D2;

    f64* work;
    int* iwork;
    int nwork;
    int liwork;

    f64 result[80];

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
        lgn = (int)(log((f64)nmax) / log(2.0));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }
    g_ws->nwork = 1 + 5 * nmax + 2 * nmax * lgn + 3 * n2;
    if (g_ws->nwork < 18) g_ws->nwork = 18;
    g_ws->liwork = 6 * nmax + 6 + 5 * nmax * lgn;
    if (g_ws->liwork < 18) g_ws->liwork = 18;

    /* DSYGV_2STAGE workspace: lwmin = 2*n + lhtrd + lwtrd where
     * lhtrd = 4*n (JOBZ='N'), lwtrd = n*kd + n*max(kd+1,nb) + 2*kd^2 + (kd+1)*n
     * with kd=32, nb=32 (iparam2stage, single-threaded real): 104*n + 2048 */
    {
        int nwork_2stg = 104 * nmax + 2048;
        if (nwork_2stg > g_ws->nwork) {
            g_ws->nwork = nwork_2stg;
        }
    }

    g_ws->A  = malloc(n2 * sizeof(f64));
    g_ws->B  = malloc(n2 * sizeof(f64));
    g_ws->Z  = malloc(n2 * sizeof(f64));
    g_ws->AB = malloc(n2 * sizeof(f64));
    g_ws->BB = malloc(n2 * sizeof(f64));
    /* AP/BP need n^2: AP for packed storage, BP doubles as Q workspace for DSBGVX */
    g_ws->AP = malloc(n2 * sizeof(f64));
    g_ws->BP = malloc(n2 * sizeof(f64));
    g_ws->D  = malloc(nmax * sizeof(f64));
    g_ws->D2 = malloc(nmax * sizeof(f64));

    g_ws->work  = malloc(g_ws->nwork * sizeof(f64));
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
static void pack_symmetric(int n, const char* uplo, const f64* A, int lda,
                           f64* AP)
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
static void dense_to_band(int n, int kd, const char* uplo, const f64* A,
                          int lda, f64* AB, int ldab)
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
                      f64* A, int lda,
                      f64* work, int* iwork,
                      uint64_t rng[static 4],
                      int* ka_out, int* kb_out)
{
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    int iinfo = 0;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);
    f64 rtovfl = sqrt(ovfl);
    f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);

    f64 anorm;
    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0;
    }

    f64 cond = ulpinv;
    int ka, kb;

    dlaset("F", lda, n, 0.0, 0.0, A, lda);

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
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, rng);

    } else if (itype == 5) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, rng);

    } else if (itype == 7) {
        ka = 0;
        kb = 0;
        int idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, rng);

    } else if (itype == 8) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        int idumma[1] = {1};
        dlatmr(n, n, "S", "H", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, rng);

    } else if (itype == 9) {
        g_kb9 = g_kb9 + 1;
        if (g_kb9 > g_ka9) {
            g_ka9 = g_ka9 + 1;
            g_kb9 = 1;
        }
        ka = (n - 1 > 0) ? ((g_ka9 < n - 1) ? g_ka9 : n - 1) : 0;
        kb = (n - 1 > 0) ? ((g_kb9 < n - 1) ? g_kb9 : n - 1) : 0;
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
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

    f64* A    = ws->A;
    f64* B    = ws->B;
    f64* Z    = ws->Z;
    f64* AB   = ws->AB;
    f64* BB   = ws->BB;
    f64* AP   = ws->AP;
    f64* BP   = ws->BP;
    f64* D    = ws->D;
    f64* D2   = ws->D2;
    f64* work = ws->work;
    int* iwork = ws->iwork;
    int nwork = ws->nwork;
    int liwork = ws->liwork;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);
    f64 rtovfl = sqrt(ovfl);
    (void)rtunfl; (void)rtovfl; (void)ovfl;

    int ntest = 0;
    int iinfo;
    int m;
    f64 vl = 0.0, vu = 0.0;
    f64 anorm;

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
        f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);
        switch (KMAGN[jtype - 1]) {
            case 1: anorm = 1.0; break;
            case 2: anorm = (rtovfl * ulp) * aninv; break;
            case 3: anorm = rtunfl * n * ulpinv; break;
            default: anorm = 1.0;
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
    f64 abstol = unfl + unfl;
    int il, iu;
    if (n <= 1) {
        il = 0;
        iu = n - 1;
    } else {
        il = (int)((n - 1) * rng_uniform(ws->rng_state2));
        iu = (int)((n - 1) * rng_uniform(ws->rng_state2));
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
            dlatms(n, n, "U", "P", work, 5, 10.0, 1.0,
                   kb, kb, uplo, B, lda, work + n, &iinfo, ws->rng_state);
            if (iinfo != 0) {
                print_message("  B generator returned info=%d\n", iinfo);
                fail_msg("B matrix generator failed");
                return;
            }

            /* ============ Test DSYGV ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, Z, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            dsygv(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                  work, nwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSYGV_2STAGE ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, Z, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            dsygv_2stage(ibtype, "N", uplo, n, Z, lda, BB, lda, D2,
                         work, nwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGV_2STAGE(N,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGV_2STAGE returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            /* | D1 - D2 | / ( |D1| ulp )
             * D1 computed using the standard 1-stage reduction as reference
             * D2 computed using the 2-stage reduction */
            {
                f64 temp1 = 0.0, temp2 = 0.0;
                for (int j = 0; j < n; j++) {
                    temp1 = fmax(temp1, fmax(fabs(D[j]), fabs(D2[j])));
                    temp2 = fmax(temp2, fabs(D[j] - D2[j]));
                }
                ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            }

            /* ============ Test DSYGVD ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, Z, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            dsygvd(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                   work, nwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSYGVX (RANGE='A') ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, AB, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            dsygvx(ibtype, "V", "A", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSYGVX (RANGE='V') ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, AB, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            vl = 0.0;
            vu = anorm;
            dsygvx(ibtype, "V", "V", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSYGVX (RANGE='I') ============ */
            ntest++;

            dlacpy(" ", n, n, A, lda, AB, lda);
            dlacpy(uplo, n, n, B, lda, BB, lda);

            dsygvx(ibtype, "V", "I", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSYGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSYGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

L100:
            /* ============ Test DSPGV ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            dspgv(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                  work, &iinfo);
            if (iinfo != 0) {
                print_message("  DSPGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSPGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSPGVD ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            dspgvd(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                   work, nwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSPGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSPGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSPGVX (RANGE='A') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            dspgvx(ibtype, "V", "A", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSPGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSPGVX (RANGE='V') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            vl = 0.0;
            vu = anorm;
            dspgvx(ibtype, "V", "V", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSPGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

            /* ============ Test DSPGVX (RANGE='I') ============ */
            ntest++;

            pack_symmetric(n, uplo, A, lda, AP);
            pack_symmetric(n, uplo, B, lda, BP);

            dspgvx(ibtype, "V", "I", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                print_message("  DSPGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("DSPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, &ws->result[ntest - 1]);

L310:
            /* ============ Banded tests (IBTYPE=1 only) ============ */

            if (ibtype == 1) {

                /* ============ Test DSBGV ============ */
                ntest++;

                /* Fortran uses LDA as band leading dimension */
                dlaset("F", lda, n, 0.0, 0.0, AB, lda);
                dlaset("F", lda, n, 0.0, 0.0, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                dsbgv("V", uplo, n, ka, kb, AB, lda, BB, lda,
                      D, Z, lda, work, &iinfo);
                if (iinfo != 0) {
                    print_message("  DSBGV(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("DSBGV returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test DSBGVD ============ */
                ntest++;

                dlaset("F", lda, n, 0.0, 0.0, AB, lda);
                dlaset("F", lda, n, 0.0, 0.0, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                dsbgvd("V", uplo, n, ka, kb, AB, lda, BB, lda,
                       D, Z, lda, work, nwork, iwork, liwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  DSBGVD(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("DSBGVD returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                dsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test DSBGVX (RANGE='A') ============ */
                ntest++;

                dlaset("F", lda, n, 0.0, 0.0, AB, lda);
                dlaset("F", lda, n, 0.0, 0.0, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                int ldq = (n > 1) ? n : 1;
                dsbgvx("V", "A", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  DSBGVX(V,A,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("DSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test DSBGVX (RANGE='V') ============ */
                ntest++;

                dlaset("F", lda, n, 0.0, 0.0, AB, lda);
                dlaset("F", lda, n, 0.0, 0.0, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                vl = 0.0;
                vu = anorm;
                dsbgvx("V", "V", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  DSBGVX(V,V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("DSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, &ws->result[ntest - 1]);

                /* ============ Test DSBGVX (RANGE='I') ============ */
                ntest++;

                dlaset("F", lda, n, 0.0, 0.0, AB, lda);
                dlaset("F", lda, n, 0.0, 0.0, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                dsbgvx("V", "I", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    print_message("  DSBGVX(V,I,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("DSBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                dsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
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
                          j + 1, ws->result[j], THRESH, n, jtype, ka, kb);
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
