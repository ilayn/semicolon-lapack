/**
 * @file test_zdrvsg2stg.c
 * @brief ZDRVSG2STG checks the complex Hermitian generalized eigenproblem
 *        drivers including ZHEGV_2STAGE.
 *
 * Port of LAPACK TESTING/EIG/zdrvsg2stg.f
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#define THRESH 50.0
#define MAXTYP 21

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT n;
    INT jtype;
    char name[96];
} zdrvsg2stg_params_t;

typedef struct {
    INT nmax;

    c128* A;
    c128* B;
    c128* Z;
    c128* AB;
    c128* BB;
    c128* AP;
    c128* BP;
    f64*  D;
    f64*  D2;

    c128* work;
    f64*  rwork;
    INT*  iwork;
    INT   nwork;
    INT   lrwork;
    INT   liwork;

    f64 result[80];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
} zdrvsg2stg_workspace_t;

static zdrvsg2stg_workspace_t* g_ws = NULL;

static INT g_ka9 = 0;
static INT g_kb9 = 0;
static INT g_last_n = -1;

static const INT KTYPE[MAXTYP] = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 9};
static const INT KMAGN[MAXTYP] = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1};
static const INT KMODE[MAXTYP] = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvsg2stg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    /* Workspace sizes from zdrvsg2stg.f:
     * NWORK  >= 2*MAX(NMAX,2)^2  (complex workspace)
     * LRWORK >= 2*MAX(NMAX,2)^2  (real workspace)
     * LIWORK >= 2*MAX(NMAX,2)^2
     *
     * Actual requirements from ZHEGVD:
     *   LWORK  >= 2*N + N^2
     *   LRWORK >= 1 + 5*N + 2*N*lg(N) + 3*N^2
     *   LIWORK >= 3 + 5*N
     * where lg(N) = smallest integer k such that 2^k >= N. */
    INT lgn = 0;
    if (nmax > 0) {
        lgn = (INT)(log((f64)nmax) / log(2.0));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }

    INT nwork1 = 2 * n2;
    INT nwork2 = 2 * nmax + n2;
    g_ws->nwork = (nwork1 > nwork2) ? nwork1 : nwork2;
    if (g_ws->nwork < 18) g_ws->nwork = 18;

    g_ws->lrwork = 1 + 5 * nmax + 2 * nmax * lgn + 3 * n2;
    if (g_ws->lrwork < 2 * n2) g_ws->lrwork = 2 * n2;
    if (g_ws->lrwork < 18) g_ws->lrwork = 18;

    g_ws->liwork = 6 * nmax + 6 + 5 * nmax * lgn;
    if (g_ws->liwork < 2 * n2) g_ws->liwork = 2 * n2;
    if (g_ws->liwork < 18) g_ws->liwork = 18;

    /* ZHEGV_2STAGE workspace: lwmin = n + lhtrd + lwtrd where
     * lhtrd = 4*n (JOBZ='N'), lwtrd = n*kd + n*max(kd+1,nb) + 2*kd^2 + (kd+1)*n
     * with kd=16, nb=32 (iparam2stage, single-threaded complex): ~70*n + 512 */
    {
        INT nwork_2stg = 70 * nmax + 512;
        if (nwork_2stg > g_ws->nwork) {
            g_ws->nwork = nwork_2stg;
        }
    }

    g_ws->A  = malloc(n2 * sizeof(c128));
    g_ws->B  = malloc(n2 * sizeof(c128));
    g_ws->Z  = malloc(n2 * sizeof(c128));
    g_ws->AB = malloc(n2 * sizeof(c128));
    g_ws->BB = malloc(n2 * sizeof(c128));
    g_ws->AP = malloc(n2 * sizeof(c128));
    g_ws->BP = malloc(n2 * sizeof(c128));
    g_ws->D  = malloc(nmax * sizeof(f64));
    g_ws->D2 = malloc(nmax * sizeof(f64));

    g_ws->work  = malloc(g_ws->nwork * sizeof(c128));
    g_ws->rwork = malloc(g_ws->lrwork * sizeof(f64));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(INT));

    if (!g_ws->A || !g_ws->B || !g_ws->Z || !g_ws->AB || !g_ws->BB ||
        !g_ws->AP || !g_ws->BP || !g_ws->D || !g_ws->D2 ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
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
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void pack_hermitian(INT n, const char* uplo, const c128* A, INT lda,
                           c128* AP)
{
    INT ij = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i <= j; i++) {
                AP[ij] = A[i + j * lda];
                ij++;
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            for (INT i = j; i < n; i++) {
                AP[ij] = A[i + j * lda];
                ij++;
            }
        }
    }
}

static void dense_to_band(INT n, INT kd, const char* uplo, const c128* A,
                          INT lda, c128* AB, INT ldab)
{
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT j = 0; j < n; j++) {
            INT imin = (j - kd > 0) ? j - kd : 0;
            for (INT i = imin; i <= j; i++) {
                AB[(kd + i - j) + j * ldab] = A[i + j * lda];
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            INT imax = (j + kd < n - 1) ? j + kd : n - 1;
            for (INT i = j; i <= imax; i++) {
                AB[(i - j) + j * ldab] = A[i + j * lda];
            }
        }
    }
}

/*
 * Generate matrix A for the given jtype.
 * Port of zdrvsg2stg.f lines 544-650.
 */
static INT generate_A(INT n, INT jtype,
                      c128* A, INT lda,
                      c128* work, f64* rwork, INT* iwork,
                      uint64_t rng[static 4],
                      INT* ka_out, INT* kb_out)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE  = CMPLX(1.0, 0.0);

    INT itype = KTYPE[jtype - 1];
    INT imode = KMODE[jtype - 1];
    INT iinfo = 0;

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
    INT ka, kb;

    zlaset("F", lda, n, CZERO, CZERO, A, lda);

    if (itype == 1) {
        ka = 0;
        kb = 0;

    } else if (itype == 2) {
        ka = 0;
        kb = 0;
        for (INT jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = CMPLX(anorm, 0.0);
        }

    } else if (itype == 4) {
        ka = 0;
        kb = 0;
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work, &iinfo, rng);

    } else if (itype == 5) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               n, n, "N", A, lda, work, &iinfo, rng);

    } else if (itype == 7) {
        ka = 0;
        kb = 0;
        INT idumma[1] = {1};
        zlatmr(n, n, "S", "H", work, 6, 1.0, CONE, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, rng);

    } else if (itype == 8) {
        ka = (n - 1 > 0) ? n - 1 : 0;
        kb = ka;
        INT idumma[1] = {1};
        zlatmr(n, n, "S", "H", work, 6, 1.0, CONE, "T", "N",
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
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               ka, ka, "N", A, lda, work, &iinfo, rng);

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
 * Port of zdrvsg2stg.f lines 478-1306.
 */
static void run_zdrvsg2stg_single(zdrvsg2stg_params_t* params)
{
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT n = params->n;
    INT jtype = params->jtype;

    zdrvsg2stg_workspace_t* ws = g_ws;
    INT nmax = ws->nmax;
    INT lda = (nmax > 1) ? nmax : 1;

    c128* A    = ws->A;
    c128* B    = ws->B;
    c128* Z    = ws->Z;
    c128* AB   = ws->AB;
    c128* BB   = ws->BB;
    c128* AP   = ws->AP;
    c128* BP   = ws->BP;
    f64*  D    = ws->D;
    f64*  D2   = ws->D2;
    c128* work = ws->work;
    f64*  rwork = ws->rwork;
    INT*  iwork = ws->iwork;
    INT   nwork = ws->nwork;
    INT   lrwork = ws->lrwork;
    INT   liwork = ws->liwork;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);
    f64 rtovfl = sqrt(ovfl);
    (void)rtunfl; (void)rtovfl; (void)ovfl;

    INT ntest = 0;
    INT iinfo;
    INT m;
    f64 vl = 0.0, vu = 0.0;
    f64 anorm;

    if (n != g_last_n) {
        g_ka9 = 0;
        g_kb9 = 0;
        g_last_n = n;
    }

    if (n == 0) {
        return;
    }

    {
        f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);
        switch (KMAGN[jtype - 1]) {
            case 1: anorm = 1.0; break;
            case 2: anorm = (rtovfl * ulp) * aninv; break;
            case 3: anorm = rtunfl * n * ulpinv; break;
            default: anorm = 1.0;
        }
    }

    INT ka, kb;
    iinfo = generate_A(n, jtype, A, lda, work, rwork, iwork, ws->rng_state,
                       &ka, &kb);
    if (iinfo != 0) {
        fprintf(stderr, "  Generator returned info=%d for n=%d jtype=%d\n",
                      iinfo, n, jtype);
        fail_msg("Matrix generator failed");
        return;
    }

    f64 abstol = unfl + unfl;
    INT il, iu;
    if (n <= 1) {
        il = 0;
        iu = n - 1;
    } else {
        il = (INT)((n - 1) * rng_uniform(ws->rng_state2));
        iu = (INT)((n - 1) * rng_uniform(ws->rng_state2));
        if (il > iu) {
            INT itemp = il;
            il = iu;
            iu = itemp;
        }
    }

    memset(ws->result, 0, sizeof(ws->result));

    for (INT ibtype = 1; ibtype <= 3; ibtype++) {
        for (INT ibuplo = 0; ibuplo < 2; ibuplo++) {
            const char* uplo = (ibuplo == 0) ? "U" : "L";

            iinfo = 0;
            zlatms(n, n, "U", "P", rwork, 5, 10.0, 1.0,
                   kb, kb, uplo, B, lda, work + n, &iinfo, ws->rng_state);
            if (iinfo != 0) {
                fprintf(stderr, "  B generator returned info=%d\n", iinfo);
                fail_msg("B matrix generator failed");
                return;
            }

            /* ============ Test ZHEGV ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, Z, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            zhegv(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                  work, nwork, rwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHEGV_2STAGE ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, Z, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            zhegv_2stage(ibtype, "N", uplo, n, Z, lda, BB, lda, D2,
                         work, nwork, rwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGV_2STAGE(N,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGV_2STAGE returned negative info");
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
                for (INT j = 0; j < n; j++) {
                    temp1 = fmax(temp1, fmax(fabs(D[j]), fabs(D2[j])));
                    temp2 = fmax(temp2, fabs(D[j] - D2[j]));
                }
                ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            }

            /* ============ Test ZHEGVD ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, Z, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            zhegvd(ibtype, "V", uplo, n, Z, lda, BB, lda, D,
                   work, nwork, rwork, lrwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHEGVX (RANGE='A') ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, AB, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            zhegvx(ibtype, "V", "A", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHEGVX (RANGE='V') ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, AB, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            vl = 0.0;
            vu = anorm;
            zhegvx(ibtype, "V", "V", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHEGVX (RANGE='I') ============ */
            ntest++;

            zlacpy(" ", n, n, A, lda, AB, lda);
            zlacpy(uplo, n, n, B, lda, BB, lda);

            zhegvx(ibtype, "V", "I", uplo, n, AB, lda, BB, lda,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, nwork, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHEGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHEGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L100;
            }

            zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

L100:
            /* ============ Test ZHPGV ============ */
            ntest++;

            pack_hermitian(n, uplo, A, lda, AP);
            pack_hermitian(n, uplo, B, lda, BP);

            zhpgv(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                  work, rwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHPGV(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHPGV returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHPGVD ============ */
            ntest++;

            pack_hermitian(n, uplo, A, lda, AP);
            pack_hermitian(n, uplo, B, lda, BP);

            zhpgvd(ibtype, "V", uplo, n, AP, BP, D, Z, lda,
                   work, nwork, rwork, lrwork, iwork, liwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHPGVD(V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHPGVD returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHPGVX (RANGE='A') ============ */
            ntest++;

            pack_hermitian(n, uplo, A, lda, AP);
            pack_hermitian(n, uplo, B, lda, BP);

            zhpgvx(ibtype, "V", "A", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHPGVX(V,A,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHPGVX (RANGE='V') ============ */
            ntest++;

            pack_hermitian(n, uplo, A, lda, AP);
            pack_hermitian(n, uplo, B, lda, BP);

            vl = 0.0;
            vu = anorm;
            zhpgvx(ibtype, "V", "V", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHPGVX(V,V,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

            /* ============ Test ZHPGVX (RANGE='I') ============ */
            ntest++;

            pack_hermitian(n, uplo, A, lda, AP);
            pack_hermitian(n, uplo, B, lda, BP);

            zhpgvx(ibtype, "V", "I", uplo, n, AP, BP,
                   vl, vu, il, iu, abstol, &m, D, Z, lda,
                   work, rwork, iwork + n, iwork, &iinfo);
            if (iinfo != 0) {
                fprintf(stderr, "  ZHPGVX(V,I,%s) info=%d n=%d jtype=%d ibtype=%d\n",
                              uplo, iinfo, n, jtype, ibtype);
                if (iinfo < 0) {
                    fail_msg("ZHPGVX returned negative info");
                    return;
                }
                ws->result[ntest - 1] = ulpinv;
                goto L310;
            }

            zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                   D, work, rwork, &ws->result[ntest - 1]);

L310:
            /* ============ Banded tests (IBTYPE=1 only) ============ */

            if (ibtype == 1) {

                /* ============ Test ZHBGV ============ */
                ntest++;

                zlaset("F", lda, n, CZERO, CZERO, AB, lda);
                zlaset("F", lda, n, CZERO, CZERO, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                zhbgv("V", uplo, n, ka, kb, AB, lda, BB, lda,
                      D, Z, lda, work, rwork, &iinfo);
                if (iinfo != 0) {
                    fprintf(stderr, "  ZHBGV(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("ZHBGV returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, rwork, &ws->result[ntest - 1]);

                /* ============ Test ZHBGVD ============ */
                ntest++;

                zlaset("F", lda, n, CZERO, CZERO, AB, lda);
                zlaset("F", lda, n, CZERO, CZERO, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                zhbgvd("V", uplo, n, ka, kb, AB, lda, BB, lda,
                       D, Z, lda, work, nwork, rwork, lrwork, iwork, liwork, &iinfo);
                if (iinfo != 0) {
                    fprintf(stderr, "  ZHBGVD(V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("ZHBGVD returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                zsgt01(ibtype, uplo, n, n, A, lda, B, lda, Z, lda,
                       D, work, rwork, &ws->result[ntest - 1]);

                /* ============ Test ZHBGVX (RANGE='A') ============ */
                ntest++;

                zlaset("F", lda, n, CZERO, CZERO, AB, lda);
                zlaset("F", lda, n, CZERO, CZERO, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                INT ldq = (n > 1) ? n : 1;
                zhbgvx("V", "A", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, rwork, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    fprintf(stderr, "  ZHBGVX(V,A,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("ZHBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, rwork, &ws->result[ntest - 1]);

                /* ============ Test ZHBGVX (RANGE='V') ============ */
                ntest++;

                zlaset("F", lda, n, CZERO, CZERO, AB, lda);
                zlaset("F", lda, n, CZERO, CZERO, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                vl = 0.0;
                vu = anorm;
                zhbgvx("V", "V", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, rwork, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    fprintf(stderr, "  ZHBGVX(V,V,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("ZHBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, rwork, &ws->result[ntest - 1]);

                /* ============ Test ZHBGVX (RANGE='I') ============ */
                ntest++;

                zlaset("F", lda, n, CZERO, CZERO, AB, lda);
                zlaset("F", lda, n, CZERO, CZERO, BB, lda);
                dense_to_band(n, ka, uplo, A, lda, AB, lda);
                dense_to_band(n, kb, uplo, B, lda, BB, lda);

                zhbgvx("V", "I", uplo, n, ka, kb, AB, lda, BB, lda,
                       BP, ldq, vl, vu, il, iu, abstol, &m, D, Z, lda,
                       work, rwork, iwork + n, iwork, &iinfo);
                if (iinfo != 0) {
                    fprintf(stderr, "  ZHBGVX(V,I,%s) info=%d n=%d jtype=%d\n",
                                  uplo, iinfo, n, jtype);
                    if (iinfo < 0) {
                        fail_msg("ZHBGVX returned negative info");
                        return;
                    }
                    ws->result[ntest - 1] = ulpinv;
                    goto L620;
                }

                zsgt01(ibtype, uplo, n, m, A, lda, B, lda, Z, lda,
                       D, work, rwork, &ws->result[ntest - 1]);

            } /* end if ibtype == 1 */

L620:
            ; /* continue to next ibuplo */
        } /* end ibuplo loop */
    } /* end ibtype loop */

    for (INT j = 0; j < ntest; j++) {
        if (ws->result[j] >= THRESH) {
            fprintf(stderr, "  Test %d: ratio = %.6e (THRESH=%.1f) n=%d jtype=%d ka=%d kb=%d\n",
                          j + 1, ws->result[j], THRESH, n, jtype, ka, kb);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

static void test_zdrvsg2stg_case(void** state)
{
    zdrvsg2stg_params_t* params = *state;
    run_zdrvsg2stg_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static zdrvsg2stg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zdrvsg2stg_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zdrvsg2stg_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zdrvsg2stg_case;
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

    return _cmocka_run_group_tests("zdrvsg2stg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
