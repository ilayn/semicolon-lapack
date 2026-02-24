/**
 * @file test_ddrvst2stg.c
 * @brief DDRVST2STG checks the symmetric eigenvalue problem drivers
 *        including the 2-stage variants.
 *
 * Port of LAPACK TESTING/EIG/ddrvst2stg.f
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 50.0
#define MAXTYP 18

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Tridiagonal eigenvalue routines */
/* Full symmetric eigenvalue routines */
/* Packed symmetric eigenvalue routines */
/* Band symmetric eigenvalue routines */
/* 2-stage symmetric eigenvalue routines */
/* 2-stage band symmetric eigenvalue routines */
/* Utility routines */
typedef struct {
    INT n;
    INT jtype;
    char name[96];
} ddrvst2stg_params_t;

typedef struct {
    INT nmax;

    f64* A;
    f64* U;
    f64* V;
    f64* Z;

    f64* D1;
    f64* D2;
    f64* D3;
    f64* D4;
    f64* WA1;
    f64* WA2;
    f64* WA3;
    f64* EVEIGS;
    f64* TAU;

    f64* work;
    INT* iwork;
    INT lwork;
    INT liwork;

    f64 result[140];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
    uint64_t rng_state3[4];
} ddrvst2stg_workspace_t;

static ddrvst2stg_workspace_t* g_ws = NULL;

static const INT KTYPE[MAXTYP]  = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9};
static const INT KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3};
static const INT KMODE[MAXTYP]  = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvst2stg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    /* Workspace sizes from ddrvst.f lines 274-282, 596-609 */
    INT lgn = 0;
    if (nmax > 0) {
        lgn = (INT)(log((f64)nmax) / log(2.0));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }
    /* LWORK: 1 + 4*Nmax + 2*Nmax*lgn + 4*Nmax**2 (line 602) */
    g_ws->lwork = 1 + 4 * nmax + 2 * nmax * lgn + 4 * n2;
    /* IWORK: 6 + 6*Nmax + 5*Nmax*lgNmax (line 281-282) */
    g_ws->liwork = 6 + 6 * nmax + 5 * nmax * lgn;

    /* 2-stage routines need additional workspace for internal band reduction.
     * Query dsyevr_2stage (most demanding) to get the required size. */
    {
        f64 work_query;
        INT iwork_query;
        INT info_query;
        dsyevr_2stage("N", "A", "L", nmax, NULL, nmax, 0.0, 0.0, 0, 0,
                      0.0, NULL, NULL, NULL, nmax, NULL, &work_query, -1,
                      &iwork_query, -1, &info_query);
        INT lwork_2stage = (INT)work_query;
        if (lwork_2stage > g_ws->lwork)
            g_ws->lwork = lwork_2stage;
    }

    g_ws->A = malloc(n2 * sizeof(f64));
    /* U: band storage needs up to LDU = max(2*nmax-1, nmax) rows */
    INT ldu_band = (2 * nmax - 1) > nmax ? (2 * nmax - 1) : nmax;
    g_ws->U = malloc(ldu_band * nmax * sizeof(f64));
    g_ws->V = malloc(n2 * sizeof(f64));
    g_ws->Z = malloc(n2 * sizeof(f64));

    g_ws->D1     = malloc(nmax * sizeof(f64));
    g_ws->D2     = malloc(nmax * sizeof(f64));
    g_ws->D3     = malloc(nmax * sizeof(f64));
    g_ws->D4     = malloc(nmax * sizeof(f64));
    g_ws->WA1    = malloc(nmax * sizeof(f64));
    g_ws->WA2    = malloc(nmax * sizeof(f64));
    g_ws->WA3    = malloc(nmax * sizeof(f64));
    g_ws->EVEIGS = malloc(nmax * sizeof(f64));
    g_ws->TAU    = malloc(nmax * sizeof(f64));

    g_ws->work  = malloc(g_ws->lwork * sizeof(f64));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(INT));

    if (!g_ws->A || !g_ws->U || !g_ws->V || !g_ws->Z ||
        !g_ws->D1 || !g_ws->D2 || !g_ws->D3 || !g_ws->D4 ||
        !g_ws->WA1 || !g_ws->WA2 || !g_ws->WA3 || !g_ws->EVEIGS ||
        !g_ws->TAU || !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xA37B1C924E68F05DULL);
    rng_seed(g_ws->rng_state2, 0xA37B1C924E68F05DULL);
    rng_seed(g_ws->rng_state3, 0xA37B1C924E68F05DULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->U);
        free(g_ws->V);
        free(g_ws->Z);
        free(g_ws->D1);
        free(g_ws->D2);
        free(g_ws->D3);
        free(g_ws->D4);
        free(g_ws->WA1);
        free(g_ws->WA2);
        free(g_ws->WA3);
        free(g_ws->EVEIGS);
        free(g_ws->TAU);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/*
 * Generate test matrix according to jtype.
 * Port of ddrvst.f lines 647-753.
 */
static INT generate_matrix(INT n, INT jtype, f64* A, INT lda,
                           f64* U, INT ldu, f64* work, INT* iwork,
                           uint64_t state[static 4],
                           uint64_t state3[static 4],
                           INT* ihbw_out)
{
    (void)ldu;
    INT itype = KTYPE[jtype - 1];
    INT imode = KMODE[jtype - 1];
    f64 anorm, cond;
    INT iinfo = 0;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);
    f64 rtovfl = sqrt(ovfl);
    f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0;
    }

    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    iinfo = 0;
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 4) {
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 7) {
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        INT ihbw = (INT)((n - 1) * rng_uniform(state3));
        if (ihbw_out) *ihbw_out = ihbw;

        INT ldu_band = 2 * ihbw + 1;
        if (ldu_band < 1) ldu_band = 1;
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               ihbw, ihbw, "Z", U, ldu_band, work + n, &iinfo, state);

        dlaset("F", lda, n, 0.0, 0.0, A, lda);
        for (INT idiag = -ihbw; idiag <= ihbw; idiag++) {
            INT irow = ihbw - idiag;
            INT j1 = (idiag > 0) ? idiag : 0;
            INT j2 = (n + idiag < n) ? n + idiag : n;
            for (INT j = j1; j < j2; j++) {
                INT i = j - idiag;
                A[i + j * lda] = U[irow + j * ldu_band];
            }
        }
    } else {
        iinfo = 1;
    }

    return iinfo;
}

/*
 * Compute VL, VU from eigenvalue array for RANGE='V' tests.
 * Port of ddrvst.f lines 1056-1076, 1282-1302, etc.
 * wa1 must contain sorted eigenvalues, il/iu are 0-based.
 */
static void compute_vl_vu(INT n, INT il, INT iu, const f64* wa1,
                          f64 temp3, f64 ulp, f64 rtunfl,
                          f64* vl_out, f64* vu_out)
{
    if (n > 0) {
        if (il != 0) {
            *vl_out = wa1[il] - fmax(0.5 * (wa1[il] - wa1[il - 1]),
                                     fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
        } else {
            *vl_out = wa1[0] - fmax(0.5 * (wa1[n - 1] - wa1[0]),
                                    fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
        }
        if (iu != n - 1) {
            *vu_out = wa1[iu] + fmax(0.5 * (wa1[iu + 1] - wa1[iu]),
                                     fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
        } else {
            *vu_out = wa1[n - 1] + fmax(0.5 * (wa1[n - 1] - wa1[0]),
                                        fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
        }
    } else {
        *vl_out = 0.0;
        *vu_out = 1.0;
    }
}

/*
 * Pack upper or lower triangular part of A into packed storage in work.
 * Returns the next free index (indx) in work.
 */
static INT pack_triangular(INT n, INT iuplo, const f64* A, INT lda, f64* work)
{
    INT indx = 0;
    if (iuplo == 1) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i <= j; i++) {
                work[indx] = A[i + j * lda];
                indx++;
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            for (INT i = j; i < n; i++) {
                work[indx] = A[i + j * lda];
                indx++;
            }
        }
    }
    return indx;
}

/*
 * Load band storage from dense matrix A.
 */
static void load_band(INT n, INT kd, INT iuplo, const f64* A, INT lda,
                      f64* V, INT ldv)
{
    if (iuplo == 1) {
        for (INT j = 0; j < n; j++) {
            INT imin = (j - kd > 0) ? j - kd : 0;
            for (INT i = imin; i <= j; i++) {
                V[(kd + i - j) + j * ldv] = A[i + j * lda];
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            INT imax = (j + kd < n - 1) ? j + kd : n - 1;
            for (INT i = j; i <= imax; i++) {
                V[(i - j) + j * ldv] = A[i + j * lda];
            }
        }
    }
}

/*
 * Run all tests for a single (n, jtype) combination.
 * Port of ddrvst.f lines 618-1736.
 */
static void run_ddrvst2stg_single(ddrvst2stg_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;

    ddrvst2stg_workspace_t* ws = g_ws;
    INT nmax = ws->nmax;
    INT lda = nmax;
    INT ldu = nmax;

    f64* A = ws->A;
    f64* U = ws->U;
    f64* V = ws->V;
    f64* Z = ws->Z;
    f64* D1 = ws->D1;
    f64* D2 = ws->D2;
    f64* D3 = ws->D3;
    f64* D4 = ws->D4;
    f64* WA1 = ws->WA1;
    f64* WA2 = ws->WA2;
    f64* WA3 = ws->WA3;
    f64* EVEIGS = ws->EVEIGS;
    f64* TAU = ws->TAU;
    f64* work = ws->work;
    INT* iwork = ws->iwork;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    // f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);

    INT iinfo;
    f64 temp1, temp2, temp3 = 0.0;
    INT ntest;
    INT m, m2, m3;
    f64 vl = 0.0, vu = 0.0;
    INT indx;

    /* Per-N workspace sizes (ddrvst.f lines 596-609) */
    INT lgn = 0, lwedc, liwedc;
    if (n > 0) {
        lgn = (INT)(log((f64)n) / log(2.0));
        if ((1 << lgn) < n) lgn++;
        if ((1 << lgn) < n) lgn++;
        lwedc = 1 + 4 * n + 2 * n * lgn + 4 * n * n;
        liwedc = 3 + 5 * n;
    } else {
        lwedc = 9;
        liwedc = 8;
    }

    f64 abstol = unfl + unfl;

    /* Initialize all results to zero */
    for (INT j = 0; j < 140; j++) {
        ws->result[j] = 0.0;
    }

    if (n == 0) return;

    /* Compute IL, IU (0-based) — ddrvst.f lines 758-769 */
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

    INT ihbw = 0;

    /* Generate matrix — ddrvst.f lines 647-753 */
    if (jtype <= MAXTYP) {
        iinfo = generate_matrix(n, jtype, A, lda, U, ldu, work, iwork,
                                ws->rng_state, ws->rng_state3, &ihbw);
        if (iinfo != 0) {
            print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                          jtype, n, iinfo);
            ws->result[0] = ulpinv;
            assert_info_success(iinfo);
            return;
        }
    }

    /* 3) If matrix is tridiagonal, call DSTEV and DSTEVX — ddrvst.f line 773 */
    if (jtype <= 7) {
        ntest = 1;
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        /* DSTEV('V') — tests 1-2 */
        dstev("V", n, D1, D2, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEV(V) returned info=%d, n=%d, jtype=%d\n", iinfo, n, jtype);
            ws->result[0] = ulpinv;
            ws->result[1] = ulpinv;
            ws->result[2] = ulpinv;
            goto L180;
        }

        /* Do tests 1 and 2 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt21(n, 0, D3, D4, D1, D2, Z, ldu, work, ws->result);

        ntest = 3;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEV('N') — test 3 */
        dstev("N", n, D3, D4, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEV(N) returned info=%d\n", iinfo);
            ws->result[2] = ulpinv;
            goto L180;
        }

        /* Do test 3 */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[2] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L180:
        ntest = 4;
        for (INT i = 0; i < n; i++) {
            EVEIGS[i] = D3[i];
            D1[i] = A[i + i * lda];
        }
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        /* DSTEVX('V','A') — tests 4-5 */
        dstevx("V", "A", n, D1, D2, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(V,A) returned info=%d\n", iinfo);
            ws->result[3] = ulpinv;
            ws->result[4] = ulpinv;
            ws->result[5] = ulpinv;
            goto L250;
        }
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;

        /* Do tests 4 and 5 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt21(n, 0, D3, D4, WA1, D2, Z, ldu, work, ws->result + 3);

        ntest = 6;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVX('N','A') — test 6 */
        dstevx("N", "A", n, D3, D4, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(N,A) returned info=%d\n", iinfo);
            ws->result[5] = ulpinv;
            goto L250;
        }

        /* Do test 6 — compare WA2 vs EVEIGS */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA2[j]), fabs(EVEIGS[j])));
            temp2 = fmax(temp2, fabs(WA2[j] - EVEIGS[j]));
        }
        ws->result[5] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L250:
        /* DSTEVR('V','A') — tests 7-8 (ddrvst.f line 915) */
        ntest = 7;
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevr("V", "A", n, D1, D2, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(V,A) returned info=%d\n", iinfo);
            ws->result[6] = ulpinv;
            ws->result[7] = ulpinv;
            ws->result[8] = ulpinv;
            goto L320;
        }
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;

        /* Do tests 7 and 8 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt21(n, 0, D3, D4, WA1, D2, Z, ldu, work, ws->result + 6);

        ntest = 9;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVR('N','A') — test 9 */
        dstevr("N", "A", n, D3, D4, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(N,A) returned info=%d\n", iinfo);
            ws->result[8] = ulpinv;
            goto L320;
        }

        /* Do test 9 — compare WA2 vs EVEIGS */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA2[j]), fabs(EVEIGS[j])));
            temp2 = fmax(temp2, fabs(WA2[j] - EVEIGS[j]));
        }
        ws->result[8] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L320:
        /* DSTEVX('V','I') — tests 10-11 (ddrvst.f line 990) */
        ntest = 10;
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevx("V", "I", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(V,I) returned info=%d\n", iinfo);
            ws->result[9] = ulpinv;
            ws->result[10] = ulpinv;
            ws->result[11] = ulpinv;
            goto L380;
        }

        /* Do tests 10 and 11 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 9);

        ntest = 12;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVX('N','I') — test 12 */
        dstevx("N", "I", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(N,I) returned info=%d\n", iinfo);
            ws->result[11] = ulpinv;
            goto L380;
        }

        /* Do test 12 */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[11] = (temp1 + temp2) / fmax(unfl, ulp * temp3);

L380:
        /* Compute VL, VU for RANGE='V' tests (ddrvst.f lines 1056-1076) */
        ntest = 12;
        compute_vl_vu(n, il, iu, WA1, temp3, ulp, rtunfl, &vl, &vu);

        /* DSTEVX('V','V') — tests 13-14 (ddrvst.f line 1085) */
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevx("V", "V", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(V,V) returned info=%d\n", iinfo);
            ws->result[12] = ulpinv;
            ws->result[13] = ulpinv;
            ws->result[14] = ulpinv;
            goto L440;
        }

        if (m2 == 0 && n > 0) {
            ws->result[12] = ulpinv;
            ws->result[13] = ulpinv;
            ws->result[14] = ulpinv;
            goto L440;
        }

        /* Do tests 13 and 14 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 12);

        ntest = 15;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVX('N','V') — test 15 */
        dstevx("N", "V", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVX(N,V) returned info=%d\n", iinfo);
            ws->result[14] = ulpinv;
            goto L440;
        }

        /* Do test 15 */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[14] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L440:
        /* DSTEVD('V') — tests 16-17 (ddrvst.f line 1148) */
        ntest = 16;
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevd("V", n, D1, D2, Z, ldu, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVD(V) returned info=%d\n", iinfo);
            ws->result[15] = ulpinv;
            ws->result[16] = ulpinv;
            ws->result[17] = ulpinv;
            goto L510;
        }

        /* Do tests 16 and 17 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt21(n, 0, D3, D4, D1, D2, Z, ldu, work, ws->result + 15);

        ntest = 18;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVD('N') — test 18 */
        dstevd("N", n, D3, D4, Z, ldu, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVD(N) returned info=%d\n", iinfo);
            ws->result[17] = ulpinv;
            goto L510;
        }

        /* Do test 18 — compare D3 vs EVEIGS */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(EVEIGS[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(EVEIGS[j] - D3[j]));
        }
        ws->result[17] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L510:
        /* DSTEVR('V','I') — tests 19-20 (ddrvst.f line 1216) */
        ntest = 19;
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevr("V", "I", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(V,I) returned info=%d\n", iinfo);
            ws->result[18] = ulpinv;
            ws->result[19] = ulpinv;
            ws->result[20] = ulpinv;
            goto L570;
        }

        /* Do tests 19 and 20 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 18);

        ntest = 21;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVR('N','I') — test 21 */
        dstevr("N", "I", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(N,I) returned info=%d\n", iinfo);
            ws->result[20] = ulpinv;
            goto L570;
        }

        /* Do test 21 */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[20] = (temp1 + temp2) / fmax(unfl, ulp * temp3);

L570:
        /* Recompute VL, VU for DSTEVR RANGE='V' (ddrvst.f lines 1282-1302) */
        ntest = 21;
        compute_vl_vu(n, il, iu, WA1, temp3, ulp, rtunfl, &vl, &vu);

        /* DSTEVR('V','V') — tests 22-23 (ddrvst.f line 1311) */
        for (INT i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        dstevr("V", "V", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(V,V) returned info=%d\n", iinfo);
            ws->result[21] = ulpinv;
            ws->result[22] = ulpinv;
            ws->result[23] = ulpinv;
            goto L630;
        }

        if (m2 == 0 && n > 0) {
            ws->result[21] = ulpinv;
            ws->result[22] = ulpinv;
            ws->result[23] = ulpinv;
            goto L630;
        }

        /* Do tests 22 and 23 */
        for (INT i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        dstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 21);

        ntest = 24;
        for (INT i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* DSTEVR('N','V') — test 24 */
        dstevr("N", "V", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSTEVR(N,V) returned info=%d\n", iinfo);
            ws->result[23] = ulpinv;
            goto L630;
        }

        /* Do test 24 */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[23] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L630:
        ;

    } else {
        /* JTYPE > 7: set tridiagonal results to zero (ddrvst.f line 1378) */
        for (INT i = 0; i < 24; i++)
            ws->result[i] = 0.0;
        ntest = 24;
    }

    /*
     * Perform remaining tests storing upper or lower triangular
     * part of matrix. (ddrvst.f line 1387)
     */
    for (INT iuplo = 0; iuplo <= 1; iuplo++) {
        const char* uplo = (iuplo == 0) ? "L" : "U";

        /* 4) Call DSYEV and DSYEVX (ddrvst.f line 1394) */
        dlacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        /* DSYEV('V', UPLO) */
        dsyev("V", uplo, n, A, ldu, D1, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L660;
        }

        /* Do tests 25 and 26 (or +54) */
        dsyt21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z, ldu,
               TAU, work, ws->result + ntest - 1);

        dlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        /* DSYEV_2STAGE('N', UPLO) */
        dsyev_2stage("N", uplo, n, A, ldu, D3, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEV_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L660;
        }

        /* Do test 27 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L660:
        dlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 1;

        /* Compute VL, VU, temp3 for DSYEVX (ddrvst.f lines 1455-1475) */
        if (n > 0) {
            temp3 = fmax(fabs(D1[0]), fabs(D1[n - 1]));
            if (il != 0) {
                vl = D1[il] - fmax(0.5 * (D1[il] - D1[il - 1]),
                                   fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            } else {
                vl = D1[0] - fmax(0.5 * (D1[n - 1] - D1[0]),
                                  fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            }
            if (iu != n - 1) {
                vu = D1[iu] + fmax(0.5 * (D1[iu + 1] - D1[iu]),
                                   fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            } else {
                vu = D1[n - 1] + fmax(0.5 * (D1[n - 1] - D1[0]),
                                      fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            }
        } else {
            temp3 = 0.0;
            vl = 0.0;
            vu = 1.0;
        }

        /* DSYEVX('V','A', UPLO) — tests 28-29 (ddrvst.f line 1478) */
        dsyevx("V", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L680;
        }

        /* Do tests 28 and 29 (or +54) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt21(1, uplo, n, 0, A, ldu, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* DSYEVX_2STAGE('N','A', UPLO) */
        dsyevx_2stage("N", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L680;
        }

        /* Do test 30 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L680:
        /* DSYEVX('V','I', UPLO) — tests 31-32 (ddrvst.f line 1534) */
        ntest = ntest + 1;
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyevx("V", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L690;
        }

        /* Do tests 31 and 32 (or +54) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        dlacpy(" ", n, n, V, ldu, A, lda);
        /* DSYEVX_2STAGE('N','I', UPLO) */
        dsyevx_2stage("N", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L690;
        }

        /* Do test 33 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, ulp * temp3);

L690:
        /* DSYEVX('V','V', UPLO) — tests 34-35 (ddrvst.f line 1587) */
        ntest = ntest + 1;
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyevx("V", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L700;
        }

        /* Do tests 34 and 35 (or +54) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        dlacpy(" ", n, n, V, ldu, A, lda);
        /* DSYEVX_2STAGE('N','V', UPLO) */
        dsyevx_2stage("N", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVX_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L700;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L700;
        }

        /* Do test 36 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L700:
        /* 5) Call DSPEV and DSPEVX (ddrvst.f line 1649) */
        dlacpy(" ", n, n, V, ldu, A, lda);

        /* Load packed storage */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        /* DSPEV('V', UPLO) */
        dspev("V", uplo, n, work, D1, Z, ldu, V, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L800;
        }

        /* Do tests 37 and 38 (or +54) */
        dsyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        /* Reload packed form */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        /* DSPEV('N', UPLO) */
        dspev("N", uplo, n, work, D3, Z, ldu, V, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEV(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L800;
        }

        /* Do test 39 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L800:
        /* Load packed form for DSPEVX */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;

        /* Compute VL, VU, temp3 for DSPEVX (ddrvst.f lines 1764-1784) */
        if (n > 0) {
            temp3 = fmax(fabs(D1[0]), fabs(D1[n - 1]));
            if (il != 0) {
                vl = D1[il] - fmax(0.5 * (D1[il] - D1[il - 1]),
                                   fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            } else {
                vl = D1[0] - fmax(0.5 * (D1[n - 1] - D1[0]),
                                  fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            }
            if (iu != n - 1) {
                vu = D1[iu] + fmax(0.5 * (D1[iu + 1] - D1[iu]),
                                   fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            } else {
                vu = D1[n - 1] + fmax(0.5 * (D1[n - 1] - D1[0]),
                                      fmax(10.0 * ulp * temp3, 10.0 * rtunfl));
            }
        } else {
            temp3 = 0.0;
            vl = 0.0;
            vu = 1.0;
        }

        /* DSPEVX('V','A', UPLO) — tests 40-41 (ddrvst.f line 1787) */
        dspevx("V", "A", uplo, n, work, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L900;
        }

        /* Do tests 40 and 41 (or +54) */
        dsyt21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        /* Reload packed form */
        indx = pack_triangular(n, iuplo, A, lda, work);

        /* DSPEVX('N','A', UPLO) */
        dspevx("N", "A", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L900;
        }

        /* Do test 42 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L900:
        /* DSPEVX('V','I', UPLO) — tests 43-44 (ddrvst.f line 1878) */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        dspevx("V", "I", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L990;
        }

        /* Do tests 43 and 44 (or +54) */
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        indx = pack_triangular(n, iuplo, A, lda, work);

        /* DSPEVX('N','I', UPLO) */
        dspevx("N", "I", uplo, n, work, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L990;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L990;
        }

        /* Do test 45 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L990:
        /* DSPEVX('V','V', UPLO) — tests 46-47 (ddrvst.f line 1975) */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        dspevx("V", "V", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1080;
        }

        /* Do tests 46 and 47 (or +54) */
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        indx = pack_triangular(n, iuplo, A, lda, work);

        /* DSPEVX('N','V', UPLO) */
        dspevx("N", "V", uplo, n, work, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVX(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1080;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1080;
        }

        /* Do test 48 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L1080:
        /* 6) Call DSBEV and DSBEVX (ddrvst.f line 2052) */
        ;
        INT kd;
        if (jtype <= 7) {
            kd = 1;
        } else if (jtype >= 8 && jtype <= 15) {
            kd = (n - 1 > 0) ? n - 1 : 0;
        } else {
            kd = ihbw;
        }

        /* Load band storage */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        /* DSBEV('V', UPLO) */
        dsbev("V", uplo, n, kd, V, ldu, D1, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1180;
        }

        /* Do tests 49 and 50 (or ... ) */
        dsyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        /* Reload band */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        /* DSBEV_2STAGE('N', UPLO) */
        dsbev_2stage("N", uplo, n, kd, V, ldu, D3, Z, ldu, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEV_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1180;
        }

        /* Do test 51 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1180:
        /* DSBEVX('V','A', UPLO) — tests 52-53 (ddrvst.f line 2163) */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        dsbevx("V", "A", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1280;
        }

        /* Do tests 52 and 53 (or +54) */
        dsyt21(1, uplo, n, 0, A, ldu, WA2, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* DSBEVX_2STAGE('N','A', UPLO) */
        dsbevx_2stage("N", "A", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, ws->lwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1280;
        }

        /* Do test 54 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA2[j]), fabs(WA3[j])));
            temp2 = fmax(temp2, fabs(WA2[j] - WA3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1280:
        /* DSBEVX('V','I', UPLO) — tests 55-56 (ddrvst.f line 2245) */
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        dsbevx("V", "I", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1370;
        }

        /* Do tests 55 and 56 (or +54) */
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* DSBEVX_2STAGE('N','I', UPLO) */
        dsbevx_2stage("N", "I", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, ws->lwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1370;
        }

        /* Do test 57 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L1370:
        /* DSBEVX('V','V', UPLO) — tests 58-59 (ddrvst.f line 2328) */
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        dsbevx("V", "V", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1460;
        }

        /* Do tests 58 and 59 (or +54) */
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* DSBEVX_2STAGE('N','V', UPLO) */
        dsbevx_2stage("N", "V", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, ws->lwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVX_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1460;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1460;
        }

        /* Do test 60 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L1460:
        /* 7) Call DSYEVD (ddrvst.f line 2401) */
        dlacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        dsyevd("V", uplo, n, A, ldu, D1, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1480;
        }

        /* Do tests 61 and 62 (or +54) */
        dsyt21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z, ldu,
               TAU, work, ws->result + ntest - 1);

        dlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        dsyevd_2stage("N", uplo, n, A, ldu, D3, work, ws->lwork, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVD_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1480;
        }

        /* Do test 63 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1480:
        /* 8) Call DSPEVD (ddrvst.f line 2459) */
        dlacpy(" ", n, n, V, ldu, A, lda);

        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        dspevd("V", uplo, n, work, D1, Z, ldu,
               work + indx, lwedc - indx, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1580;
        }

        /* Do tests 64 and 65 (or +54) */
        dsyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        dspevd("N", uplo, n, work, D3, Z, ldu,
               work + indx, lwedc - indx, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSPEVD(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1580;
        }

        /* Do test 66 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1580:
        /* 9) Call DSBEVD (ddrvst.f line 2556) */
        ;
        if (jtype <= 7)
            kd = 1;
        else if (jtype >= 8 && jtype <= 15)
            kd = (n - 1 > 0) ? n - 1 : 0;
        else
            kd = ihbw;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        dsbevd("V", uplo, n, kd, V, ldu, D1, Z, ldu, work, lwedc,
               iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1680;
        }

        /* Do tests 67 and 68 (or +54) */
        dsyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        dsbevd_2stage("N", uplo, n, kd, V, ldu, D3, Z, ldu, work, ws->lwork,
               iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("DSBEVD_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1680;
        }

        /* Do test 69 (or +54) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1680:
        /* 10) Call DSYEVR (ddrvst.f line 2650) */
        dlacpy(" ", n, n, A, lda, V, ldu);
        ntest = ntest + 1;

        /* DSYEVR('V','A', UPLO) */
        dsyevr("V", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1700;
        }

        /* Do tests 70 and 71 (or ...) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* DSYEVR_2STAGE('N','A', UPLO) */
        dsyevr_2stage("N", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1700;
        }

        /* Do test 72 (or ...) */
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1700:
        /* DSYEVR('V','I', UPLO) — tests 73-74 (ddrvst.f line 2710) */
        ntest = ntest + 1;
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyevr("V", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1710;
        }

        /* Do tests 73 and 74 (or +54) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        dlacpy(" ", n, n, V, ldu, A, lda);
        /* DSYEVR_2STAGE('N','I', UPLO) */
        dsyevr_2stage("N", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1710;
        }

        /* Do test 75 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, ulp * temp3);

L1710:
        /* DSYEVR('V','V', UPLO) — tests 76-77 (ddrvst.f line 2763) */
        ntest = ntest + 1;
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyevr("V", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1750;
        }

        /* Do tests 76 and 77 (or +54) */
        dlacpy(" ", n, n, V, ldu, A, lda);
        dsyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        dlacpy(" ", n, n, V, ldu, A, lda);
        /* DSYEVR_2STAGE('N','V', UPLO) */
        dsyevr_2stage("N", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("DSYEVR_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1750;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1750;
        }

        /* Do test 78 (or +54) */
        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

        dlacpy(" ", n, n, V, ldu, A, lda);

L1750:
        ;
    } /* end IUPLO loop */

    /* Check results against threshold */
    for (INT j = 0; j < ntest; j++) {
        if (ws->result[j] >= THRESH) {
            print_message("  Test %d: ratio = %.6e (THRESH=%.1f) n=%d jtype=%d\n",
                          j + 1, ws->result[j], THRESH, n, jtype);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

static void test_ddrvst2stg_case(void** state)
{
    ddrvst2stg_params_t* params = *state;
    run_ddrvst2stg_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static ddrvst2stg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            ddrvst2stg_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "ddrvst2stg_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_ddrvst2stg_case;
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

    return _cmocka_run_group_tests("ddrvst2stg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
