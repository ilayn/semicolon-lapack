/**
 * @file test_sdrvst.c
 * @brief DDRVST checks the symmetric eigenvalue problem drivers.
 *
 * Port of LAPACK TESTING/EIG/ddrvst.f
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 50.0f
#define MAXTYP 18

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Tridiagonal eigenvalue routines */
extern void sstev(const char* jobz, const int n, f32* D, f32* E,
                  f32* Z, const int ldz, f32* work, int* info);
extern void sstevd(const char* jobz, const int n, f32* D, f32* E,
                   f32* Z, const int ldz, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void sstevx(const char* jobz, const char* range, const int n,
                   f32* D, f32* E, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol, int* m,
                   f32* W, f32* Z, const int ldz, f32* work,
                   int* iwork, int* ifail, int* info);
extern void sstevr(const char* jobz, const char* range, const int n,
                   f32* D, f32* E, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol, int* m,
                   f32* W, f32* Z, const int ldz, int* isuppz,
                   f32* work, const int lwork, int* iwork, const int liwork,
                   int* info);

/* Full symmetric eigenvalue routines */
extern void ssyev(const char* jobz, const char* uplo, const int n,
                  f32* A, const int lda, f32* W, f32* work,
                  const int lwork, int* info);
extern void ssyevd(const char* jobz, const char* uplo, const int n,
                   f32* A, const int lda, f32* W, f32* work,
                   const int lwork, int* iwork, const int liwork, int* info);
extern void ssyevx(const char* jobz, const char* range, const char* uplo,
                   const int n, f32* A, const int lda, const f32 vl,
                   const f32 vu, const int il, const int iu,
                   const f32 abstol, int* m, f32* W, f32* Z,
                   const int ldz, f32* work, const int lwork, int* iwork,
                   int* ifail, int* info);
extern void ssyevr(const char* jobz, const char* range, const char* uplo,
                   const int n, f32* A, const int lda, const f32 vl,
                   const f32 vu, const int il, const int iu,
                   const f32 abstol, int* m, f32* W, f32* Z,
                   const int ldz, int* isuppz, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);

/* Packed symmetric eigenvalue routines */
extern void sspev(const char* jobz, const char* uplo, const int n,
                  f32* AP, f32* W, f32* Z, const int ldz,
                  f32* work, int* info);
extern void sspevd(const char* jobz, const char* uplo, const int n,
                   f32* AP, f32* W, f32* Z, const int ldz,
                   f32* work, const int lwork, int* iwork, const int liwork,
                   int* info);
extern void sspevx(const char* jobz, const char* range, const char* uplo,
                   const int n, f32* AP, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol, int* m,
                   f32* W, f32* Z, const int ldz, f32* work, int* iwork,
                   int* ifail, int* info);

/* Band symmetric eigenvalue routines */
extern void ssbev(const char* jobz, const char* uplo, const int n,
                  const int kd, f32* AB, const int ldab, f32* W,
                  f32* Z, const int ldz, f32* work, int* info);
extern void ssbevd(const char* jobz, const char* uplo, const int n,
                   const int kd, f32* AB, const int ldab, f32* W,
                   f32* Z, const int ldz, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void ssbevx(const char* jobz, const char* range, const char* uplo,
                   const int n, const int kd, f32* AB, const int ldab,
                   f32* Q, const int ldq, const f32 vl, const f32 vu,
                   const int il, const int iu, const f32 abstol, int* m,
                   f32* W, f32* Z, const int ldz, f32* work,
                   int* iwork, int* ifail, int* info);

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
} ddrvst_params_t;

typedef struct {
    int nmax;

    f32* A;
    f32* U;
    f32* V;
    f32* Z;

    f32* D1;
    f32* D2;
    f32* D3;
    f32* D4;
    f32* WA1;
    f32* WA2;
    f32* WA3;
    f32* EVEIGS;
    f32* TAU;

    f32* work;
    int* iwork;
    int lwork;
    int liwork;

    f32 result[140];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
    uint64_t rng_state3[4];
} ddrvst_workspace_t;

static ddrvst_workspace_t* g_ws = NULL;

static const int KTYPE[MAXTYP]  = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9};
static const int KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3};
static const int KMODE[MAXTYP]  = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvst_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

    /* Workspace sizes from ddrvst.f lines 274-282, 596-609 */
    int lgn = 0;
    if (nmax > 0) {
        lgn = (int)(logf((f32)nmax) / logf(2.0f));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }
    /* LWORK: 1 + 4*Nmax + 2*Nmax*lgn + 4*Nmax**2 (line 602) */
    g_ws->lwork = 1 + 4 * nmax + 2 * nmax * lgn + 4 * n2;
    /* IWORK: 6 + 6*Nmax + 5*Nmax*lgNmax (line 281-282) */
    g_ws->liwork = 6 + 6 * nmax + 5 * nmax * lgn;

    g_ws->A = malloc(n2 * sizeof(f32));
    /* U: band storage needs up to LDU = max(2*nmax-1, nmax) rows */
    int ldu_band = (2 * nmax - 1) > nmax ? (2 * nmax - 1) : nmax;
    g_ws->U = malloc(ldu_band * nmax * sizeof(f32));
    g_ws->V = malloc(n2 * sizeof(f32));
    g_ws->Z = malloc(n2 * sizeof(f32));

    g_ws->D1     = malloc(nmax * sizeof(f32));
    g_ws->D2     = malloc(nmax * sizeof(f32));
    g_ws->D3     = malloc(nmax * sizeof(f32));
    g_ws->D4     = malloc(nmax * sizeof(f32));
    g_ws->WA1    = malloc(nmax * sizeof(f32));
    g_ws->WA2    = malloc(nmax * sizeof(f32));
    g_ws->WA3    = malloc(nmax * sizeof(f32));
    g_ws->EVEIGS = malloc(nmax * sizeof(f32));
    g_ws->TAU    = malloc(nmax * sizeof(f32));

    g_ws->work  = malloc(g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(int));

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
static int generate_matrix(int n, int jtype, f32* A, int lda,
                           f32* U, int ldu, f32* work, int* iwork,
                           uint64_t state[static 4],
                           uint64_t state3[static 4],
                           int* ihbw_out)
{
    (void)ldu;
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    f32 anorm, cond;
    int iinfo = 0;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);
    f32 rtovfl = sqrtf(ovfl);
    f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0f; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0f;
    }

    slaset("F", lda, n, 0.0f, 0.0f, A, lda);
    iinfo = 0;
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (int jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 4) {
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 7) {
        int idumma[1] = {1};
        slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, 0, 0, 0.0f, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        int idumma[1] = {1};
        slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, n, n, 0.0f, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        int ihbw = (int)((n - 1) * rng_uniform_f32(state3));
        if (ihbw_out) *ihbw_out = ihbw;

        int ldu_band = 2 * ihbw + 1;
        if (ldu_band < 1) ldu_band = 1;
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               ihbw, ihbw, "Z", U, ldu_band, work + n, &iinfo, state);

        slaset("F", lda, n, 0.0f, 0.0f, A, lda);
        for (int idiag = -ihbw; idiag <= ihbw; idiag++) {
            int irow = ihbw - idiag;
            int j1 = (idiag > 0) ? idiag : 0;
            int j2 = (n + idiag < n) ? n + idiag : n;
            for (int j = j1; j < j2; j++) {
                int i = j - idiag;
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
static void compute_vl_vu(int n, int il, int iu, const f32* wa1,
                          f32 temp3, f32 ulp, f32 rtunfl,
                          f32* vl_out, f32* vu_out)
{
    if (n > 0) {
        if (il != 0) {
            *vl_out = wa1[il] - fmaxf(0.5f * (wa1[il] - wa1[il - 1]),
                                     fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
        } else {
            *vl_out = wa1[0] - fmaxf(0.5f * (wa1[n - 1] - wa1[0]),
                                    fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
        }
        if (iu != n - 1) {
            *vu_out = wa1[iu] + fmaxf(0.5f * (wa1[iu + 1] - wa1[iu]),
                                     fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
        } else {
            *vu_out = wa1[n - 1] + fmaxf(0.5f * (wa1[n - 1] - wa1[0]),
                                        fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
        }
    } else {
        *vl_out = 0.0f;
        *vu_out = 1.0f;
    }
}

/*
 * Pack upper or lower triangular part of A into packed storage in work.
 * Returns the next free index (indx) in work.
 */
static int pack_triangular(int n, int iuplo, const f32* A, int lda, f32* work)
{
    int indx = 0;
    if (iuplo == 1) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                work[indx] = A[i + j * lda];
                indx++;
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
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
static void load_band(int n, int kd, int iuplo, const f32* A, int lda,
                      f32* V, int ldv)
{
    if (iuplo == 1) {
        for (int j = 0; j < n; j++) {
            int imin = (j - kd > 0) ? j - kd : 0;
            for (int i = imin; i <= j; i++) {
                V[(kd + i - j) + j * ldv] = A[i + j * lda];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            int imax = (j + kd < n - 1) ? j + kd : n - 1;
            for (int i = j; i <= imax; i++) {
                V[(i - j) + j * ldv] = A[i + j * lda];
            }
        }
    }
}

/*
 * Run all tests for a single (n, jtype) combination.
 * Port of ddrvst.f lines 618-1736.
 */
static void run_ddrvst_single(ddrvst_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;

    ddrvst_workspace_t* ws = g_ws;
    int nmax = ws->nmax;
    int lda = nmax;
    int ldu = nmax;

    f32* A = ws->A;
    f32* U = ws->U;
    f32* V = ws->V;
    f32* Z = ws->Z;
    f32* D1 = ws->D1;
    f32* D2 = ws->D2;
    f32* D3 = ws->D3;
    f32* D4 = ws->D4;
    f32* WA1 = ws->WA1;
    f32* WA2 = ws->WA2;
    f32* WA3 = ws->WA3;
    f32* EVEIGS = ws->EVEIGS;
    f32* TAU = ws->TAU;
    f32* work = ws->work;
    int* iwork = ws->iwork;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    // f64 ovfl = 1.0 / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);

    int iinfo;
    f32 temp1, temp2, temp3 = 0.0f;
    int ntest;
    int m, m2, m3;
    f32 vl = 0.0f, vu = 0.0f;
    int indx;

    /* Per-N workspace sizes (ddrvst.f lines 596-609) */
    int lgn = 0, lwedc, liwedc;
    if (n > 0) {
        lgn = (int)(logf((f32)n) / logf(2.0f));
        if ((1 << lgn) < n) lgn++;
        if ((1 << lgn) < n) lgn++;
        lwedc = 1 + 4 * n + 2 * n * lgn + 4 * n * n;
        liwedc = 3 + 5 * n;
    } else {
        lwedc = 9;
        liwedc = 8;
    }

    f32 abstol = unfl + unfl;

    /* Initialize all results to zero */
    for (int j = 0; j < 140; j++) {
        ws->result[j] = 0.0f;
    }

    if (n == 0) return;

    /* Compute IL, IU (0-based) — ddrvst.f lines 758-769 */
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

    int ihbw = 0;

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

    /* 3) If matrix is tridiagonal, call SSTEV and SSTEVX — ddrvst.f line 773 */
    if (jtype <= 7) {
        ntest = 1;
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        /* SSTEV('V') — tests 1-2 */
        sstev("V", n, D1, D2, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEV(V) returned info=%d, n=%d, jtype=%d\n", iinfo, n, jtype);
            ws->result[0] = ulpinv;
            ws->result[1] = ulpinv;
            ws->result[2] = ulpinv;
            goto L180;
        }

        /* Do tests 1 and 2 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt21(n, 0, D3, D4, D1, D2, Z, ldu, work, ws->result);

        ntest = 3;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEV('N') — test 3 */
        sstev("N", n, D3, D4, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEV(N) returned info=%d\n", iinfo);
            ws->result[2] = ulpinv;
            goto L180;
        }

        /* Do test 3 */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[2] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L180:
        ntest = 4;
        for (int i = 0; i < n; i++) {
            EVEIGS[i] = D3[i];
            D1[i] = A[i + i * lda];
        }
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        /* SSTEVX('V','A') — tests 4-5 */
        sstevx("V", "A", n, D1, D2, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(V,A) returned info=%d\n", iinfo);
            ws->result[3] = ulpinv;
            ws->result[4] = ulpinv;
            ws->result[5] = ulpinv;
            goto L250;
        }
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;

        /* Do tests 4 and 5 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt21(n, 0, D3, D4, WA1, D2, Z, ldu, work, ws->result + 3);

        ntest = 6;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVX('N','A') — test 6 */
        sstevx("N", "A", n, D3, D4, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(N,A) returned info=%d\n", iinfo);
            ws->result[5] = ulpinv;
            goto L250;
        }

        /* Do test 6 — compare WA2 vs EVEIGS */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA2[j]), fabsf(EVEIGS[j])));
            temp2 = fmaxf(temp2, fabsf(WA2[j] - EVEIGS[j]));
        }
        ws->result[5] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L250:
        /* SSTEVR('V','A') — tests 7-8 (ddrvst.f line 915) */
        ntest = 7;
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevr("V", "A", n, D1, D2, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(V,A) returned info=%d\n", iinfo);
            ws->result[6] = ulpinv;
            ws->result[7] = ulpinv;
            ws->result[8] = ulpinv;
            goto L320;
        }
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;

        /* Do tests 7 and 8 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt21(n, 0, D3, D4, WA1, D2, Z, ldu, work, ws->result + 6);

        ntest = 9;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVR('N','A') — test 9 */
        sstevr("N", "A", n, D3, D4, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(N,A) returned info=%d\n", iinfo);
            ws->result[8] = ulpinv;
            goto L320;
        }

        /* Do test 9 — compare WA2 vs EVEIGS */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA2[j]), fabsf(EVEIGS[j])));
            temp2 = fmaxf(temp2, fabsf(WA2[j] - EVEIGS[j]));
        }
        ws->result[8] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L320:
        /* SSTEVX('V','I') — tests 10-11 (ddrvst.f line 990) */
        ntest = 10;
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevx("V", "I", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(V,I) returned info=%d\n", iinfo);
            ws->result[9] = ulpinv;
            ws->result[10] = ulpinv;
            ws->result[11] = ulpinv;
            goto L380;
        }

        /* Do tests 10 and 11 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 9);

        ntest = 12;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVX('N','I') — test 12 */
        sstevx("N", "I", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(N,I) returned info=%d\n", iinfo);
            ws->result[11] = ulpinv;
            goto L380;
        }

        /* Do test 12 */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[11] = (temp1 + temp2) / fmaxf(unfl, ulp * temp3);

L380:
        /* Compute VL, VU for RANGE='V' tests (ddrvst.f lines 1056-1076) */
        ntest = 12;
        compute_vl_vu(n, il, iu, WA1, temp3, ulp, rtunfl, &vl, &vu);

        /* SSTEVX('V','V') — tests 13-14 (ddrvst.f line 1085) */
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevx("V", "V", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(V,V) returned info=%d\n", iinfo);
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
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 12);

        ntest = 15;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVX('N','V') — test 15 */
        sstevx("N", "V", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVX(N,V) returned info=%d\n", iinfo);
            ws->result[14] = ulpinv;
            goto L440;
        }

        /* Do test 15 */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[14] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L440:
        /* SSTEVD('V') — tests 16-17 (ddrvst.f line 1148) */
        ntest = 16;
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevd("V", n, D1, D2, Z, ldu, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVD(V) returned info=%d\n", iinfo);
            ws->result[15] = ulpinv;
            ws->result[16] = ulpinv;
            ws->result[17] = ulpinv;
            goto L510;
        }

        /* Do tests 16 and 17 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt21(n, 0, D3, D4, D1, D2, Z, ldu, work, ws->result + 15);

        ntest = 18;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVD('N') — test 18 */
        sstevd("N", n, D3, D4, Z, ldu, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVD(N) returned info=%d\n", iinfo);
            ws->result[17] = ulpinv;
            goto L510;
        }

        /* Do test 18 — compare D3 vs EVEIGS */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(EVEIGS[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(EVEIGS[j] - D3[j]));
        }
        ws->result[17] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L510:
        /* SSTEVR('V','I') — tests 19-20 (ddrvst.f line 1216) */
        ntest = 19;
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevr("V", "I", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(V,I) returned info=%d\n", iinfo);
            ws->result[18] = ulpinv;
            ws->result[19] = ulpinv;
            ws->result[20] = ulpinv;
            goto L570;
        }

        /* Do tests 19 and 20 */
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 18);

        ntest = 21;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVR('N','I') — test 21 */
        sstevr("N", "I", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(N,I) returned info=%d\n", iinfo);
            ws->result[20] = ulpinv;
            goto L570;
        }

        /* Do test 21 */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[20] = (temp1 + temp2) / fmaxf(unfl, ulp * temp3);

L570:
        /* Recompute VL, VU for SSTEVR RANGE='V' (ddrvst.f lines 1282-1302) */
        ntest = 21;
        compute_vl_vu(n, il, iu, WA1, temp3, ulp, rtunfl, &vl, &vu);

        /* SSTEVR('V','V') — tests 22-23 (ddrvst.f line 1311) */
        for (int i = 0; i < n; i++)
            D1[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D2[i] = A[(i + 1) + i * lda];

        sstevr("V", "V", n, D1, D2, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(V,V) returned info=%d\n", iinfo);
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
        for (int i = 0; i < n; i++)
            D3[i] = A[i + i * lda];
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];
        sstt22(n, m2, 0, D3, D4, WA2, D2, Z, ldu, work,
               m2 > 1 ? m2 : 1, ws->result + 21);

        ntest = 24;
        for (int i = 0; i < n - 1; i++)
            D4[i] = A[(i + 1) + i * lda];

        /* SSTEVR('N','V') — test 24 */
        sstevr("N", "V", n, D3, D4, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEVR(N,V) returned info=%d\n", iinfo);
            ws->result[23] = ulpinv;
            goto L630;
        }

        /* Do test 24 */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[23] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L630:
        ;

    } else {
        /* JTYPE > 7: set tridiagonal results to zero (ddrvst.f line 1378) */
        for (int i = 0; i < 24; i++)
            ws->result[i] = 0.0f;
        ntest = 24;
    }

    /*
     * Perform remaining tests storing upper or lower triangular
     * part of matrix. (ddrvst.f line 1387)
     */
    for (int iuplo = 0; iuplo <= 1; iuplo++) {
        const char* uplo = (iuplo == 0) ? "L" : "U";

        /* 4) Call SSYEV and SSYEVX (ddrvst.f line 1394) */
        slacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        /* SSYEV('V', UPLO) */
        ssyev("V", uplo, n, A, ldu, D1, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L660;
        }

        /* Do tests 25 and 26 (or +54) */
        ssyt21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z, ldu,
               TAU, work, ws->result + ntest - 1);

        slacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        /* SSYEV('N', UPLO) */
        ssyev("N", uplo, n, A, ldu, D3, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEV(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L660;
        }

        /* Do test 27 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L660:
        slacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 1;

        /* Compute VL, VU, temp3 for SSYEVX (ddrvst.f lines 1455-1475) */
        if (n > 0) {
            temp3 = fmaxf(fabsf(D1[0]), fabsf(D1[n - 1]));
            if (il != 0) {
                vl = D1[il] - fmaxf(0.5f * (D1[il] - D1[il - 1]),
                                   fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            } else {
                vl = D1[0] - fmaxf(0.5f * (D1[n - 1] - D1[0]),
                                  fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            }
            if (iu != n - 1) {
                vu = D1[iu] + fmaxf(0.5f * (D1[iu + 1] - D1[iu]),
                                   fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            } else {
                vu = D1[n - 1] + fmaxf(0.5f * (D1[n - 1] - D1[0]),
                                      fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            }
        } else {
            temp3 = 0.0f;
            vl = 0.0f;
            vu = 1.0f;
        }

        /* SSYEVX('V','A', UPLO) — tests 28-29 (ddrvst.f line 1478) */
        ssyevx("V", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L680;
        }

        /* Do tests 28 and 29 (or +54) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt21(1, uplo, n, 0, A, ldu, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* SSYEVX('N','A', UPLO) */
        ssyevx("N", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L680;
        }

        /* Do test 30 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA1[j]), fabsf(WA2[j])));
            temp2 = fmaxf(temp2, fabsf(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L680:
        /* SSYEVX('V','I', UPLO) — tests 31-32 (ddrvst.f line 1534) */
        ntest = ntest + 1;
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyevx("V", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L690;
        }

        /* Do tests 31 and 32 (or +54) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        slacpy(" ", n, n, V, ldu, A, lda);
        /* SSYEVX('N','I', UPLO) */
        ssyevx("N", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L690;
        }

        /* Do test 33 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, ulp * temp3);

L690:
        /* SSYEVX('V','V', UPLO) — tests 34-35 (ddrvst.f line 1587) */
        ntest = ntest + 1;
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyevx("V", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L700;
        }

        /* Do tests 34 and 35 (or +54) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        slacpy(" ", n, n, V, ldu, A, lda);
        /* SSYEVX('N','V', UPLO) */
        ssyevx("N", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, work, ws->lwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVX(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L700;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L700;
        }

        /* Do test 36 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L700:
        /* 5) Call SSPEV and SSPEVX (ddrvst.f line 1649) */
        slacpy(" ", n, n, V, ldu, A, lda);

        /* Load packed storage */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        /* SSPEV('V', UPLO) */
        sspev("V", uplo, n, work, D1, Z, ldu, V, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L800;
        }

        /* Do tests 37 and 38 (or +54) */
        ssyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        /* Reload packed form */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        /* SSPEV('N', UPLO) */
        sspev("N", uplo, n, work, D3, Z, ldu, V, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEV(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L800;
        }

        /* Do test 39 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L800:
        /* Load packed form for SSPEVX */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;

        /* Compute VL, VU, temp3 for SSPEVX (ddrvst.f lines 1764-1784) */
        if (n > 0) {
            temp3 = fmaxf(fabsf(D1[0]), fabsf(D1[n - 1]));
            if (il != 0) {
                vl = D1[il] - fmaxf(0.5f * (D1[il] - D1[il - 1]),
                                   fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            } else {
                vl = D1[0] - fmaxf(0.5f * (D1[n - 1] - D1[0]),
                                  fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            }
            if (iu != n - 1) {
                vu = D1[iu] + fmaxf(0.5f * (D1[iu + 1] - D1[iu]),
                                   fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            } else {
                vu = D1[n - 1] + fmaxf(0.5f * (D1[n - 1] - D1[0]),
                                      fmaxf(10.0f * ulp * temp3, 10.0f * rtunfl));
            }
        } else {
            temp3 = 0.0f;
            vl = 0.0f;
            vu = 1.0f;
        }

        /* SSPEVX('V','A', UPLO) — tests 40-41 (ddrvst.f line 1787) */
        sspevx("V", "A", uplo, n, work, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L900;
        }

        /* Do tests 40 and 41 (or +54) */
        ssyt21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        /* Reload packed form */
        indx = pack_triangular(n, iuplo, A, lda, work);

        /* SSPEVX('N','A', UPLO) */
        sspevx("N", "A", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L900;
        }

        /* Do test 42 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA1[j]), fabsf(WA2[j])));
            temp2 = fmaxf(temp2, fabsf(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L900:
        /* SSPEVX('V','I', UPLO) — tests 43-44 (ddrvst.f line 1878) */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        sspevx("V", "I", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L990;
        }

        /* Do tests 43 and 44 (or +54) */
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        indx = pack_triangular(n, iuplo, A, lda, work);

        /* SSPEVX('N','I', UPLO) */
        sspevx("N", "I", uplo, n, work, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L990;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L990;
        }

        /* Do test 45 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L990:
        /* SSPEVX('V','V', UPLO) — tests 46-47 (ddrvst.f line 1975) */
        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        sspevx("V", "V", uplo, n, work, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1080;
        }

        /* Do tests 46 and 47 (or +54) */
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        indx = pack_triangular(n, iuplo, A, lda, work);

        /* SSPEVX('N','V', UPLO) */
        sspevx("N", "V", uplo, n, work, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, V, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVX(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1080;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1080;
        }

        /* Do test 48 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L1080:
        /* 6) Call SSBEV and SSBEVX (ddrvst.f line 2052) */
        ;
        int kd;
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
        /* SSBEV('V', UPLO) */
        ssbev("V", uplo, n, kd, V, ldu, D1, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1180;
        }

        /* Do tests 49 and 50 (or ... ) */
        ssyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        /* Reload band */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        /* SSBEV('N', UPLO) */
        ssbev("N", uplo, n, kd, V, ldu, D3, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEV(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1180;
        }

        /* Do test 51 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1180:
        /* SSBEVX('V','A', UPLO) — tests 52-53 (ddrvst.f line 2163) */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        ssbevx("V", "A", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1280;
        }

        /* Do tests 52 and 53 (or +54) */
        ssyt21(1, uplo, n, 0, A, ldu, WA2, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* SSBEVX('N','A', UPLO) */
        ssbevx("N", "A", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1280;
        }

        /* Do test 54 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA2[j]), fabsf(WA3[j])));
            temp2 = fmaxf(temp2, fabsf(WA2[j] - WA3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1280:
        /* SSBEVX('V','I', UPLO) — tests 55-56 (ddrvst.f line 2245) */
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ssbevx("V", "I", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1370;
        }

        /* Do tests 55 and 56 (or +54) */
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* SSBEVX('N','I', UPLO) */
        ssbevx("N", "I", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1370;
        }

        /* Do test 57 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L1370:
        /* SSBEVX('V','V', UPLO) — tests 58-59 (ddrvst.f line 2328) */
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ssbevx("V", "V", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1460;
        }

        /* Do tests 58 and 59 (or +54) */
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* SSBEVX('N','V', UPLO) */
        ssbevx("N", "V", uplo, n, kd, V, ldu, U, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVX(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1460;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1460;
        }

        /* Do test 60 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

L1460:
        /* 7) Call SSYEVD (ddrvst.f line 2401) */
        slacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        ssyevd("V", uplo, n, A, ldu, D1, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1480;
        }

        /* Do tests 61 and 62 (or +54) */
        ssyt21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z, ldu,
               TAU, work, ws->result + ntest - 1);

        slacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        ssyevd("N", uplo, n, A, ldu, D3, work, lwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVD(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1480;
        }

        /* Do test 63 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1480:
        /* 8) Call SSPEVD (ddrvst.f line 2459) */
        slacpy(" ", n, n, V, ldu, A, lda);

        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        sspevd("V", uplo, n, work, D1, Z, ldu,
               work + indx, lwedc - indx, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1580;
        }

        /* Do tests 64 and 65 (or +54) */
        ssyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        indx = pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        sspevd("N", uplo, n, work, D3, Z, ldu,
               work + indx, lwedc - indx, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSPEVD(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1580;
        }

        /* Do test 66 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1580:
        /* 9) Call SSBEVD (ddrvst.f line 2556) */
        ;
        if (jtype <= 7)
            kd = 1;
        else if (jtype >= 8 && jtype <= 15)
            kd = (n - 1 > 0) ? n - 1 : 0;
        else
            kd = ihbw;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        ssbevd("V", uplo, n, kd, V, ldu, D1, Z, ldu, work, lwedc,
               iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1680;
        }

        /* Do tests 67 and 68 (or +54) */
        ssyt21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        ssbevd("N", uplo, n, kd, V, ldu, D3, Z, ldu, work, lwedc,
               iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            print_message("SSBEVD(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1680;
        }

        /* Do test 69 (or +54) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D3[j])));
            temp2 = fmaxf(temp2, fabsf(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1680:
        /* 10) Call SSYEVR (ddrvst.f line 2650) */
        slacpy(" ", n, n, A, lda, V, ldu);
        ntest = ntest + 1;

        /* SSYEVR('V','A', UPLO) */
        ssyevr("V", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m,
               WA1, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1700;
        }

        /* Do tests 70 and 71 (or ...) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V, ldu,
               TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* SSYEVR('N','A', UPLO) */
        ssyevr("N", "A", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1700;
        }

        /* Do test 72 (or ...) */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(WA1[j]), fabsf(WA2[j])));
            temp2 = fmaxf(temp2, fabsf(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));

L1700:
        /* SSYEVR('V','I', UPLO) — tests 73-74 (ddrvst.f line 2710) */
        ntest = ntest + 1;
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyevr("V", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1710;
        }

        /* Do tests 73 and 74 (or +54) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        slacpy(" ", n, n, V, ldu, A, lda);
        /* SSYEVR('N','I', UPLO) */
        ssyevr("N", "I", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1710;
        }

        /* Do test 75 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, ulp * temp3);

L1710:
        /* SSYEVR('V','V', UPLO) — tests 76-77 (ddrvst.f line 2763) */
        ntest = ntest + 1;
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyevr("V", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1750;
        }

        /* Do tests 76 and 77 (or +54) */
        slacpy(" ", n, n, V, ldu, A, lda);
        ssyt22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, ws->result + ntest - 1);

        ntest = ntest + 2;
        slacpy(" ", n, n, V, ldu, A, lda);
        /* SSYEVR('N','V', UPLO) */
        ssyevr("N", "V", uplo, n, A, ldu, vl, vu, il, iu, abstol, &m3,
               WA3, Z, ldu, iwork, work, ws->lwork,
               iwork + 2 * n, ws->liwork - 2 * n, &iinfo);
        if (iinfo != 0) {
            print_message("SSYEVR(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1750;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1750;
        }

        /* Do test 78 (or +54) */
        temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmaxf(fabsf(WA1[0]), fabsf(WA1[n - 1]));
        else
            temp3 = 0.0f;
        ws->result[ntest - 1] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

        slacpy(" ", n, n, V, ldu, A, lda);

L1750:
        ;
    } /* end IUPLO loop */

    /* Check results against threshold */
    for (int j = 0; j < ntest; j++) {
        if (ws->result[j] >= THRESH) {
            print_message("  Test %d: ratio = %.6e (THRESH=%.1f) n=%d jtype=%d\n",
                          j + 1, (double)ws->result[j], (double)THRESH, n, jtype);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

static void test_ddrvst_case(void** state)
{
    ddrvst_params_t* params = *state;
    run_ddrvst_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static ddrvst_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            ddrvst_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "ddrvst_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_ddrvst_case;
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

    return _cmocka_run_group_tests("ddrvst", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
