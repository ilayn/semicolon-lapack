/**
 * @file test_zdrvst2stg.c
 * @brief ZDRVST2STG checks the Hermitian eigenvalue problem drivers
 *        including the 2-stage variants.
 *
 * Port of LAPACK TESTING/EIG/zdrvst2stg.f
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#define THRESH 50.0
#define MAXTYP 18

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT n;
    INT jtype;
    char name[96];
} zdrvst2stg_params_t;

typedef struct {
    INT nmax;

    c128* A;
    c128* U;
    c128* V;
    c128* Z;
    c128* TAU;

    f64* D1;
    f64* D2;
    f64* D3;
    f64* WA1;
    f64* WA2;
    f64* WA3;

    c128* work;
    f64* rwork;
    INT* iwork;
    INT lwork;
    INT lrwork;
    INT liwork;

    f64 result[140];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
    uint64_t rng_state3[4];
} zdrvst2stg_workspace_t;

static zdrvst2stg_workspace_t* g_ws = NULL;

static const INT KTYPE[MAXTYP]  = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9};
static const INT KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3};
static const INT KMODE[MAXTYP]  = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvst2stg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    INT lgn = 0;
    if (nmax > 0) {
        lgn = (INT)(log((f64)nmax) / log(2.0));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
    }
    /* LWEDC: MAX(2*N+N*N, 2*N*N) (zdrvst2stg.f line 480) */
    INT lwedc1 = 2 * nmax + n2;
    INT lwedc2 = 2 * n2;
    g_ws->lwork = (lwedc1 > lwedc2) ? lwedc1 : lwedc2;
    /* LRWEDC: 1 + 4*N + 2*N*LGN + 3*N**2 (zdrvst2stg.f line 481) */
    g_ws->lrwork = 1 + 4 * nmax + 2 * nmax * lgn + 3 * n2;
    /* LIWEDC: 6 + 6*N + 5*N*LGN (zdrvst2stg.f line 486) */
    g_ws->liwork = 6 + 6 * nmax + 5 * nmax * lgn;

    /* 2-stage routines need additional workspace for internal band reduction.
     * Query the most demanding 2-stage routines to get the required sizes. */
    {
        c128 work_query;
        f64 rwork_query;
        INT iwork_query;
        INT info_query;
        INT lw2, lrw2;

        zheevd_2stage("N", "L", nmax, NULL, nmax, NULL,
                      &work_query, -1, &rwork_query, -1, &iwork_query, -1, &info_query);
        lw2 = (INT)creal(work_query);
        lrw2 = (INT)rwork_query;
        if (lw2 > g_ws->lwork) g_ws->lwork = lw2;
        if (lrw2 > g_ws->lrwork) g_ws->lrwork = lrw2;

        zheevr_2stage("N", "A", "L", nmax, NULL, nmax, 0.0, 0.0, 0, 0,
                      0.0, NULL, NULL, NULL, nmax, NULL,
                      &work_query, -1, &rwork_query, -1, &iwork_query, -1, &info_query);
        lw2 = (INT)creal(work_query);
        lrw2 = (INT)rwork_query;
        if (lw2 > g_ws->lwork) g_ws->lwork = lw2;
        if (lrw2 > g_ws->lrwork) g_ws->lrwork = lrw2;
    }

    g_ws->A   = malloc(n2 * sizeof(c128));
    INT ldu_band = (2 * nmax - 1) > nmax ? (2 * nmax - 1) : nmax;
    g_ws->U   = malloc(ldu_band * nmax * sizeof(c128));
    g_ws->V   = malloc(n2 * sizeof(c128));
    g_ws->Z   = malloc(n2 * sizeof(c128));
    g_ws->TAU = malloc(nmax * sizeof(c128));

    g_ws->D1  = malloc(nmax * sizeof(f64));
    g_ws->D2  = malloc(nmax * sizeof(f64));
    g_ws->D3  = malloc(nmax * sizeof(f64));
    g_ws->WA1 = malloc(nmax * sizeof(f64));
    g_ws->WA2 = malloc(nmax * sizeof(f64));
    g_ws->WA3 = malloc(nmax * sizeof(f64));

    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));
    g_ws->rwork = malloc(g_ws->lrwork * sizeof(f64));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(INT));

    if (!g_ws->A || !g_ws->U || !g_ws->V || !g_ws->Z || !g_ws->TAU ||
        !g_ws->D1 || !g_ws->D2 || !g_ws->D3 ||
        !g_ws->WA1 || !g_ws->WA2 || !g_ws->WA3 ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
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
        free(g_ws->TAU);
        free(g_ws->D1);
        free(g_ws->D2);
        free(g_ws->D3);
        free(g_ws->WA1);
        free(g_ws->WA2);
        free(g_ws->WA3);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static INT generate_matrix(INT n, INT jtype, c128* A, INT lda,
                           c128* U, INT ldu, c128* work, f64* rwork, INT* iwork,
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

    zlaset("F", lda, n, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), A, lda);
    iinfo = 0;
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = CMPLX(anorm, 0.0);
        }

    } else if (itype == 4) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work, &iinfo, state);

    } else if (itype == 5) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               n, n, "N", A, lda, work, &iinfo, state);

    } else if (itype == 7) {
        INT idumma[1] = {1};
        zlatmr(n, n, "S", "H", work, 6, CMPLX(1.0, 0.0), CMPLX(1.0, 0.0), "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        INT idumma[1] = {1};
        zlatmr(n, n, "S", "H", work, 6, CMPLX(1.0, 0.0), CMPLX(1.0, 0.0), "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        INT ihbw = (INT)((n - 1) * rng_uniform(state3));
        if (ihbw_out) *ihbw_out = ihbw;

        INT ldu_band = 2 * ihbw + 1;
        if (ldu_band < 1) ldu_band = 1;
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               ihbw, ihbw, "Z", U, ldu_band, work, &iinfo, state);

        zlaset("F", lda, n, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), A, lda);
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

static INT pack_triangular(INT n, INT iuplo, const c128* A, INT lda, c128* work)
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

static void load_band(INT n, INT kd, INT iuplo, const c128* A, INT lda,
                      c128* V, INT ldv)
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

static void run_zdrvst2stg_single(zdrvst2stg_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;

    zdrvst2stg_workspace_t* ws = g_ws;
    INT nmax = ws->nmax;
    INT lda = nmax;
    INT ldu = nmax;

    c128* A = ws->A;
    c128* U = ws->U;
    c128* V = ws->V;
    c128* Z = ws->Z;
    c128* TAU = ws->TAU;
    f64* D1 = ws->D1;
    f64* D2 = ws->D2;
    f64* D3 = ws->D3;
    f64* WA1 = ws->WA1;
    f64* WA2 = ws->WA2;
    f64* WA3 = ws->WA3;
    c128* work = ws->work;
    f64* rwork = ws->rwork;
    INT* iwork = ws->iwork;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ulpinv = 1.0 / ulp;
    f64 rtunfl = sqrt(unfl);

    INT iinfo;
    f64 temp1, temp2, temp3 = 0.0;
    INT ntest;
    INT m, m2, m3;
    f64 vl = 0.0, vu = 0.0;

    INT lgn = 0, lwedc, lrwedc, liwedc;
    if (n > 0) {
        lgn = (INT)(log((f64)n) / log(2.0));
        if ((1 << lgn) < n) lgn++;
        if ((1 << lgn) < n) lgn++;
        INT lw1 = 2 * n + n * n;
        INT lw2 = 2 * n * n;
        lwedc = (lw1 > lw2) ? lw1 : lw2;
        lrwedc = 1 + 4 * n + 2 * n * lgn + 3 * n * n;
        liwedc = 3 + 5 * n;
    } else {
        lwedc = 2;
        lrwedc = 8;
        liwedc = 8;
    }

    f64 abstol = unfl + unfl;

    for (INT j = 0; j < 140; j++) {
        ws->result[j] = 0.0;
    }

    if (n == 0) return;

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

    if (jtype <= MAXTYP) {
        iinfo = generate_matrix(n, jtype, A, lda, U, ldu, work, rwork, iwork,
                                ws->rng_state, ws->rng_state3, &ihbw);
        if (iinfo != 0) {
            fprintf(stderr, "Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                          jtype, n, iinfo);
            ws->result[0] = ulpinv;
            assert_info_success(iinfo);
            return;
        }
    }

    ntest = 0;

    for (INT iuplo = 0; iuplo <= 1; iuplo++) {
        const char* uplo = (iuplo == 0) ? "L" : "U";

        /* ZHEEVD('V') — zdrvst2stg.f line 658 */
        zlacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        zheevd("V", uplo, n, A, ldu, D1, work, lwedc,
               rwork, lrwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L130;
        }

        zhet21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        zlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        /* ZHEEVD_2STAGE('N') — zdrvst2stg.f line 683 */
        zheevd_2stage("N", uplo, n, A, ldu, D3, work, ws->lwork,
               rwork, lrwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVD_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L130;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L130:
        zlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 1;

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

        /* ZHEEVX('V','A') — zdrvst2stg.f line 736 */
        zheevx("V", "A", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m, WA1, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L150;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* ZHEEVX_2STAGE('N','A') — zdrvst2stg.f line 761 */
        zheevx_2stage("N", "A", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L150;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L150:
        zlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 1;

        /* ZHEEVX('V','I') — zdrvst2stg.f line 794 */
        zheevx("V", "I", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L160;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        /* ZHEEVX_2STAGE('N','I') — zdrvst2stg.f line 818 */
        zheevx_2stage("N", "I", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L160;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L160:
        zlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 1;

        /* ZHEEVX('V','V') — zdrvst2stg.f line 852 */
        zheevx("V", "V", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L170;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        /* ZHEEVX_2STAGE('N','V') — zdrvst2stg.f line 876 */
        zheevx_2stage("N", "V", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, work, ws->lwork, rwork,
               iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVX_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L170;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L170;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L170:
        /* ZHPEVD — zdrvst2stg.f line 912 */
        zlacpy(" ", n, n, V, ldu, A, lda);

        pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        INT indwrk = n * (n + 1) / 2;
        zhpevd("V", uplo, n, work, D1, Z, ldu,
               work + indwrk, lwedc, rwork, lrwedc, iwork,
               liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L270;
        }

        zhet21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        indwrk = n * (n + 1) / 2;
        zhpevd("N", uplo, n, work, D3, Z, ldu,
               work + indwrk, lwedc, rwork, lrwedc, iwork,
               liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVD(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L270;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L270:
        /* ZHPEVX — zdrvst2stg.f line 1029 */
        pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;

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

        /* ZHPEVX('V','A') — zdrvst2stg.f line 1072 */
        zhpevx("V", "A", uplo, n, work, vl, vu, il, iu,
               abstol, &m, WA1, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L370;
        }

        zhet21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        pack_triangular(n, iuplo, A, lda, work);

        zhpevx("N", "A", uplo, n, work, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L370;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L370:
        ntest = ntest + 1;
        pack_triangular(n, iuplo, A, lda, work);

        /* ZHPEVX('V','I') — zdrvst2stg.f line 1161 */
        zhpevx("V", "I", uplo, n, work, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L460;
        }

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        pack_triangular(n, iuplo, A, lda, work);

        zhpevx("N", "I", uplo, n, work, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L460;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L460:
        ntest = ntest + 1;
        pack_triangular(n, iuplo, A, lda, work);

        /* ZHPEVX('V','V') — zdrvst2stg.f line 1250 */
        zhpevx("V", "V", uplo, n, work, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L550;
        }

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        pack_triangular(n, iuplo, A, lda, work);

        zhpevx("N", "V", uplo, n, work, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, V, rwork, iwork,
               iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEVX(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L550;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L550;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L550:
        ;
        /* ZHBEVD — zdrvst2stg.f line 1306 */
        INT kd;
        if (jtype <= 7) {
            kd = 0;
        } else if (jtype >= 8 && jtype <= 15) {
            kd = (n - 1 > 0) ? n - 1 : 0;
        } else {
            kd = ihbw;
        }

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        zhbevd("V", uplo, n, kd, V, ldu, D1, Z, ldu, work,
               lwedc, rwork, lrwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVD(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L650;
        }

        zhet21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        /* ZHBEVD_2STAGE('N') — zdrvst2stg.f line 1370 */
        zhbevd_2stage("N", uplo, n, kd, V, ldu, D3, Z, ldu, work,
               ws->lwork, rwork, lrwedc, iwork, liwedc, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVD_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L650;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L650:
        /* ZHBEVX — zdrvst2stg.f line 1415 */
        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        zhbevx("V", "A", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m, WA1, Z, ldu, work,
               rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L750;
        }

        zhet21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* ZHBEVX_2STAGE('N','A') — zdrvst2stg.f line 1454 */
        zhbevx_2stage("N", "A", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m2, WA2, Z, ldu, work,
               ws->lwork, rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L750;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L750:
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* ZHBEVX('V','I') — zdrvst2stg.f line 1497 */
        zhbevx("V", "I", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m2, WA2, Z, ldu, work,
               rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L840;
        }

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* ZHBEVX_2STAGE('N','I') — zdrvst2stg.f line 1538 */
        zhbevx_2stage("N", "I", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m3, WA3, Z, ldu, work,
               ws->lwork, rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L840;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L840:
        ntest = ntest + 1;
        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* ZHBEVX('V','V') — zdrvst2stg.f line 1581 */
        zhbevx("V", "V", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m2, WA2, Z, ldu, work,
               rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L930;
        }

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;

        load_band(n, kd, iuplo, A, lda, V, ldu);

        /* ZHBEVX_2STAGE('N','V') — zdrvst2stg.f line 1622 */
        zhbevx_2stage("N", "V", uplo, n, kd, V, ldu, U, ldu, vl,
               vu, il, iu, abstol, &m3, WA3, Z, ldu, work,
               ws->lwork, rwork, iwork, iwork + 5 * n, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEVX_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L930;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L930;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

L930:
        /* ZHEEV — zdrvst2stg.f line 1660 */
        zlacpy(" ", n, n, A, lda, V, ldu);

        ntest = ntest + 1;
        zheev("V", uplo, n, A, ldu, D1, work, ws->lwork, rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L950;
        }

        zhet21(1, uplo, n, 0, V, ldu, D1, D2, A, ldu, Z,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        zlacpy(" ", n, n, V, ldu, A, lda);

        ntest = ntest + 2;
        /* ZHEEV_2STAGE('N') — zdrvst2stg.f line 1687 */
        zheev_2stage("N", uplo, n, A, ldu, D3, work, ws->lwork, rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEV_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L950;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L950:
        zlacpy(" ", n, n, V, ldu, A, lda);

        /* ZHPEV — zdrvst2stg.f line 1720 */
        pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 1;
        indwrk = n * (n + 1) / 2;
        zhpev("V", uplo, n, work, D1, Z, ldu,
              work + indwrk, rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1050;
        }

        zhet21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        pack_triangular(n, iuplo, A, lda, work);

        ntest = ntest + 2;
        indwrk = n * (n + 1) / 2;
        zhpev("N", uplo, n, work, D3, Z, ldu,
              work + indwrk, rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHPEV(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1050;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1050:
        /* ZHBEV — zdrvst2stg.f line 1813 */
        ;
        if (jtype <= 7) {
            kd = 0;
        } else if (jtype >= 8 && jtype <= 15) {
            kd = (n - 1 > 0) ? n - 1 : 0;
        } else {
            kd = ihbw;
        }

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 1;
        zhbev("V", uplo, n, kd, V, ldu, D1, Z, ldu, work,
              rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEV(V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1140;
        }

        zhet21(1, uplo, n, 0, A, lda, D1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        load_band(n, kd, iuplo, A, lda, V, ldu);

        ntest = ntest + 2;
        /* ZHBEV_2STAGE('N') — zdrvst2stg.f line 1874 */
        zhbev_2stage("N", uplo, n, kd, V, ldu, D3, Z, ldu, work,
              ws->lwork, rwork, &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHBEV_2STAGE(N,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1140;
        }

L1140:
        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
            temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

        /* ZHEEVR — zdrvst2stg.f line 1904 */
        zlacpy(" ", n, n, A, lda, V, ldu);
        ntest = ntest + 1;
        zheevr("V", "A", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m, WA1, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR(V,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1170;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet21(1, uplo, n, 0, A, ldu, WA1, D2, Z, ldu, V,
               ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;
        /* ZHEEVR_2STAGE('N','A') — zdrvst2stg.f line 1930 */
        zheevr_2stage("N", "A", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR_2STAGE(N,A,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1170;
        }

        temp1 = 0.0;
        temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
            temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
        }
        ws->result[ntest - 1] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));

L1170:
        ntest = ntest + 1;
        zlacpy(" ", n, n, V, ldu, A, lda);
        zheevr("V", "I", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR(V,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1180;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;
        zlacpy(" ", n, n, V, ldu, A, lda);
        /* ZHEEVR_2STAGE('N','I') — zdrvst2stg.f line 1989 */
        zheevr_2stage("N", "I", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR_2STAGE(N,I,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1180;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, ulp * temp3);

L1180:
        ntest = ntest + 1;
        zlacpy(" ", n, n, V, ldu, A, lda);
        zheevr("V", "V", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m2, WA2, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR(V,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            ws->result[ntest] = ulpinv;
            ws->result[ntest + 1] = ulpinv;
            goto L1190;
        }

        zlacpy(" ", n, n, V, ldu, A, lda);

        zhet22(1, uplo, n, m2, 0, A, ldu, WA2, D2, Z, ldu,
               V, ldu, TAU, work, rwork, ws->result + ntest - 1);

        ntest = ntest + 2;
        zlacpy(" ", n, n, V, ldu, A, lda);
        /* ZHEEVR_2STAGE('N','V') — zdrvst2stg.f line 2043 */
        zheevr_2stage("N", "V", uplo, n, A, ldu, vl, vu, il, iu,
               abstol, &m3, WA3, Z, ldu, iwork, work, ws->lwork,
               rwork, ws->lrwork, iwork + 2 * n, ws->liwork - 2 * n,
               &iinfo);
        if (iinfo != 0) {
            fprintf(stderr, "ZHEEVR_2STAGE(N,V,%s) returned info=%d\n", uplo, iinfo);
            ws->result[ntest - 1] = ulpinv;
            goto L1190;
        }

        if (m3 == 0 && n > 0) {
            ws->result[ntest - 1] = ulpinv;
            goto L1190;
        }

        temp1 = dsxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
        temp2 = dsxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
        if (n > 0)
            temp3 = fmax(fabs(WA1[0]), fabs(WA1[n - 1]));
        else
            temp3 = 0.0;
        ws->result[ntest - 1] = (temp1 + temp2) / fmax(unfl, temp3 * ulp);

        zlacpy(" ", n, n, V, ldu, A, lda);

L1190:
        ;
    } /* end IUPLO loop */

    for (INT j = 0; j < ntest; j++) {
        if (ws->result[j] >= THRESH) {
            fprintf(stderr, "  Test %d: ratio = %.6e (THRESH=%.1f) n=%d jtype=%d\n",
                          j + 1, ws->result[j], THRESH, n, jtype);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

static void test_zdrvst2stg_case(void** state)
{
    zdrvst2stg_params_t* params = *state;
    run_zdrvst2stg_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static zdrvst2stg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zdrvst2stg_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zdrvst2stg_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zdrvst2stg_case;
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

    (void)_cmocka_run_group_tests("zdrvst2stg", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
