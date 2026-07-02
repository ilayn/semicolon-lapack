/**
 * @file test_cchkdmd.c
 * @brief Test driver for the Dynamic Mode Decomposition routines CGEDMD and
 *        CGEDMDQ - port of LAPACK TESTING/EIG/cchkdmd.f90.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const INT MVAL[] = {10, 20, 30, 50};
static const INT NVAL[] = {5, 10, 11, 20};
#define NDIM (INT)(sizeof(MVAL) / sizeof(MVAL[0]))
#define MAXM 50
#define MAXN 20
#define WORKSZ 50000
#define IWORKSZ 4000

typedef struct {
    c64 *ZA, *ZAC, *ZF, *ZF0, *ZF1, *ZX, *ZX0, *ZY, *ZY0, *ZY1, *ZZ, *ZZ1;
    c64 *ZS, *ZAU, *ZW, *ZEIGS, *ZEIGSA, *ZDA, *ZDL, *ZDR;
    f32 *RES, *RES1, *RESEX, *SINGVX, *SINGVQX;
    c64 *zwork;
    f32 *rwork;
    INT *iwork;
    INT *iwork_gen;
    uint64_t state[4];
} zdmd_ws_t;

static zdmd_ws_t* g_ws = NULL;

static void* xmalloc(size_t n) { void* p = malloc(n); return p; }

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(zdmd_ws_t));
    if (!g_ws) return -1;
    const size_t m2 = (size_t)MAXM * MAXM;
    g_ws->ZA  = xmalloc(m2 * sizeof(c64));
    g_ws->ZAC = xmalloc(m2 * sizeof(c64));
    g_ws->ZF  = xmalloc(m2 * sizeof(c64));
    g_ws->ZF0 = xmalloc(m2 * sizeof(c64));
    g_ws->ZF1 = xmalloc(m2 * sizeof(c64));
    g_ws->ZX  = xmalloc(m2 * sizeof(c64));
    g_ws->ZX0 = xmalloc(m2 * sizeof(c64));
    g_ws->ZY  = xmalloc(m2 * sizeof(c64));
    g_ws->ZY0 = xmalloc(m2 * sizeof(c64));
    g_ws->ZY1 = xmalloc(m2 * sizeof(c64));
    g_ws->ZZ  = xmalloc(m2 * sizeof(c64));
    g_ws->ZZ1 = xmalloc(m2 * sizeof(c64));
    g_ws->ZS  = xmalloc(m2 * sizeof(c64));
    g_ws->ZAU = xmalloc(m2 * sizeof(c64));
    g_ws->ZW  = xmalloc(m2 * sizeof(c64));
    g_ws->ZEIGS  = xmalloc(MAXM * sizeof(c64));
    g_ws->ZEIGSA = xmalloc(MAXM * sizeof(c64));
    g_ws->ZDA = xmalloc(MAXM * sizeof(c64));
    g_ws->ZDL = xmalloc(MAXM * sizeof(c64));
    g_ws->ZDR = xmalloc(MAXM * sizeof(c64));
    g_ws->RES     = xmalloc(MAXM * sizeof(f32));
    g_ws->RES1    = xmalloc(MAXM * sizeof(f32));
    g_ws->RESEX   = xmalloc(MAXM * sizeof(f32));
    g_ws->SINGVX  = xmalloc(MAXM * sizeof(f32));
    g_ws->SINGVQX = xmalloc(MAXM * sizeof(f32));
    g_ws->zwork = xmalloc(WORKSZ * sizeof(c64));
    g_ws->rwork = xmalloc(WORKSZ * sizeof(f32));
    g_ws->iwork = xmalloc(IWORKSZ * sizeof(INT));
    g_ws->iwork_gen = xmalloc(4 * MAXM * sizeof(INT));

    if (!g_ws->ZA || !g_ws->ZAC || !g_ws->ZF || !g_ws->ZF0 || !g_ws->ZF1 ||
        !g_ws->ZX || !g_ws->ZX0 || !g_ws->ZY || !g_ws->ZY0 || !g_ws->ZY1 ||
        !g_ws->ZZ || !g_ws->ZZ1 || !g_ws->ZS || !g_ws->ZAU || !g_ws->ZW ||
        !g_ws->ZEIGS || !g_ws->ZEIGSA || !g_ws->ZDA || !g_ws->ZDL ||
        !g_ws->ZDR || !g_ws->RES || !g_ws->RES1 || !g_ws->RESEX ||
        !g_ws->SINGVX || !g_ws->SINGVQX || !g_ws->zwork || !g_ws->rwork ||
        !g_ws->iwork || !g_ws->iwork_gen) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_ws) {
        free(g_ws->ZA); free(g_ws->ZAC); free(g_ws->ZF); free(g_ws->ZF0);
        free(g_ws->ZF1); free(g_ws->ZX); free(g_ws->ZX0); free(g_ws->ZY);
        free(g_ws->ZY0); free(g_ws->ZY1); free(g_ws->ZZ); free(g_ws->ZZ1);
        free(g_ws->ZS); free(g_ws->ZAU); free(g_ws->ZW); free(g_ws->ZEIGS);
        free(g_ws->ZEIGSA); free(g_ws->ZDA); free(g_ws->ZDL); free(g_ws->ZDR);
        free(g_ws->RES); free(g_ws->RES1); free(g_ws->RESEX); free(g_ws->SINGVX);
        free(g_ws->SINGVQX); free(g_ws->zwork); free(g_ws->rwork);
        free(g_ws->iwork); free(g_ws->iwork_gen);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void run_case(INT M, INT N)
{
    zdmd_ws_t* w = g_ws;
    const f32 ZERO = 0.0, ONE = 1.0;
    const c64 CZERO = CMPLXF(0.0, 0.0), CONE = CMPLXF(1.0, 0.0);
    const INT lda = M, ldf = M, ldx = M, ldy = M, ldw = N, ldz = M, ldau = M, lds = N;
    const f32 eps = slamch("P");
    const f32 tol = M * eps;
    INT info, i;

    f32 e_zxw = ZERO, e_au = ZERO, e_rez = ZERO, e_fqr = ZERO, e_rezq = ZERO;

    rng_seed(w->state, 0x9E3779B97F4A7C15ULL);
    INT iseed[4] = {4, 3, 2, 1};

    for (INT k_traj = 1; k_traj <= 2; k_traj++) {
        const f32 cond = 1.0e4, condl = 1.0e1, condr = 1.0e1;
        const c64 zmax = CMPLXF(1.0e1, 1.0e1);

        for (INT mode = 1; mode <= 6; mode++) {
            clatmr(M, M, "N", "N", w->ZDA, mode, cond, zmax, "F", "N", w->ZDL, 6,
                   condl, w->ZDR, 6, condr, "N", w->iwork_gen, M, M, ZERO, -ONE,
                   "N", w->ZA, lda, w->iwork_gen + 2 * M, &info, w->state);

            /* Spectral radius of ZA via its eigenvalues. */
            memcpy(w->ZAC, w->ZA, (size_t)lda * M * sizeof(c64));
            c64 zdum[4];
            cgeev("N", "N", M, w->ZAC, lda, w->ZEIGSA, zdum, 2, zdum, 2,
                  w->zwork, WORKSZ, w->rwork, &info);
            f32 tmp = cabsf(w->ZEIGSA[cblas_icamax(M, w->ZEIGSA, 1)]);
            clascl("G", 0, 0, tmp, ONE, M, M, w->ZA, lda, &info);
            clascl("G", 0, 0, tmp, ONE, M, 1, w->ZEIGSA, M, &info);
            const f32 anorm = clange("F", M, M, w->ZA, lda, NULL);

            if (k_traj == 2) {
                clarnv(2, iseed, M, &w->ZF[0]);
                for (i = 0; i < N / 2; i++)
                    cblas_cgemv(CblasColMajor, CblasNoTrans, M, M, &CONE, w->ZA, lda,
                                &w->ZF[i * ldf], 1, &CZERO, &w->ZF[(i + 1) * ldf], 1);
                for (INT jc = 0; jc < N / 2; jc++) {
                    memcpy(&w->ZX0[jc * ldx], &w->ZF[jc * ldf], (size_t)M * sizeof(c64));
                    memcpy(&w->ZY0[jc * ldy], &w->ZF[(jc + 1) * ldf], (size_t)M * sizeof(c64));
                }
                clarnv(2, iseed, M, &w->ZF[0]);
                for (i = 0; i < N - N / 2; i++)
                    cblas_cgemv(CblasColMajor, CblasNoTrans, M, M, &CONE, w->ZA, lda,
                                &w->ZF[i * ldf], 1, &CZERO, &w->ZF[(i + 1) * ldf], 1);
                for (INT jc = 0; jc < N - N / 2; jc++) {
                    memcpy(&w->ZX0[(N / 2 + jc) * ldx], &w->ZF[jc * ldf], (size_t)M * sizeof(c64));
                    memcpy(&w->ZY0[(N / 2 + jc) * ldy], &w->ZF[(jc + 1) * ldf], (size_t)M * sizeof(c64));
                }
            } else {
                clarnv(2, iseed, M, &w->ZF[0]);
                for (i = 0; i < N; i++)
                    cblas_cgemv(CblasColMajor, CblasNoTrans, M, M, &CONE, w->ZA, M,
                                &w->ZF[i * ldf], 1, &CZERO, &w->ZF[(i + 1) * ldf], 1);
                memcpy(w->ZF0, w->ZF, (size_t)ldf * (N + 1) * sizeof(c64));
                for (INT jc = 0; jc < N; jc++) {
                    memcpy(&w->ZX0[jc * ldx], &w->ZF0[jc * ldf], (size_t)M * sizeof(c64));
                    memcpy(&w->ZY0[jc * ldy], &w->ZF0[(jc + 1) * ldf], (size_t)M * sizeof(c64));
                }
            }

            for (INT ijobz = 1; ijobz <= 4; ijobz++) {
                const char* jobz;
                const char* resids;
                switch (ijobz) {
                case 1: jobz = "V"; resids = "R"; break;
                case 2: jobz = "V"; resids = "N"; break;
                case 3: jobz = "F"; resids = "N"; break;
                default: jobz = "N"; resids = "N"; break;
                }

                for (INT ijobref = 1; ijobref <= 3; ijobref++) {
                    const char* jobref;
                    switch (ijobref) {
                    case 1: jobref = "R"; break;
                    case 2: jobref = "E"; break;
                    default: jobref = "N"; break;
                    }

                    for (INT iscale = 1; iscale <= 4; iscale++) {
                        const char* scale;
                        switch (iscale) {
                        case 1: scale = "S"; break;
                        case 2: scale = "C"; break;
                        case 3: scale = "Y"; break;
                        default: scale = "N"; break;
                        }

                        for (INT inrnk = -1; inrnk >= -2; inrnk--) {
                            const INT nrnk = inrnk;

                            for (INT iwhtsvd = 1; iwhtsvd <= 3; iwhtsvd++) {
                                const INT whtsvd = iwhtsvd;

                                for (INT lwminopt = 1; lwminopt <= 2; lwminopt++) {
                                    INT k = 0, kq = 0;
                                    c64 zdummy[22];
                                    f32 wdummy[2];
                                    INT idummy[2];

                                    for (INT jc = 0; jc < N; jc++) {
                                        memcpy(&w->ZX[jc * ldx], &w->ZX0[jc * ldx], (size_t)M * sizeof(c64));
                                        memcpy(&w->ZY[jc * ldy], &w->ZY0[jc * ldy], (size_t)M * sizeof(c64));
                                    }

                                    /* CGEDMD: workspace query */
                                    cgedmd(scale, jobz, resids, jobref, whtsvd, M, N,
                                           w->ZX, ldx, w->ZY, ldy, nrnk, tol, &k,
                                           w->ZEIGS, w->ZZ, ldz, w->RES, w->ZAU, ldau,
                                           w->ZW, ldw, w->ZS, lds, zdummy, -1, wdummy, -1,
                                           idummy, -1, &info);
                                    INT lzwork = (INT)crealf(zdummy[lwminopt - 1]);
                                    INT lrwork = (INT)wdummy[0];
                                    INT liwork = idummy[0];
                                    assert_true(lzwork <= WORKSZ && lrwork <= WORKSZ && liwork <= IWORKSZ);

                                    /* CGEDMD: actual call */
                                    cgedmd(scale, jobz, resids, jobref, whtsvd, M, N,
                                           w->ZX, ldx, w->ZY, ldy, nrnk, tol, &k,
                                           w->ZEIGS, w->ZZ, ldz, w->RES, w->ZAU, ldau,
                                           w->ZW, ldw, w->ZS, lds, w->zwork, lzwork,
                                           w->rwork, lrwork, w->iwork, liwork, &info);
                                    assert_true(info == 0 || info == 4);

                                    for (i = 0; i < N; i++) w->SINGVX[i] = w->rwork[i];

                                    /* Check: Z = X*W */
                                    if (jobz[0] == 'V') {
                                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, k, &CONE, w->ZX, ldx, w->ZW, ldw,
                                                    &CZERO, w->ZZ1, ldz);
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            c64 neg = -CONE;
                                            cblas_caxpy(M, &neg, &w->ZZ[i * ldz], 1, &w->ZZ1[i * ldz], 1);
                                            f32 nn = cblas_scnrm2(M, &w->ZZ1[i * ldz], 1);
                                            if (nn > tmp1) tmp1 = nn;
                                        }
                                        if (tmp1 > e_zxw) e_zxw = tmp1;
                                    }

                                    /* Check: A*U (refinement) or Exact DMD vectors */
                                    if (jobref[0] == 'R') {
                                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, &CONE, w->ZA, lda, w->ZX, ldx,
                                                    &CZERO, w->ZZ1, ldz);
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            c64 neg = -CONE;
                                            cblas_caxpy(M, &neg, &w->ZAU[i * ldau], 1, &w->ZZ1[i * ldz], 1);
                                            f32 v = cblas_scnrm2(M, &w->ZZ1[i * ldz], 1) *
                                                    w->SINGVX[k - 1] / (anorm * w->SINGVX[0]);
                                            if (v > tmp1) tmp1 = v;
                                        }
                                        if (tmp1 > e_au) e_au = tmp1;
                                    } else if (jobref[0] == 'E') {
                                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, &CONE, w->ZA, lda, w->ZAU, ldau,
                                                    &CZERO, w->ZY1, ldy);
                                        for (i = 0; i < k; i++) {
                                            c64 neg = -w->ZEIGS[i];
                                            cblas_caxpy(M, &neg, &w->ZAU[i * ldau], 1, &w->ZY1[i * ldy], 1);
                                            w->RESEX[i] = cblas_scnrm2(M, &w->ZY1[i * ldy], 1) /
                                                          cblas_scnrm2(M, &w->ZAU[i * ldau], 1);
                                        }
                                    }

                                    /* Check: residuals returned vs explicitly computed */
                                    if (resids[0] == 'R') {
                                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, &CONE, w->ZA, lda, w->ZZ, ldz,
                                                    &CZERO, w->ZY1, ldy);
                                        for (i = 0; i < k; i++) {
                                            c64 neg = -w->ZEIGS[i];
                                            cblas_caxpy(M, &neg, &w->ZZ[i * ldz], 1, &w->ZY1[i * ldy], 1);
                                            w->RES1[i] = cblas_scnrm2(M, &w->ZY1[i * ldy], 1);
                                        }
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            f32 v = fabsf(w->RES[i] - w->RES1[i]) *
                                                    w->SINGVX[k - 1] / (anorm * w->SINGVX[0]);
                                            if (v > tmp1) tmp1 = v;
                                        }
                                        if (tmp1 > e_rez) e_rez = tmp1;
                                    }

                                    /*==========  CGEDMDQ  ==========*/
                                    if (k_traj == 1) {
                                        memcpy(w->ZF, w->ZF0, (size_t)ldf * (N + 1) * sizeof(c64));

                                        cgedmdq(scale, jobz, resids, "Q", "R", jobref,
                                                whtsvd, M, N + 1, w->ZF, ldf, w->ZX, ldx,
                                                w->ZY, ldy, nrnk, tol, &kq, w->ZEIGS, w->ZZ,
                                                ldz, w->RES, w->ZAU, ldau, w->ZW, ldw, w->ZS,
                                                lds, zdummy, -1, wdummy, -1, idummy, -1, &info);
                                        lzwork = (INT)crealf(zdummy[lwminopt - 1]);
                                        lrwork = (INT)wdummy[0];
                                        liwork = idummy[0];
                                        assert_true(lzwork <= WORKSZ && lrwork <= WORKSZ && liwork <= IWORKSZ);

                                        cgedmdq(scale, jobz, resids, "Q", "R", jobref,
                                                whtsvd, M, N + 1, w->ZF, ldf, w->ZX, ldx,
                                                w->ZY, ldy, nrnk, tol, &kq, w->ZEIGS, w->ZZ,
                                                ldz, w->RES, w->ZAU, ldau, w->ZW, ldw, w->ZS,
                                                lds, w->zwork, lzwork, w->rwork, lrwork,
                                                w->iwork, liwork, &info);
                                        assert_true(info == 0 || info == 4);

                                        for (i = 0; i < N; i++) w->SINGVQX[i] = w->rwork[i];

                                        /* F = Q*R : ||F0 - Q*R||_F / ||F0||_F */
                                        memcpy(w->ZF1, w->ZF0, (size_t)ldf * (N + 1) * sizeof(c64));
                                        const INT mn = (M < N + 1) ? M : N + 1;
                                        c64 negone = -CONE;
                                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, N + 1, mn, &negone, w->ZF, ldf, w->ZY, ldy,
                                                    &CONE, w->ZF1, ldf);
                                        f32 fqr = clange("F", M, N + 1, w->ZF1, ldf, NULL) /
                                                  clange("F", M, N + 1, w->ZF0, ldf, NULL);
                                        if (fqr > e_fqr) e_fqr = fqr;

                                        /* CGEDMDQ residuals */
                                        if (resids[0] == 'R') {
                                            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                        M, kq, M, &CONE, w->ZA, lda, w->ZZ, ldz,
                                                        &CZERO, w->ZY1, ldy);
                                            for (i = 0; i < kq; i++) {
                                                c64 neg = -w->ZEIGS[i];
                                                cblas_caxpy(M, &neg, &w->ZZ[i * ldz], 1, &w->ZY1[i * ldy], 1);
                                                w->RES1[i] = cblas_scnrm2(M, &w->ZY1[i * ldy], 1);
                                            }
                                            f32 tmpq = ZERO;
                                            for (i = 0; i < kq; i++) {
                                                f32 v = fabsf(w->RES[i] - w->RES1[i]) *
                                                        w->SINGVQX[kq - 1] / (anorm * w->SINGVQX[0]);
                                                if (v > tmpq) tmpq = v;
                                            }
                                            if (tmpq > e_rezq) e_rezq = tmpq;
                                        }
                                    }
                                } /* lwminopt */
                            } /* whtsvd */
                        } /* nrnk */
                    } /* scale */
                } /* jobref */
            } /* jobz */
        } /* mode */
    } /* k_traj */

    assert_residual_below(e_zxw / (M * eps), 10.0);
    assert_residual_below(e_au / (M * N * eps), 10.0);
    assert_residual_below(e_rez / (M * N * eps), 10.0);
    assert_residual_below(e_fqr / (M * N * eps), 10.0);
    assert_residual_below(e_rezq / (M * N * eps), 10.0);
}

static void test_cchkdmd_case(void** state)
{
    const INT idx = (INT)(intptr_t)*state;
    run_case(MVAL[idx], NVAL[idx]);
}

int main(void)
{
    struct CMUnitTest tests[NDIM];
    for (INT i = 0; i < NDIM; i++) {
        tests[i].name = "cchkdmd";
        tests[i].test_func = test_cchkdmd_case;
        tests[i].setup_func = NULL;
        tests[i].teardown_func = NULL;
        tests[i].initial_state = (void*)(intptr_t)i;
    }
    (void)_cmocka_run_group_tests("cchkdmd", tests, NDIM, group_setup, group_teardown);
    return 0;
}
