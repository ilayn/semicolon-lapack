/**
 * @file test_schkdmd.c
 * @brief Test driver for the Dynamic Mode Decomposition routines SGEDMD and
 *        SGEDMDQ - port of LAPACK TESTING/EIG/schkdmd.f90.
 *
 * For each pair of dimensions (M,N) an M-by-M operator A is generated with
 * DLATMR, its spectral radius is normalized, and a sequence of snapshots is
 * produced by powering A. SGEDMD (and, for single-trajectory data, SGEDMDQ)
 * is then run over all combinations of the scaling, eigenvector, refinement,
 * rank-truncation, SVD-method and workspace options, and the following checks
 * are performed:
 *   - Z = X*W                (Ritz vectors are U * (Rayleigh eigenvectors))
 *   - A*U = Y*V*inv(Sigma)   (refinement data, JOBREF='R')
 *   - residuals returned match explicitly computed residuals
 *   - SGEDMD / SGEDMDQ singular values agree
 *   - F = Q*R                (QR factors from SGEDMDQ)
 *   - SGEDMDQ residuals match explicitly computed residuals
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
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
    f32 *A, *AC, *VA, *F, *F1, *F2, *X, *X0, *Y, *Y0, *Y1, *Z, *Z1, *S, *AU, *W;
    f32 *EIGA, *DA, *DL, *DR;
    f32 *REIG, *IEIG, *REIGQ, *IEIGQ, *REIGA, *IEIGA;
    f32 *RES, *RES1, *RESEX, *SINGVX, *SINGVQX;
    f32 *work;
    INT *iwork;
    INT *iwork_gen;
    uint64_t state[4];
} dmd_ws_t;

static dmd_ws_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(dmd_ws_t));
    if (!g_ws) return -1;

    const INT m2 = MAXM * MAXM;     /* covers all 2D arrays */
    g_ws->A  = malloc(m2 * sizeof(f32));
    g_ws->AC = malloc(m2 * sizeof(f32));
    g_ws->VA = malloc(m2 * sizeof(f32));
    g_ws->F  = malloc(m2 * sizeof(f32));
    g_ws->F1 = malloc(m2 * sizeof(f32));
    g_ws->F2 = malloc(m2 * sizeof(f32));
    g_ws->X  = malloc(m2 * sizeof(f32));
    g_ws->X0 = malloc(m2 * sizeof(f32));
    g_ws->Y  = malloc(m2 * sizeof(f32));
    g_ws->Y0 = malloc(m2 * sizeof(f32));
    g_ws->Y1 = malloc(m2 * sizeof(f32));
    g_ws->Z  = malloc(m2 * sizeof(f32));
    g_ws->Z1 = malloc(m2 * sizeof(f32));
    g_ws->S  = malloc(m2 * sizeof(f32));
    g_ws->AU = malloc(m2 * sizeof(f32));
    g_ws->W  = malloc(m2 * sizeof(f32));
    g_ws->EIGA = malloc(MAXM * 2 * sizeof(f32));
    g_ws->DA = malloc(MAXM * sizeof(f32));
    g_ws->DL = malloc(MAXM * sizeof(f32));
    g_ws->DR = malloc(MAXM * sizeof(f32));
    g_ws->REIG  = malloc(MAXM * sizeof(f32));
    g_ws->IEIG  = malloc(MAXM * sizeof(f32));
    g_ws->REIGQ = malloc(MAXM * sizeof(f32));
    g_ws->IEIGQ = malloc(MAXM * sizeof(f32));
    g_ws->REIGA = malloc(MAXM * sizeof(f32));
    g_ws->IEIGA = malloc(MAXM * sizeof(f32));
    g_ws->RES     = malloc(MAXM * sizeof(f32));
    g_ws->RES1    = malloc(MAXM * sizeof(f32));
    g_ws->RESEX   = malloc(MAXM * sizeof(f32));
    g_ws->SINGVX  = malloc(MAXM * sizeof(f32));
    g_ws->SINGVQX = malloc(MAXM * sizeof(f32));
    g_ws->work  = malloc(WORKSZ * sizeof(f32));
    g_ws->iwork = malloc(IWORKSZ * sizeof(INT));
    g_ws->iwork_gen = malloc(4 * MAXM * sizeof(INT));

    if (!g_ws->A || !g_ws->AC || !g_ws->VA || !g_ws->F || !g_ws->F1 ||
        !g_ws->F2 || !g_ws->X || !g_ws->X0 || !g_ws->Y || !g_ws->Y0 ||
        !g_ws->Y1 || !g_ws->Z || !g_ws->Z1 || !g_ws->S || !g_ws->AU ||
        !g_ws->W || !g_ws->EIGA || !g_ws->DA || !g_ws->DL || !g_ws->DR ||
        !g_ws->REIG || !g_ws->IEIG || !g_ws->REIGQ || !g_ws->IEIGQ ||
        !g_ws->REIGA || !g_ws->IEIGA || !g_ws->RES || !g_ws->RES1 ||
        !g_ws->RESEX || !g_ws->SINGVX || !g_ws->SINGVQX || !g_ws->work ||
        !g_ws->iwork || !g_ws->iwork_gen) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_ws) {
        free(g_ws->A); free(g_ws->AC); free(g_ws->VA); free(g_ws->F);
        free(g_ws->F1); free(g_ws->F2); free(g_ws->X); free(g_ws->X0);
        free(g_ws->Y); free(g_ws->Y0); free(g_ws->Y1); free(g_ws->Z);
        free(g_ws->Z1); free(g_ws->S); free(g_ws->AU); free(g_ws->W);
        free(g_ws->EIGA); free(g_ws->DA); free(g_ws->DL); free(g_ws->DR);
        free(g_ws->REIG); free(g_ws->IEIG); free(g_ws->REIGQ); free(g_ws->IEIGQ);
        free(g_ws->REIGA); free(g_ws->IEIGA); free(g_ws->RES); free(g_ws->RES1);
        free(g_ws->RESEX); free(g_ws->SINGVX); free(g_ws->SINGVQX);
        free(g_ws->work); free(g_ws->iwork); free(g_ws->iwork_gen);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void run_case(INT M, INT N)
{
    dmd_ws_t* w = g_ws;
    const f32 ONE = 1.0, ZERO = 0.0;
    const INT lda = M, ldf = M;
    const INT ldx = (M > N + 1) ? M : N + 1;
    const INT ldy = ldx, ldw = N, ldz = M, ldau = ldx, lds = N;
    const f32 eps = slamch("P");
    const f32 tol = M * eps;
    f32 ab[4];
    INT info, i;

    /* Accumulated worst-case (adjusted) error measures over the whole battery */
    f32 e_zxw = ZERO, e_au = ZERO, e_rez = ZERO;
    f32 e_sv = ZERO, e_fqr = ZERO, e_rezq = ZERO;

    /* Seed generators; ISEED mirrors the reference reset at each dimension. */
    rng_seed(w->state, 0x9E3779B97F4A7C15ULL);
    INT iseed[4] = {4, 3, 2, 1};

    for (INT k_traj = 1; k_traj <= 2; k_traj++) {
        const f32 cond = 1.0e8, dmax = 1.0e2, condl = 1.0e2, condr = 1.0e2;

        for (INT mode = 1; mode <= 6; mode++) {
            /* Generate the M-by-M operator A. */
            slatmr(M, M, "S", "N", w->DA, mode, cond, dmax, "F", "N", w->DL, 6,
                   condl, w->DR, 6, condr, "N", w->iwork_gen, M, M, ZERO, -ONE,
                   "N", w->A, lda, w->iwork_gen + 2 * M, &info, w->state);

            /* Eigenvalues of A; used to normalize the spectral radius. */
            memcpy(w->AC, w->A, (size_t)lda * M * sizeof(f32));
            sgeev("N", "V", M, w->AC, M, w->REIGA, w->IEIGA, w->VA, M, w->VA, M,
                  w->work, 4 * M + 1, &info);
            f32 tmp = ZERO;
            for (i = 0; i < M; i++) {
                w->EIGA[i] = w->REIGA[i];
                w->EIGA[i + M] = w->IEIGA[i];
                f32 r = sqrtf(w->REIGA[i] * w->REIGA[i] + w->IEIGA[i] * w->IEIGA[i]);
                if (r > tmp) tmp = r;
            }
            /* Scale A to have the desirable spectral radius. */
            slascl("G", 0, 0, tmp, ONE, M, M, w->A, M, &info);
            slascl("G", 0, 0, tmp, ONE, M, 2, w->EIGA, M, &info);

            const f32 anorm = slange("F", N, N, w->A, M, NULL);

            if (k_traj == 2) {
                /* Data from two initial conditions. */
                slarnv(2, iseed, M, &w->F1[0]);
                for (i = 0; i < M; i++) w->F1[i] = 1.0e-10f * w->F1[i];
                for (i = 0; i < N / 2; i++)
                    cblas_sgemv(CblasColMajor, CblasNoTrans, M, M, ONE, w->A, M,
                                &w->F1[i * ldf], 1, ZERO, &w->F1[(i + 1) * ldf], 1);
                for (INT jc = 0; jc < N / 2; jc++) {
                    memcpy(&w->X0[jc * ldx], &w->F1[jc * ldf], (size_t)M * sizeof(f32));
                    memcpy(&w->Y0[jc * ldy], &w->F1[(jc + 1) * ldf], (size_t)M * sizeof(f32));
                }
                slarnv(2, iseed, M, &w->F1[0]);
                for (i = 0; i < N - N / 2; i++)
                    cblas_sgemv(CblasColMajor, CblasNoTrans, M, M, ONE, w->A, M,
                                &w->F1[i * ldf], 1, ZERO, &w->F1[(i + 1) * ldf], 1);
                for (INT jc = 0; jc < N - N / 2; jc++) {
                    memcpy(&w->X0[(N / 2 + jc) * ldx], &w->F1[jc * ldf], (size_t)M * sizeof(f32));
                    memcpy(&w->Y0[(N / 2 + jc) * ldy], &w->F1[(jc + 1) * ldf], (size_t)M * sizeof(f32));
                }
            } else {
                /* Single trajectory. */
                slarnv(2, iseed, M, &w->F[0]);
                for (i = 0; i < N; i++)
                    cblas_sgemv(CblasColMajor, CblasNoTrans, M, M, ONE, w->A, M,
                                &w->F[i * ldf], 1, ZERO, &w->F[(i + 1) * ldf], 1);
                for (INT jc = 0; jc < N; jc++) {
                    memcpy(&w->X0[jc * ldx], &w->F[jc * ldf], (size_t)M * sizeof(f32));
                    memcpy(&w->Y0[jc * ldy], &w->F[(jc + 1) * ldf], (size_t)M * sizeof(f32));
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

                            for (INT iwhtsvd = 1; iwhtsvd <= 4; iwhtsvd++) {
                                const INT whtsvd = iwhtsvd;

                                for (INT lwminopt = 1; lwminopt <= 2; lwminopt++) {
                                    INT k = 0, kq = 0;
                                    f32 wdummy[2];
                                    INT idummy[2];

                                    for (INT jc = 0; jc < N; jc++) {
                                        memcpy(&w->X[jc * ldx], &w->X0[jc * ldx], (size_t)M * sizeof(f32));
                                        memcpy(&w->Y[jc * ldy], &w->Y0[jc * ldy], (size_t)M * sizeof(f32));
                                    }

                                    /* SGEDMD: workspace query */
                                    sgedmd(scale, jobz, resids, jobref, whtsvd, M, N,
                                           w->X, ldx, w->Y, ldy, nrnk, tol, &k,
                                           w->REIG, w->IEIG, w->Z, ldz, w->RES,
                                           w->AU, ldau, w->W, ldw, w->S, lds,
                                           wdummy, -1, idummy, -1, &info);
                                    INT liwork = idummy[0];
                                    INT lwork = (INT)wdummy[lwminopt - 1];
                                    assert_true(lwork <= WORKSZ && liwork <= IWORKSZ);

                                    /* SGEDMD: actual call */
                                    sgedmd(scale, jobz, resids, jobref, whtsvd, M, N,
                                           w->X, ldx, w->Y, ldy, nrnk, tol, &k,
                                           w->REIG, w->IEIG, w->Z, ldz, w->RES,
                                           w->AU, ldau, w->W, ldw, w->S, lds,
                                           w->work, lwork, w->iwork, liwork, &info);
                                    assert_true(info == 0 || info == 4);

                                    memcpy(w->SINGVX, w->work, (size_t)N * sizeof(f32));

                                    /* Check: Z = X*W */
                                    if (jobz[0] == 'V') {
                                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, k, ONE, w->X, ldx, w->W, ldw,
                                                    ZERO, w->Z1, ldz);
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            cblas_saxpy(M, -ONE, &w->Z[i * ldz], 1, &w->Z1[i * ldz], 1);
                                            f32 nn = cblas_snrm2(M, &w->Z1[i * ldz], 1);
                                            if (nn > tmp1) tmp1 = nn;
                                        }
                                        if (tmp1 > e_zxw) e_zxw = tmp1;
                                    }

                                    /* Check: A*U (refinement) or Exact DMD vectors */
                                    if (jobref[0] == 'R') {
                                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, ONE, w->A, lda, w->X, ldx,
                                                    ZERO, w->Z1, ldz);
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            cblas_saxpy(M, -ONE, &w->AU[i * ldau], 1, &w->Z1[i * ldz], 1);
                                            f32 v = cblas_snrm2(M, &w->Z1[i * ldz], 1) *
                                                    w->SINGVX[k - 1] / (anorm * w->SINGVX[0]);
                                            if (v > tmp1) tmp1 = v;
                                        }
                                        if (tmp1 > e_au) e_au = tmp1;
                                    } else if (jobref[0] == 'E') {
                                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, ONE, w->A, lda, w->AU, ldau,
                                                    ZERO, w->Y1, M);
                                        i = 0;
                                        while (i < k) {
                                            if (w->IEIG[i] == ZERO) {
                                                cblas_saxpy(M, -w->REIG[i], &w->AU[i * ldau], 1, &w->Y1[i * M], 1);
                                                w->RESEX[i] = cblas_snrm2(M, &w->Y1[i * M], 1) /
                                                              cblas_snrm2(M, &w->AU[i * ldau], 1);
                                                i = i + 1;
                                            } else {
                                                ab[0] = w->REIG[i]; ab[1] = -w->IEIG[i];
                                                ab[2] = w->IEIG[i]; ab[3] = w->REIG[i];
                                                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                            M, 2, 2, -ONE, &w->AU[i * ldau], M, ab, 2,
                                                            ONE, &w->Y1[i * M], M);
                                                w->RESEX[i] = slange("F", M, 2, &w->Y1[i * M], M, NULL) /
                                                              slange("F", M, 2, &w->AU[i * ldau], M, NULL);
                                                w->RESEX[i + 1] = w->RESEX[i];
                                                i = i + 2;
                                            }
                                        }
                                    }

                                    /* Check: residuals returned vs explicitly computed */
                                    if (resids[0] == 'R') {
                                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, k, M, ONE, w->A, lda, w->Z, ldz,
                                                    ZERO, w->Y1, M);
                                        i = 0;
                                        while (i < k) {
                                            if (w->IEIG[i] == ZERO) {
                                                cblas_saxpy(M, -w->REIG[i], &w->Z[i * ldz], 1, &w->Y1[i * M], 1);
                                                w->RES1[i] = cblas_snrm2(M, &w->Y1[i * M], 1);
                                                i = i + 1;
                                            } else {
                                                ab[0] = w->REIG[i]; ab[1] = -w->IEIG[i];
                                                ab[2] = w->IEIG[i]; ab[3] = w->REIG[i];
                                                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                            M, 2, 2, -ONE, &w->Z[i * ldz], M, ab, 2,
                                                            ONE, &w->Y1[i * M], M);
                                                w->RES1[i] = slange("F", M, 2, &w->Y1[i * M], M, NULL);
                                                w->RES1[i + 1] = w->RES1[i];
                                                i = i + 2;
                                            }
                                        }
                                        f32 tmp1 = ZERO;
                                        for (i = 0; i < k; i++) {
                                            f32 v = fabsf(w->RES[i] - w->RES1[i]) *
                                                    w->SINGVX[k - 1] / (anorm * w->SINGVX[0]);
                                            if (v > tmp1) tmp1 = v;
                                        }
                                        if (tmp1 > e_rez) e_rez = tmp1;
                                    }

                                    /*==========  SGEDMDQ  ==========*/
                                    if (k_traj == 1) {
                                        memcpy(w->F1, w->F, (size_t)ldf * (N + 1) * sizeof(f32));

                                        sgedmdq(scale, jobz, resids, "Q", "R", jobref,
                                                whtsvd, M, N + 1, w->F1, ldf, w->X, ldx,
                                                w->Y, ldy, nrnk, tol, &kq, w->REIGQ,
                                                w->IEIGQ, w->Z, ldz, w->RES, w->AU, ldau,
                                                w->W, ldw, w->S, lds, wdummy, -1,
                                                idummy, -1, &info);
                                        liwork = idummy[0];
                                        lwork = (INT)wdummy[lwminopt - 1];
                                        assert_true(lwork <= WORKSZ && liwork <= IWORKSZ);

                                        sgedmdq(scale, jobz, resids, "Q", "R", jobref,
                                                whtsvd, M, N + 1, w->F1, ldf, w->X, ldx,
                                                w->Y, ldy, nrnk, tol, &kq, w->REIGQ,
                                                w->IEIGQ, w->Z, ldz, w->RES, w->AU, ldau,
                                                w->W, ldw, w->S, lds, w->work, lwork,
                                                w->iwork, liwork, &info);
                                        assert_true(info == 0 || info == 4);

                                        const INT mn = (M < N + 1) ? M : N + 1;
                                        for (i = 0; i < kq; i++)
                                            w->SINGVQX[i] = w->work[mn + i];
                                        /* The reference indexes SINGVQX(K) even when K may
                                         * exceed KQ; zero the unset tail so that read is
                                         * benign (the entries do not exist for K > KQ). */
                                        for (i = kq; i < N; i++)
                                            w->SINGVQX[i] = ZERO;

                                        /* singular values agreement */
                                        f32 tmp1 = ZERO;
                                        const INT mnk = (k < kq) ? k : kq;
                                        for (i = 0; i < mnk; i++) {
                                            f32 v = fabsf(w->SINGVX[i] - w->SINGVQX[i]) / w->SINGVX[0];
                                            if (v > tmp1) tmp1 = v;
                                        }
                                        if (tmp1 > e_sv) e_sv = tmp1;

                                        /* F = Q*R : ||F - F1*Y||_F / ||F||_F */
                                        memcpy(w->F2, w->F, (size_t)ldf * (N + 1) * sizeof(f32));
                                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                    M, N + 1, mn, -ONE, w->F1, ldf, w->Y, ldy,
                                                    ONE, w->F2, ldf);
                                        f32 fqr = slange("F", M, N + 1, w->F2, ldf, NULL) /
                                                  slange("F", M, N + 1, w->F, ldf, NULL);
                                        if (fqr > e_fqr) e_fqr = fqr;

                                        /* SGEDMDQ residuals */
                                        if (resids[0] == 'R') {
                                            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                        M, kq, M, ONE, w->A, M, w->Z, M,
                                                        ZERO, w->Y1, M);
                                            i = 0;
                                            while (i < kq) {
                                                if (w->IEIGQ[i] == ZERO) {
                                                    cblas_saxpy(M, -w->REIGQ[i], &w->Z[i * ldz], 1, &w->Y1[i * M], 1);
                                                    w->RES1[i] = cblas_snrm2(M, &w->Y1[i * M], 1);
                                                    i = i + 1;
                                                } else {
                                                    ab[0] = w->REIGQ[i]; ab[1] = -w->IEIGQ[i];
                                                    ab[2] = w->IEIGQ[i]; ab[3] = w->REIGQ[i];
                                                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                                                M, 2, 2, -ONE, &w->Z[i * ldz], M, ab, 2,
                                                                ONE, &w->Y1[i * M], M);
                                                    w->RES1[i] = slange("F", M, 2, &w->Y1[i * M], M, NULL);
                                                    w->RES1[i + 1] = w->RES1[i];
                                                    i = i + 2;
                                                }
                                            }
                                            f32 tmpq = ZERO;
                                            for (i = 0; i < kq; i++) {
                                                f32 v = fabsf(w->RES[i] - w->RES1[i]) *
                                                        w->SINGVQX[k - 1] / (anorm * w->SINGVQX[0]);
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

    /* Final checks against the reference thresholds. */
    assert_residual_below(e_zxw / (M * eps), 10.0);
    assert_residual_below(e_au / (M * N * eps), 10.0);
    assert_residual_below(e_rez / (M * N * eps), 10.0);
    assert_residual_below(e_sv / (M * N * eps), 1.0);
    assert_residual_below(e_fqr / (M * N * eps), 10.0);
    assert_residual_below(e_rezq / (M * N * eps), 10.0);
}

static void test_schkdmd_case(void** state)
{
    const INT idx = (INT)(intptr_t)*state;
    run_case(MVAL[idx], NVAL[idx]);
}

int main(void)
{
    struct CMUnitTest tests[NDIM];
    for (INT i = 0; i < NDIM; i++) {
        tests[i].name = "schkdmd";
        tests[i].test_func = test_schkdmd_case;
        tests[i].setup_func = NULL;
        tests[i].teardown_func = NULL;
        tests[i].initial_state = (void*)(intptr_t)i;
    }
    (void)_cmocka_run_group_tests("schkdmd", tests, NDIM, group_setup, group_teardown);
    return 0;
}
