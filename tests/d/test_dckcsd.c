/**
 * @file test_dckcsd.c
 * @brief CS decomposition test driver - port of LAPACK TESTING/EIG/dckcsd.f
 *
 * Tests DORCSD and DORCSD2BY1:
 *   the CSD for an M-by-M orthogonal matrix X partitioned as
 *   [ X11 X12; X21 X22 ]. X11 is P-by-Q.
 *
 * Test ratios (15 total):
 *
 * For 2-by-2 CSD (DORCSD):
 *   (1)  | U1'*X11*V1 - D11 | / ( max(P,Q) * eps2 )
 *   (2)  | U1'*X12*V2 - D12 | / ( max(P,M-Q) * eps2 )
 *   (3)  | U2'*X21*V1 - D21 | / ( max(M-P,Q) * eps2 )
 *   (4)  | U2'*X22*V2 - D22 | / ( max(M-P,M-Q) * eps2 )
 *   (5)  | I - U1'*U1 | / ( P * ulp )
 *   (6)  | I - U2'*U2 | / ( (M-P) * ulp )
 *   (7)  | I - V1T*V1T' | / ( Q * ulp )
 *   (8)  | I - V2T*V2T' | / ( (M-Q) * ulp )
 *   (9)  theta is nondecreasing and in [0, pi/2]
 *
 * For 2-by-1 CSD (DORCSD2BY1):
 *   (10) | U1'*X11*V1 - D11 | / ( max(P,Q) * eps2 )
 *   (11) | U2'*X21*V1 - D21 | / ( max(M-P,Q) * eps2 )
 *   (12) | I - U1'*U1 | / ( P * ulp )
 *   (13) | I - U2'*U2 | / ( (M-P) * ulp )
 *   (14) | I - V1T*V1T' | / ( Q * ulp )
 *   (15) theta is nondecreasing and in [0, pi/2]
 *
 * Matrix types: 4 types (NTYPES = 4)
 *   Type 1: Random orthogonal matrix via DLAROR
 *   Type 2: Constructed CSD with random angles + noise
 *   Type 3: Constructed CSD with gap-separated angles
 *   Type 4: Random permutation matrix
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include "semicolon_cblas.h"

#define THRESH 30.0
#define NTYPES 4
#define NTESTS 15

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))

static const f64 PIOVER2 = 1.57079632679489661923132169163975144210e0;
static const f64 GAPDIGIT = 18.0;
static const f64 ORTH = 1.0e-12;

/* (M,P,Q) triplets from csd.in */
static const INT MVAL[] = {0, 10, 10, 10, 10, 21, 24, 30, 22, 32, 55};
static const INT PVAL[] = {0,  4,  4,  0, 10,  9, 10, 20, 12, 12, 40};
static const INT QVAL[] = {0,  0, 10,  4,  4, 15, 12,  8, 20,  8, 20};
#define NSIZES (sizeof(MVAL) / sizeof(MVAL[0]))

/* External function declarations */
/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT im;       /* index into MVAL/PVAL/QVAL */
    INT imat;     /* matrix type 1..4 */
    char name[128];
} dckcsd_params_t;

typedef struct {
    INT mmax;
    INT ldx;
    f64* X;
    f64* XF;
    f64* U1;
    f64* U2;
    f64* V1T;
    f64* V2T;
    f64* theta;
    INT* iwork;
    f64* work;
    INT lwork;
    f64* rwork;
    uint64_t rng_state[4];
} dckcsd_workspace_t;

static dckcsd_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* DLACSG: Construct orthogonal matrix with prescribed CS angles         */
/* Port of DLACSG embedded in dckcsd.f (lines 350-402)                   */
/* ===================================================================== */

static void dlacsg(const INT m, const INT p, const INT q,
                   const f64* theta, f64* X, const INT ldx,
                   f64* work, uint64_t rng[static 4])
{
    INT r = MIN4(p, m - p, q, m - q);
    INT info;

    dlaset("Full", m, m, 0.0, 0.0, X, ldx);

    for (INT i = 0; i < MIN(p, q) - r; i++)
        X[i + i * ldx] = 1.0;
    for (INT i = 0; i < r; i++)
        X[(MIN(p, q) - r + i) + (MIN(p, q) - r + i) * ldx] = cos(theta[i]);

    for (INT i = 0; i < MIN(p, m - q) - r; i++)
        X[(p - 1 - i) + (m - 1 - i) * ldx] = -1.0;
    for (INT i = 0; i < r; i++) {
        INT k = MIN(p, m - q) - r;
        X[(p - k - 1 - i) + (m - k - 1 - i) * ldx] = -sin(theta[r - 1 - i]);
    }

    for (INT i = 0; i < MIN(m - p, q) - r; i++)
        X[(m - 1 - i) + (q - 1 - i) * ldx] = 1.0;
    for (INT i = 0; i < r; i++) {
        INT k2 = MIN(m - p, q) - r;
        X[(m - k2 - 1 - i) + (q - k2 - 1 - i) * ldx] = sin(theta[r - 1 - i]);
    }

    for (INT i = 0; i < MIN(m - p, m - q) - r; i++)
        X[(p + i) + (q + i) * ldx] = 1.0;
    for (INT i = 0; i < r; i++) {
        INT k3 = MIN(m - p, m - q) - r;
        X[(p + k3 + i) + (q + k3 + i) * ldx] = cos(theta[i]);
    }

    dlaror("Left", "No init", p, m, X, ldx, work, &info, rng);
    dlaror("Left", "No init", m - p, m, &X[p], ldx, work, &info, rng);
    dlaror("Right", "No init", m, q, X, ldx, work, &info, rng);
    dlaror("Right", "No init", m, m - q, &X[q * ldx], ldx, work, &info, rng);
}

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dckcsd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = 1;
    for (INT i = 0; i < (INT)NSIZES; i++) {
        if (MVAL[i] > g_ws->mmax) g_ws->mmax = MVAL[i];
    }

    INT mmax = g_ws->mmax;
    g_ws->ldx = mmax;
    if (g_ws->ldx < 1) g_ws->ldx = 1;
    INT ldx = g_ws->ldx;

    g_ws->X     = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->XF    = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->U1    = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->U2    = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->V1T   = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->V2T   = malloc((size_t)ldx * mmax * sizeof(f64));
    g_ws->theta = malloc((size_t)mmax * sizeof(f64));
    g_ws->iwork = malloc((size_t)mmax * sizeof(INT));
    g_ws->rwork = malloc((size_t)mmax * sizeof(f64));

    g_ws->lwork = mmax * mmax;
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(f64));

    if (!g_ws->X || !g_ws->XF || !g_ws->U1 || !g_ws->U2 ||
        !g_ws->V1T || !g_ws->V2T || !g_ws->theta || !g_ws->iwork ||
        !g_ws->rwork || !g_ws->work) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xC5D15EEDULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->X);
        free(g_ws->XF);
        free(g_ws->U1);
        free(g_ws->U2);
        free(g_ws->V1T);
        free(g_ws->V2T);
        free(g_ws->theta);
        free(g_ws->iwork);
        free(g_ws->rwork);
        free(g_ws->work);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_dckcsd_single(dckcsd_params_t* params)
{
    const INT m = MVAL[params->im];
    const INT p = PVAL[params->im];
    const INT q = QVAL[params->im];
    const INT imat = params->imat;

    INT ldx   = g_ws->ldx;
    f64* X    = g_ws->X;
    f64* XF   = g_ws->XF;
    f64* U1   = g_ws->U1;
    f64* U2   = g_ws->U2;
    f64* V1T  = g_ws->V1T;
    f64* V2T  = g_ws->V2T;
    f64* theta = g_ws->theta;
    INT* iwork = g_ws->iwork;
    f64* work  = g_ws->work;
    INT lwork  = g_ws->lwork;
    f64* rwork = g_ws->rwork;
    uint64_t* rng = g_ws->rng_state;

    INT iinfo;
    char ctx[256];

    if (imat == 1) {
        dlaror("Left", "I", m, m, X, ldx, work, &iinfo, rng);
        if (m != 0 && iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DLAROR in DCKCSD: M=%d, INFO=%d", m, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            return;
        }
    } else if (imat == 2) {
        INT r = MIN4(p, m - p, q, m - q);
        for (INT i = 0; i < r; i++)
            theta[i] = PIOVER2 * rng_uniform(rng);
        dlacsg(m, p, q, theta, X, ldx, work, rng);
        for (INT i = 0; i < m; i++)
            for (INT j = 0; j < m; j++)
                X[i + j * ldx] += ORTH * rng_uniform_symmetric(rng);
    } else if (imat == 3) {
        INT r = MIN4(p, m - p, q, m - q);
        for (INT i = 0; i < r + 1; i++)
            theta[i] = pow(10.0, -rng_uniform(rng) * GAPDIGIT);
        for (INT i = 1; i < r + 1; i++)
            theta[i] = theta[i - 1] + theta[i];
        for (INT i = 0; i < r; i++)
            theta[i] = PIOVER2 * theta[i] / theta[r];
        dlacsg(m, p, q, theta, X, ldx, work, rng);
    } else {
        dlaset("Full", m, m, 0.0, 1.0, X, ldx);
        for (INT i = 0; i < m; i++) {
            INT j = (INT)(rng_uniform(rng) * m);
            if (j != i) {
                cblas_drot(m, &X[i * ldx], 1, &X[j * ldx], 1, 0.0, 1.0);
            }
        }
    }

    f64 result[NTESTS];

    dcsdts(m, p, q, X, XF, ldx, U1, ldx, U2, ldx,
           V1T, ldx, V2T, ldx, theta, iwork,
           work, lwork, rwork, result);

    for (INT i = 0; i < NTESTS; i++) {
        snprintf(ctx, sizeof(ctx),
                 "M=%d P=%d Q=%d type=%d test=%d ratio=%.6e",
                 m, p, q, imat, i + 1, result[i]);
        set_test_context(ctx);
        assert_residual_ok(result[i]);
    }
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_dckcsd(void** state)
{
    (void)state;
    dckcsd_params_t* params = (dckcsd_params_t*)(*state);
    run_dckcsd_single(params);
}

/* ===================================================================== */
/* Main: build parametrized test list                                    */
/* ===================================================================== */

int main(void)
{
    static dckcsd_params_t all_params[NSIZES * NTYPES];
    static struct CMUnitTest all_tests[NSIZES * NTYPES];
    INT ntests = 0;

    for (INT im = 0; im < (INT)NSIZES; im++) {
        for (INT imat = 1; imat <= NTYPES; imat++) {
            dckcsd_params_t* p = &all_params[ntests];
            p->im = im;
            p->imat = imat;
            snprintf(p->name, sizeof(p->name),
                     "dckcsd_m%d_p%d_q%d_type%d",
                     MVAL[im], PVAL[im], QVAL[im], imat);

            all_tests[ntests] = (struct CMUnitTest){
                .name = p->name,
                .test_func = test_dckcsd,
                .initial_state = p,
                .setup_func = NULL,
                .teardown_func = NULL,
            };
            ntests++;
        }
    }

    return _cmocka_run_group_tests("dckcsd", all_tests, ntests,
                                    group_setup, group_teardown);
}
