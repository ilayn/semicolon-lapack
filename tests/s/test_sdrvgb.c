/**
 * @file test_sdrvgb.c
 * @brief DDRVGB tests the driver routines SGBSV and SGBSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvgb.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  8
#define NTESTS  7
#define NTRAN   3
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2
#define NBW     4

/* Maximum workspace dimensions */
#define LA      ((2*NMAX - 1) * NMAX)
#define LAFB    ((3*NMAX - 2) * NMAX)

/* Routines under test */
/* Supporting routines */
/* Verification routines */
/* Matrix generation */
/* Utilities */
typedef struct {
    INT n;
    INT kl;
    INT ku;
    INT imat;
    INT ifact;     /* 0='F', 1='N', 2='E' */
    INT itran;     /* 0='N', 1='T', 2='C' */
    INT iequed;    /* 0='N', 1='R', 2='C', 3='B' */
    char name[96];
} ddrvgb_params_t;

typedef struct {
    f32* A;
    f32* AFB;
    f32* ASAV;
    f32* B;
    f32* BSAV;
    f32* X;
    f32* XACT;
    f32* S;
    f32* WORK;
    f32* RWORK;
    INT* IWORK;
} ddrvgb_workspace_t;

static ddrvgb_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvgb_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;
    if (lwork < 3 * NMAX) lwork = 3 * NMAX;
    if (lwork < NMAX * NRHS) lwork = NMAX * NRHS;

    g_workspace->A     = calloc(LA, sizeof(f32));
    g_workspace->AFB   = calloc(LAFB, sizeof(f32));
    g_workspace->ASAV  = calloc(LA, sizeof(f32));
    g_workspace->B     = calloc(NMAX * NRHS, sizeof(f32));
    g_workspace->BSAV  = calloc(NMAX * NRHS, sizeof(f32));
    g_workspace->X     = calloc(NMAX * NRHS, sizeof(f32));
    g_workspace->XACT  = calloc(NMAX * NRHS, sizeof(f32));
    g_workspace->S     = calloc(2 * NMAX, sizeof(f32));
    g_workspace->WORK  = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f32));
    g_workspace->IWORK = calloc(2 * NMAX, sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFB || !g_workspace->ASAV ||
        !g_workspace->B || !g_workspace->BSAV || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->S || !g_workspace->WORK ||
        !g_workspace->RWORK || !g_workspace->IWORK) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AFB);
        free(g_workspace->ASAV);
        free(g_workspace->B);
        free(g_workspace->BSAV);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->S);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Copy band matrix A (lda rows) into AFB (ldafb rows) with KL row offset.
 * A has rows 0..kl+ku, AFB has rows kl..2*kl+ku for the same data.
 */
static void copy_band_to_factor(const f32* A, INT lda,
                                 f32* AFB, INT ldafb,
                                 INT kl, INT ku, INT n)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < kl + ku + 1; i++) {
            AFB[(kl + i) + j * ldafb] = A[i + j * lda];
        }
    }
}

static void run_ddrvgb_single(INT n, INT kl, INT ku, INT imat,
                               INT ifact, INT itran, INT iequed)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    ddrvgb_workspace_t* ws = g_workspace;
    const char* fact  = FACTS[ifact];
    const char* trans = TRANSS[itran];
    char equed = EQUEDS[iequed][0];

    INT prefac = (fact[0] == 'F');
    INT nofact = (fact[0] == 'N');
    INT equil  = (fact[0] == 'E');

    INT lda   = kl + ku + 1;
    INT ldafb = 2 * kl + ku + 1;
    INT ldb   = (n > 1) ? n : 1;

    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    /* Set up parameters with SLATB4 */
    char type, dist;
    INT kl_out = kl, ku_out = ku, mode;
    f32 anorm, cndnum;
    slatb4("SGB", imat, n, n, &type, &kl_out, &ku_out, &anorm, &mode,
           &cndnum, &dist);

    /* Generate test matrix with SLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat);
    INT info;
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "Z", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("SLATMS info=%d (n=%d kl=%d ku=%d imat=%d)", info, n, kl, ku, imat);
        return;
    }

    /* For types 2-4, zero one or more columns */
    INT izero = 0;
    INT zerot = (imat >= 2 && imat <= 4);
    if (zerot) {
        if (imat == 2)
            izero = 1;
        else if (imat == 3)
            izero = n;
        else
            izero = n / 2 + 1;

        if (imat < 4) {
            /* Zero single column IZERO (1-based) */
            INT ioff = (izero - 1) * lda;
            INT i1_f = ku + 2 - izero;
            INT i2_f = ku + 1 + (n - izero);
            INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
            INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (INT i = i1; i < i2; i++)
                ws->A[ioff + i] = 0.0f;
        } else {
            /* Zero columns IZERO..N (1-based) */
            INT ioff = (izero - 1) * lda;
            for (INT j = izero; j <= n; j++) {
                INT i1_f = ku + 2 - j;
                INT i2_f = ku + 1 + (n - j);
                INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
                INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
                for (INT i = i1; i < i2; i++)
                    ws->A[ioff + i] = 0.0f;
                ioff += lda;
            }
        }
    }

    /* Save a copy of the matrix A in ASAV */
    slacpy("Full", kl + ku + 1, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) {
        return;
    }

    f32 rcondo = 0.0f, rcondi = 0.0f;
    f32 roldo = 0.0f, roldi = 0.0f;
    f32 rowcnd = 0.0f, colcnd = 0.0f, amax = 0.0f;

    if (zerot) {
        rcondo = 0.0f;
        rcondi = 0.0f;
    } else {
        /*
         * Compute condition numbers for verification.
         *
         * In the Fortran, FACT='N' reuses condition numbers from the
         * previous FACT='F' iteration. Since our tests are independent,
         * we always compute the non-equilibrated condition first, then
         * the equilibrated condition if needed.
         */

        /* First compute condition of non-equilibrated matrix */
        copy_band_to_factor(ws->ASAV, lda, ws->AFB, ldafb, kl, ku, n);

        f32 anormo = slangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
        f32 anormi = slangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

        sgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

        /* Form the inverse of A: solve A * X = I */
        slaset("Full", n, n, 0.0f, 1.0f, ws->WORK, ldb);
        sgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, ldb, &info);

        f32 ainvnm = slange("1", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormo <= 0.0f || ainvnm <= 0.0f)
            rcondo = 1.0f;
        else
            rcondo = (1.0f / anormo) / ainvnm;

        ainvnm = slange("I", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormi <= 0.0f || ainvnm <= 0.0f)
            rcondi = 1.0f;
        else
            rcondi = (1.0f / anormi) / ainvnm;

        /* Save non-equilibrated condition for ROLDO/ROLDI */
        roldo = rcondo;
        roldi = rcondi;

        /* If equilibration is needed, compute equilibrated condition */
        if (equil || iequed > 0) {
            copy_band_to_factor(ws->ASAV, lda, ws->AFB, ldafb, kl, ku, n);

            sgbequ(n, n, kl, ku, &ws->AFB[kl], ldafb,
                   ws->S, &ws->S[n], &rowcnd, &colcnd, &amax, &info);
            if (info == 0 && n > 0) {
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0f; colcnd = 1.0f;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0f; colcnd = 0.0f;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0f; colcnd = 0.0f;
                }

                slaqgb(n, n, kl, ku, &ws->AFB[kl], ldafb,
                       ws->S, &ws->S[n], rowcnd, colcnd, amax, &equed);
            }

            /* Compute condition of equilibrated matrix */
            anormo = slangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
            anormi = slangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

            sgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

            slaset("Full", n, n, 0.0f, 1.0f, ws->WORK, ldb);
            sgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
                   ws->IWORK, ws->WORK, ldb, &info);

            ainvnm = slange("1", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormo <= 0.0f || ainvnm <= 0.0f)
                rcondo = 1.0f;
            else
                rcondo = (1.0f / anormo) / ainvnm;

            ainvnm = slange("I", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormi <= 0.0f || ainvnm <= 0.0f)
                rcondi = 1.0f;
            else
                rcondi = (1.0f / anormi) / ainvnm;
        }
    }

    /* Select condition number based on TRANS */
    f32 rcondc = (itran == 0) ? rcondo : rcondi;

    /* Restore the matrix A */
    slacpy("Full", kl + ku + 1, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side */
    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat + itran);
    char xtype = 'N';
    slarhs("SGB", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
    slacpy("Full", n, NRHS, ws->B, ldb, ws->BSAV, ldb);

    /* --- Test SGBSV --- */
    if (nofact && itran == 0) {
        copy_band_to_factor(ws->A, lda, ws->AFB, ldafb, kl, ku, n);
        slacpy("Full", n, NRHS, ws->B, ldb, ws->X, ldb);

        sgbsv(n, kl, ku, NRHS, ws->AFB, ldafb, ws->IWORK, ws->X, ldb, &info);

        if (info != izero) {
            fail_msg("SGBSV info=%d expected=%d (n=%d kl=%d ku=%d imat=%d)",
                     info, izero, n, kl, ku, imat);
        }

        /* TEST 1: Reconstruct matrix from factors */
        sgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);

        INT nt = 1;
        if (izero == 0) {
            /* TEST 2: Compute residual of computed solution */
            slacpy("Full", n, NRHS, ws->B, ldb, ws->WORK, ldb);
            sgbt02("N", n, n, kl, ku, NRHS, ws->A, lda,
                   ws->X, ldb, ws->WORK, ldb, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, NRHS, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);
            nt = 3;
        }

        for (INT k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SGBSV n=%d kl=%d ku=%d type %d test %d: result=%e >= thresh=%e",
                         n, kl, ku, imat, k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test SGBSVX --- */

    /* Zero AFB if not prefactored */
    if (!prefac)
        slaset("Full", 2 * kl + ku + 1, n, 0.0f, 0.0f, ws->AFB, ldafb);
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, ldb);

    /* Equilibrate matrix if IEQUED > 0 */
    if (iequed > 0 && n > 0) {
        slaqgb(n, n, kl, ku, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed);
    }

    /* Call SGBSVX */
    char equed_inout = equed;
    f32 rcond;
    sgbsvx(fact, trans, n, kl, ku, NRHS, ws->A, lda, ws->AFB, ldafb,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, ldb, ws->X, ldb, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->IWORK[n], &info);

    /* Check error code */
    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("SGBSVX info=%d expected=%d (n=%d kl=%d ku=%d imat=%d fact=%s trans=%s)",
                     info, izero, n, kl, ku, imat, fact, trans);
        }
    }

    /*
     * TEST 7: Compare RPVGRW from SGBSVX (work[0]) with computed value.
     * Save work[0] before it could be overwritten by slantb/slangb.
     * Note: 'M' norm does not use the work array, so work[0] is preserved,
     * but we save it defensively.
     */
    f32 rpvgrw_svx = ws->WORK[0];
    f32 rpvgrw;

    if (info != 0 && info <= n) {
        /* Singularity detected at column INFO (1-based) */
        f32 anrmpv = 0.0f;
        for (INT j = 0; j < info; j++) {
            /* Fortran: DO I = MAX(KU+2-J, 1), MIN(N+KU+1-J, KL+KU+1) */
            INT i1_f = ku + 2 - (j + 1);
            INT i2_f = n + ku + 1 - (j + 1);
            INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
            INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (INT i = i1; i < i2; i++) {
                f32 val = fabsf(ws->A[i + j * lda]);
                if (val > anrmpv) anrmpv = val;
            }
        }
        /* Fortran: SLANTB('M','U','N',INFO,MIN(INFO-1,KL+KU),
         *                  AFB(MAX(1,KL+KU+2-INFO)),LDAFB,WORK) */
        INT kband = (info - 1 < kl + ku) ? info - 1 : kl + ku;
        INT afb_off_f = kl + ku + 2 - info;
        INT afb_off = (afb_off_f > 1) ? afb_off_f - 1 : 0;
        rpvgrw = slantb("M", "U", "N", info, kband,
                        &ws->AFB[afb_off], ldafb, ws->WORK);
        if (rpvgrw == 0.0f)
            rpvgrw = 1.0f;
        else
            rpvgrw = anrmpv / rpvgrw;
    } else {
        rpvgrw = slantb("M", "U", "N", n, kl + ku,
                        ws->AFB, ldafb, ws->WORK);
        if (rpvgrw == 0.0f) {
            rpvgrw = 1.0f;
        } else {
            rpvgrw = slangb("M", n, kl, ku, ws->A, lda, ws->WORK) / rpvgrw;
        }
    }
    f32 denom = fmaxf(rpvgrw_svx, rpvgrw);
    if (denom > 0.0f)
        result[6] = fabsf(rpvgrw - rpvgrw_svx) / denom / slamch("E");
    else
        result[6] = 0.0f;

    /* TEST 1: Reconstruct matrix from factors */
    INT k1;
    if (!prefac) {
        sgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    INT trfcon;
    if (info == 0) {
        trfcon = 0;

        /* TEST 2: Compute residual of computed solution */
        slacpy("Full", n, NRHS, ws->BSAV, ldb, ws->WORK, ldb);
        sgbt02(trans, n, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->X, ldb, ws->WORK, ldb, &ws->RWORK[2 * NRHS],
               &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        f32 rcond_for_get04;
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            f32 roldc = (itran == 0) ? roldo : roldi;
            rcond_for_get04 = roldc;
        }
        sget04(n, NRHS, ws->X, ldb, ws->XACT, ldb,
               rcond_for_get04, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        sgbt05(trans, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->B, ldb, ws->X, ldb, ws->XACT, ldb,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    /* TEST 6: Compare RCOND from SGBSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    if (!trfcon) {
        for (INT k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                if (prefac) {
                    fail_msg("SGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d EQUED=%c "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku, equed_inout,
                             imat, k + 1, (double)result[k], (double)THRESH);
                } else {
                    fail_msg("SGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku,
                             imat, k + 1, (double)result[k], (double)THRESH);
                }
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("SGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 1: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, (double)result[0], (double)THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("SGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 6: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, (double)result[5], (double)THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("SGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 7: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, (double)result[6], (double)THRESH);
        }
    }
}

static void test_ddrvgb_case(void** state)
{
    ddrvgb_params_t* p = *state;
    run_ddrvgb_single(p->n, p->kl, p->ku, p->imat,
                      p->ifact, p->itran, p->iequed);
}

/**
 * Compute KL bandwidth values for a given N.
 * Order: 0, N-1, (3N-1)/4, (N+1)/4
 */
static void get_klku_values(INT n, INT vals[NBW])
{
    vals[0] = 0;
    vals[1] = (n > 0) ? n - 1 : 0;
    vals[2] = (3 * n - 1) / 4;
    vals[3] = (n + 1) / 4;
}

/**
 * Count total test cases and optionally fill parameter/test arrays.
 * If params and tests are NULL, only counting is done.
 */
static INT build_test_array(ddrvgb_params_t* params, struct CMUnitTest* tests)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    INT idx = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nkl = (n > 4) ? 4 : ((n > 0) ? n : 1);
        INT nku = nkl;
        INT nimat = (n <= 0) ? 1 : NTYPES;

        INT klval[NBW], kuval[NBW];
        get_klku_values(n, klval);
        get_klku_values(n, kuval);

        for (INT ikl = 0; ikl < nkl; ikl++) {
            INT kl = klval[ikl];
            for (INT iku = 0; iku < nku; iku++) {
                INT ku = kuval[iku];
                INT lda   = kl + ku + 1;
                INT ldafb = 2 * kl + ku + 1;

                /* Check that workspace is big enough */
                if (lda * n > LA || ldafb * n > LAFB)
                    continue;

                for (INT imat = 1; imat <= nimat; imat++) {
                    INT zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1)
                        continue;

                    for (INT iequed = 0; iequed < 4; iequed++) {
                        INT nfact = (iequed == 0) ? 3 : 1;

                        for (INT ifact = 0; ifact < nfact; ifact++) {
                            /* Skip FACT='F' for singular matrices */
                            if (zerot && ifact == 0)
                                continue;

                            for (INT itran = 0; itran < NTRAN; itran++) {
                                if (params && tests) {
                                    ddrvgb_params_t* p = &params[idx];
                                    p->n      = n;
                                    p->kl     = kl;
                                    p->ku     = ku;
                                    p->imat   = imat;
                                    p->ifact  = ifact;
                                    p->itran  = itran;
                                    p->iequed = iequed;
                                    snprintf(p->name, sizeof(p->name),
                                             "n%d_kl%d_ku%d_t%d_%s_%s_%s",
                                             n, kl, ku, imat,
                                             FACTS[ifact], TRANSS[itran],
                                             EQUEDS[iequed]);

                                    tests[idx].name = p->name;
                                    tests[idx].test_func = test_ddrvgb_case;
                                    tests[idx].setup_func = NULL;
                                    tests[idx].teardown_func = NULL;
                                    tests[idx].initial_state = p;
                                }
                                idx++;
                            }
                        }
                    }
                }
            }
        }
    }
    return idx;
}

int main(void)
{
    INT count = build_test_array(NULL, NULL);
    if (count == 0) return 0;

    ddrvgb_params_t* params = malloc(count * sizeof(*params));
    struct CMUnitTest* tests = malloc(count * sizeof(*tests));
    if (!params || !tests) {
        free(params);
        free(tests);
        return 1;
    }

    build_test_array(params, tests);
    INT result = _cmocka_run_group_tests("ddrvgb", tests, count,
                                          group_setup, group_teardown);
    free(tests);
    free(params);
    return result;
}
