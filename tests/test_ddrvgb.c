/**
 * @file test_ddrvgb.c
 * @brief DDRVGB tests the driver routines DGBSV and DGBSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvgb.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  8
#define NTESTS  7
#define NTRAN   3
#define THRESH  30.0
#define NMAX    50
#define NRHS    2
#define NBW     4

/* Maximum workspace dimensions */
#define LA      ((2*NMAX - 1) * NMAX)
#define LAFB    ((3*NMAX - 2) * NMAX)

/* Routines under test */
extern void dgbsv(const int n, const int kl, const int ku, const int nrhs,
                   f64* AB, const int ldab, int* ipiv,
                   f64* B, const int ldb, int* info);
extern void dgbsvx(const char* fact, const char* trans, const int n,
                    const int kl, const int ku, const int nrhs,
                    f64* AB, const int ldab, f64* AFB, const int ldafb,
                    int* ipiv, char* equed, f64* R, f64* C,
                    f64* B, const int ldb, f64* X, const int ldx,
                    f64* rcond, f64* ferr, f64* berr,
                    f64* work, int* iwork, int* info);

/* Supporting routines */
extern void dgbtrf(const int m, const int n, const int kl, const int ku,
                    f64* AB, const int ldab, int* ipiv, int* info);
extern void dgbtrs(const char* trans, const int n, const int kl, const int ku,
                    const int nrhs, const f64* AB, const int ldab,
                    const int* ipiv, f64* B, const int ldb, int* info);
extern void dgbequ(const int m, const int n, const int kl, const int ku,
                    const f64* AB, const int ldab,
                    f64* R, f64* C, f64* rowcnd, f64* colcnd,
                    f64* amax, int* info);
extern void dlaqgb(const int m, const int n, const int kl, const int ku,
                    f64* AB, const int ldab,
                    const f64* R, const f64* C,
                    const f64 rowcnd, const f64 colcnd,
                    const f64 amax, char* equed);
extern f64 dlangb(const char* norm, const int n, const int kl, const int ku,
                     const f64* AB, const int ldab, f64* work);
extern f64 dlantb(const char* norm, const char* uplo, const char* diag,
                     const int n, const int k, const f64* AB, const int ldab,
                     f64* work);

/* Verification routines */
extern void dgbt01(const int m, const int n, const int kl, const int ku,
                    const f64* A, const int lda,
                    const f64* AFAC, const int ldafac,
                    const int* ipiv, f64* work, f64* resid);
extern void dgbt02(const char* trans, const int m, const int n,
                    const int kl, const int ku, const int nrhs,
                    const f64* A, const int lda,
                    const f64* X, const int ldx,
                    f64* B, const int ldb,
                    f64* rwork, f64* resid);
extern void dgbt05(const char* trans, const int n, const int kl, const int ku,
                    const int nrhs,
                    const f64* AB, const int ldab,
                    const f64* B, const int ldb,
                    const f64* X, const int ldx,
                    const f64* XACT, const int ldxact,
                    const f64* ferr, const f64* berr,
                    f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                    const f64* XACT, const int ldxact, const f64 rcond,
                    f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                    char* type, int* kl, int* ku, f64* anorm, int* mode,
                    f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                    const char* sym, f64* d,
                    const int mode, const f64 cond, const f64 dmax,
                    const int kl, const int ku, const char* pack,
                    f64* A, const int lda, f64* work, int* info,
                    uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                    const char* trans, const int m, const int n,
                    const int kl, const int ku, const int nrhs,
                    const f64* A, const int lda, const f64* XACT, const int ldxact,
                    f64* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                    const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                    const f64 alpha, const f64 beta,
                    f64* A, const int lda);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern f64 dlamch(const char* cmach);

typedef struct {
    int n;
    int kl;
    int ku;
    int imat;
    int ifact;     /* 0='F', 1='N', 2='E' */
    int itran;     /* 0='N', 1='T', 2='C' */
    int iequed;    /* 0='N', 1='R', 2='C', 3='B' */
    char name[96];
} ddrvgb_params_t;

typedef struct {
    f64* A;
    f64* AFB;
    f64* ASAV;
    f64* B;
    f64* BSAV;
    f64* X;
    f64* XACT;
    f64* S;
    f64* WORK;
    f64* RWORK;
    int* IWORK;
} ddrvgb_workspace_t;

static ddrvgb_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvgb_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * NMAX;
    if (lwork < 3 * NMAX) lwork = 3 * NMAX;
    if (lwork < NMAX * NRHS) lwork = NMAX * NRHS;

    g_workspace->A     = calloc(LA, sizeof(f64));
    g_workspace->AFB   = calloc(LAFB, sizeof(f64));
    g_workspace->ASAV  = calloc(LA, sizeof(f64));
    g_workspace->B     = calloc(NMAX * NRHS, sizeof(f64));
    g_workspace->BSAV  = calloc(NMAX * NRHS, sizeof(f64));
    g_workspace->X     = calloc(NMAX * NRHS, sizeof(f64));
    g_workspace->XACT  = calloc(NMAX * NRHS, sizeof(f64));
    g_workspace->S     = calloc(2 * NMAX, sizeof(f64));
    g_workspace->WORK  = calloc(lwork, sizeof(f64));
    g_workspace->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f64));
    g_workspace->IWORK = calloc(2 * NMAX, sizeof(int));

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
static void copy_band_to_factor(const f64* A, int lda,
                                 f64* AFB, int ldafb,
                                 int kl, int ku, int n)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < kl + ku + 1; i++) {
            AFB[(kl + i) + j * ldafb] = A[i + j * lda];
        }
    }
}

static void run_ddrvgb_single(int n, int kl, int ku, int imat,
                               int ifact, int itran, int iequed)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    ddrvgb_workspace_t* ws = g_workspace;
    const char* fact  = FACTS[ifact];
    const char* trans = TRANSS[itran];
    char equed = EQUEDS[iequed][0];

    int prefac = (fact[0] == 'F');
    int nofact = (fact[0] == 'N');
    int equil  = (fact[0] == 'E');

    int lda   = kl + ku + 1;
    int ldafb = 2 * kl + ku + 1;
    int ldb   = (n > 1) ? n : 1;

    f64 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl_out = kl, ku_out = ku, mode;
    f64 anorm, cndnum;
    dlatb4("DGB", imat, n, n, &type, &kl_out, &ku_out, &anorm, &mode,
           &cndnum, &dist);

    /* Generate test matrix with DLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat);
    int info;
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "Z", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("DLATMS info=%d (n=%d kl=%d ku=%d imat=%d)", info, n, kl, ku, imat);
        return;
    }

    /* For types 2-4, zero one or more columns */
    int izero = 0;
    int zerot = (imat >= 2 && imat <= 4);
    if (zerot) {
        if (imat == 2)
            izero = 1;
        else if (imat == 3)
            izero = n;
        else
            izero = n / 2 + 1;

        if (imat < 4) {
            /* Zero single column IZERO (1-based) */
            int ioff = (izero - 1) * lda;
            int i1_f = ku + 2 - izero;
            int i2_f = ku + 1 + (n - izero);
            int i1 = (i1_f > 1) ? i1_f - 1 : 0;
            int i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (int i = i1; i < i2; i++)
                ws->A[ioff + i] = 0.0;
        } else {
            /* Zero columns IZERO..N (1-based) */
            int ioff = (izero - 1) * lda;
            for (int j = izero; j <= n; j++) {
                int i1_f = ku + 2 - j;
                int i2_f = ku + 1 + (n - j);
                int i1 = (i1_f > 1) ? i1_f - 1 : 0;
                int i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
                for (int i = i1; i < i2; i++)
                    ws->A[ioff + i] = 0.0;
                ioff += lda;
            }
        }
    }

    /* Save a copy of the matrix A in ASAV */
    dlacpy("Full", kl + ku + 1, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) {
        return;
    }

    f64 rcondo = 0.0, rcondi = 0.0;
    f64 roldo = 0.0, roldi = 0.0;
    f64 rowcnd = 0.0, colcnd = 0.0, amax = 0.0;

    if (zerot) {
        rcondo = 0.0;
        rcondi = 0.0;
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

        f64 anormo = dlangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
        f64 anormi = dlangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

        dgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

        /* Form the inverse of A: solve A * X = I */
        dlaset("Full", n, n, 0.0, 1.0, ws->WORK, ldb);
        dgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, ldb, &info);

        f64 ainvnm = dlange("1", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormo <= 0.0 || ainvnm <= 0.0)
            rcondo = 1.0;
        else
            rcondo = (1.0 / anormo) / ainvnm;

        ainvnm = dlange("I", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormi <= 0.0 || ainvnm <= 0.0)
            rcondi = 1.0;
        else
            rcondi = (1.0 / anormi) / ainvnm;

        /* Save non-equilibrated condition for ROLDO/ROLDI */
        roldo = rcondo;
        roldi = rcondi;

        /* If equilibration is needed, compute equilibrated condition */
        if (equil || iequed > 0) {
            copy_band_to_factor(ws->ASAV, lda, ws->AFB, ldafb, kl, ku, n);

            dgbequ(n, n, kl, ku, &ws->AFB[kl], ldafb,
                   ws->S, &ws->S[n], &rowcnd, &colcnd, &amax, &info);
            if (info == 0 && n > 0) {
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0; colcnd = 1.0;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0; colcnd = 0.0;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0; colcnd = 0.0;
                }

                dlaqgb(n, n, kl, ku, &ws->AFB[kl], ldafb,
                       ws->S, &ws->S[n], rowcnd, colcnd, amax, &equed);
            }

            /* Compute condition of equilibrated matrix */
            anormo = dlangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
            anormi = dlangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

            dgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

            dlaset("Full", n, n, 0.0, 1.0, ws->WORK, ldb);
            dgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
                   ws->IWORK, ws->WORK, ldb, &info);

            ainvnm = dlange("1", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormo <= 0.0 || ainvnm <= 0.0)
                rcondo = 1.0;
            else
                rcondo = (1.0 / anormo) / ainvnm;

            ainvnm = dlange("I", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormi <= 0.0 || ainvnm <= 0.0)
                rcondi = 1.0;
            else
                rcondi = (1.0 / anormi) / ainvnm;
        }
    }

    /* Select condition number based on TRANS */
    f64 rcondc = (itran == 0) ? rcondo : rcondi;

    /* Restore the matrix A */
    dlacpy("Full", kl + ku + 1, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side */
    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat + itran);
    char xtype = 'N';
    dlarhs("DGB", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
    dlacpy("Full", n, NRHS, ws->B, ldb, ws->BSAV, ldb);

    /* --- Test DGBSV --- */
    if (nofact && itran == 0) {
        copy_band_to_factor(ws->A, lda, ws->AFB, ldafb, kl, ku, n);
        dlacpy("Full", n, NRHS, ws->B, ldb, ws->X, ldb);

        dgbsv(n, kl, ku, NRHS, ws->AFB, ldafb, ws->IWORK, ws->X, ldb, &info);

        if (info != izero) {
            fail_msg("DGBSV info=%d expected=%d (n=%d kl=%d ku=%d imat=%d)",
                     info, izero, n, kl, ku, imat);
        }

        /* TEST 1: Reconstruct matrix from factors */
        dgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);

        int nt = 1;
        if (izero == 0) {
            /* TEST 2: Compute residual of computed solution */
            dlacpy("Full", n, NRHS, ws->B, ldb, ws->WORK, ldb);
            dgbt02("N", n, n, kl, ku, NRHS, ws->A, lda,
                   ws->X, ldb, ws->WORK, ldb, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            dget04(n, NRHS, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DGBSV n=%d kl=%d ku=%d type %d test %d: result=%e >= thresh=%e",
                         n, kl, ku, imat, k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DGBSVX --- */

    /* Zero AFB if not prefactored */
    if (!prefac)
        dlaset("Full", 2 * kl + ku + 1, n, 0.0, 0.0, ws->AFB, ldafb);
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, ldb);

    /* Equilibrate matrix if IEQUED > 0 */
    if (iequed > 0 && n > 0) {
        dlaqgb(n, n, kl, ku, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed);
    }

    /* Call DGBSVX */
    char equed_inout = equed;
    f64 rcond;
    dgbsvx(fact, trans, n, kl, ku, NRHS, ws->A, lda, ws->AFB, ldafb,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, ldb, ws->X, ldb, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->IWORK[n], &info);

    /* Check error code */
    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("DGBSVX info=%d expected=%d (n=%d kl=%d ku=%d imat=%d fact=%s trans=%s)",
                     info, izero, n, kl, ku, imat, fact, trans);
        }
    }

    /*
     * TEST 7: Compare RPVGRW from DGBSVX (work[0]) with computed value.
     * Save work[0] before it could be overwritten by dlantb/dlangb.
     * Note: 'M' norm does not use the work array, so work[0] is preserved,
     * but we save it defensively.
     */
    f64 rpvgrw_svx = ws->WORK[0];
    f64 rpvgrw;

    if (info != 0 && info <= n) {
        /* Singularity detected at column INFO (1-based) */
        f64 anrmpv = 0.0;
        for (int j = 0; j < info; j++) {
            /* Fortran: DO I = MAX(KU+2-J, 1), MIN(N+KU+1-J, KL+KU+1) */
            int i1_f = ku + 2 - (j + 1);
            int i2_f = n + ku + 1 - (j + 1);
            int i1 = (i1_f > 1) ? i1_f - 1 : 0;
            int i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (int i = i1; i < i2; i++) {
                f64 val = fabs(ws->A[i + j * lda]);
                if (val > anrmpv) anrmpv = val;
            }
        }
        /* Fortran: DLANTB('M','U','N',INFO,MIN(INFO-1,KL+KU),
         *                  AFB(MAX(1,KL+KU+2-INFO)),LDAFB,WORK) */
        int kband = (info - 1 < kl + ku) ? info - 1 : kl + ku;
        int afb_off_f = kl + ku + 2 - info;
        int afb_off = (afb_off_f > 1) ? afb_off_f - 1 : 0;
        rpvgrw = dlantb("M", "U", "N", info, kband,
                        &ws->AFB[afb_off], ldafb, ws->WORK);
        if (rpvgrw == 0.0)
            rpvgrw = 1.0;
        else
            rpvgrw = anrmpv / rpvgrw;
    } else {
        rpvgrw = dlantb("M", "U", "N", n, kl + ku,
                        ws->AFB, ldafb, ws->WORK);
        if (rpvgrw == 0.0) {
            rpvgrw = 1.0;
        } else {
            rpvgrw = dlangb("M", n, kl, ku, ws->A, lda, ws->WORK) / rpvgrw;
        }
    }
    f64 denom = fmax(rpvgrw_svx, rpvgrw);
    if (denom > 0.0)
        result[6] = fabs(rpvgrw - rpvgrw_svx) / denom / dlamch("E");
    else
        result[6] = 0.0;

    /* TEST 1: Reconstruct matrix from factors */
    int k1;
    if (!prefac) {
        dgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    int trfcon;
    if (info == 0) {
        trfcon = 0;

        /* TEST 2: Compute residual of computed solution */
        dlacpy("Full", n, NRHS, ws->BSAV, ldb, ws->WORK, ldb);
        dgbt02(trans, n, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->X, ldb, ws->WORK, ldb, &ws->RWORK[2 * NRHS],
               &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        f64 rcond_for_get04;
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            f64 roldc = (itran == 0) ? roldo : roldi;
            rcond_for_get04 = roldc;
        }
        dget04(n, NRHS, ws->X, ldb, ws->XACT, ldb,
               rcond_for_get04, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        dgbt05(trans, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->B, ldb, ws->X, ldb, ws->XACT, ldb,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    /* TEST 6: Compare RCOND from DGBSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    if (!trfcon) {
        for (int k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                if (prefac) {
                    fail_msg("DGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d EQUED=%c "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku, equed_inout,
                             imat, k + 1, result[k], THRESH);
                } else {
                    fail_msg("DGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku,
                             imat, k + 1, result[k], THRESH);
                }
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("DGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 1: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("DGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 6: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[5], THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("DGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 7: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[6], THRESH);
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
static void get_klku_values(int n, int vals[NBW])
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
static int build_test_array(ddrvgb_params_t* params, struct CMUnitTest* tests)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    int idx = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nkl = (n > 4) ? 4 : ((n > 0) ? n : 1);
        int nku = nkl;
        int nimat = (n <= 0) ? 1 : NTYPES;

        int klval[NBW], kuval[NBW];
        get_klku_values(n, klval);
        get_klku_values(n, kuval);

        for (int ikl = 0; ikl < nkl; ikl++) {
            int kl = klval[ikl];
            for (int iku = 0; iku < nku; iku++) {
                int ku = kuval[iku];
                int lda   = kl + ku + 1;
                int ldafb = 2 * kl + ku + 1;

                /* Check that workspace is big enough */
                if (lda * n > LA || ldafb * n > LAFB)
                    continue;

                for (int imat = 1; imat <= nimat; imat++) {
                    int zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1)
                        continue;

                    for (int iequed = 0; iequed < 4; iequed++) {
                        int nfact = (iequed == 0) ? 3 : 1;

                        for (int ifact = 0; ifact < nfact; ifact++) {
                            /* Skip FACT='F' for singular matrices */
                            if (zerot && ifact == 0)
                                continue;

                            for (int itran = 0; itran < NTRAN; itran++) {
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
    int count = build_test_array(NULL, NULL);
    if (count == 0) return 0;

    ddrvgb_params_t* params = malloc(count * sizeof(*params));
    struct CMUnitTest* tests = malloc(count * sizeof(*tests));
    if (!params || !tests) {
        free(params);
        free(tests);
        return 1;
    }

    build_test_array(params, tests);
    int result = _cmocka_run_group_tests("ddrvgb", tests, count,
                                          group_setup, group_teardown);
    free(tests);
    free(params);
    return result;
}
