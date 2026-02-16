/**
 * @file test_dchkec.c
 * @brief Port of LAPACK TESTING/EIG/dchkec.f — eigencondition estimation tests.
 *
 * Tests eigen-condition estimation routines:
 *   DLALN2, DLASY2, DLANV2, DLAQTR, DLAEXC,
 *   DTRSYL, DTRSYL3, DTREXC, DTRSNA, DTRSEN, DTGEXC
 *
 * Each sub-routine runs through a fixed set of numerical examples,
 * subjects them to various tests, and compares the test results to
 * a threshold THRESH. DTREXC, DTRSNA, DTRSEN, and DTGEXC also use
 * precomputed examples (embedded as static data in the C port).
 */

#include "test_harness.h"
#include "verify.h"

extern f64 dlamch(const char* cmach);

#define THRESH 30.0

/* ------------------------------------------------------------------ */
/*  Error exit tests                                                  */
/* ------------------------------------------------------------------ */

static void test_derrec(void** state)
{
    (void)state;
    int ok, nt;
    derrec(&ok, &nt);
    if (!ok) {
        fprintf(stderr, " *** DEC routines failed the tests of the error exits ***\n");
    } else {
        fprintf(stderr, " DEC routines passed the tests of the error exits (%d tests done)\n", nt);
    }
    assert_true(ok);
}

/* ------------------------------------------------------------------ */
/*  DLALN2 — small quasi-triangular system solver                     */
/* ------------------------------------------------------------------ */

static void test_dget31_dlaln2(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo[2], knt;

    dget31(&rmax, &lmax, ninfo, &knt);

    if (rmax > THRESH || ninfo[0] != 0) {
        fprintf(stderr, " Error in DLALN2: RMAX = %.3e\n LMAX = %d NINFO= %d %d KNT= %d\n",
                rmax, lmax, ninfo[0], ninfo[1], knt);
    }
    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo[0], 0);
}

/* ------------------------------------------------------------------ */
/*  DLASY2 — Sylvester-like equation solver                           */
/* ------------------------------------------------------------------ */

static void test_dget32_dlasy2(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo, knt;

    dget32(&rmax, &lmax, &ninfo, &knt);

    if (rmax > THRESH) {
        fprintf(stderr, " Error in DLASY2: RMAX = %.3e\n LMAX = %d NINFO= %d KNT= %d\n",
                rmax, lmax, ninfo, knt);
    }
    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  DLANV2 — 2x2 real Schur form standardization                     */
/* ------------------------------------------------------------------ */

static void test_dget33_dlanv2(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo, knt;

    dget33(&rmax, &lmax, &ninfo, &knt);

    if (rmax > THRESH || ninfo != 0) {
        fprintf(stderr, " Error in DLANV2: RMAX = %.3e\n LMAX = %d NINFO= %d KNT= %d\n",
                rmax, lmax, ninfo, knt);
    }
    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo, 0);
}

/* ------------------------------------------------------------------ */
/*  DLAEXC — swap adjacent diagonal blocks in Schur form              */
/* ------------------------------------------------------------------ */

static void test_dget34_dlaexc(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo[2], knt;

    dget34(&rmax, &lmax, ninfo, &knt);

    if (rmax > THRESH || ninfo[1] != 0) {
        fprintf(stderr, " Error in DLAEXC: RMAX = %.3e\n LMAX = %d NINFO= %d %d KNT= %d\n",
                rmax, lmax, ninfo[0], ninfo[1], knt);
    }
    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo[1], 0);
}

/* ------------------------------------------------------------------ */
/*  DTRSYL — Sylvester equation solver                                */
/* ------------------------------------------------------------------ */

static void test_dget35_dtrsyl(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo, knt;

    dget35(&rmax, &lmax, &ninfo, &knt);

    if (rmax > THRESH) {
        fprintf(stderr, " Error in DTRSYL: RMAX = %.3e\n LMAX = %d NINFO= %d KNT= %d\n",
                rmax, lmax, ninfo, knt);
    }
    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  DTRSYL3 — Sylvester equation solver (level 3)                     */
/* ------------------------------------------------------------------ */

static void test_dsyl01_dtrsyl3(void** state)
{
    (void)state;
    int ftrsyl[3];
    f64 rtrsyl[2];
    int itrsyl[2];
    int ktrsyl3;

    dsyl01(THRESH, ftrsyl, rtrsyl, itrsyl, &ktrsyl3);

    if (ftrsyl[0] > 0) {
        fprintf(stderr, "Error in DTRSYL: %d tests fail the threshold.\n"
                "Maximum test ratio = %.3e threshold = %.3e\n",
                ftrsyl[0], rtrsyl[0], THRESH);
    }
    if (ftrsyl[1] > 0) {
        fprintf(stderr, "Error in DTRSYL3: %d tests fail the threshold.\n"
                "Maximum test ratio = %.3e threshold = %.3e\n",
                ftrsyl[1], rtrsyl[1], THRESH);
    }
    if (ftrsyl[2] > 0) {
        fprintf(stderr, "DTRSYL and DTRSYL3 compute an inconsistent result "
                "factor in %d tests.\n", ftrsyl[2]);
    }
    assert_int_equal(ftrsyl[0], 0);
    assert_int_equal(ftrsyl[1], 0);
    assert_int_equal(ftrsyl[2], 0);
}

/* ------------------------------------------------------------------ */
/*  DTREXC — reorder Schur form                                       */
/* ------------------------------------------------------------------ */

static void test_dget36_dtrexc(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo[3], knt;

    dget36(&rmax, &lmax, ninfo, &knt);

    if (rmax > THRESH || ninfo[2] > 0) {
        fprintf(stderr, " Error in DTREXC: RMAX = %.3e\n LMAX = %d NINFO= %d %d %d KNT= %d\n",
                rmax, lmax, ninfo[0], ninfo[1], ninfo[2], knt);
    }
    assert_true(rmax <= THRESH);
    assert_true(ninfo[2] <= 0);
}

/* ------------------------------------------------------------------ */
/*  DTRSNA — condition numbers for eigenvalues/eigenvectors            */
/* ------------------------------------------------------------------ */

static void test_dget37_dtrsna(void** state)
{
    (void)state;
    f64 rmax[3];
    int lmax[3], ninfo[3], knt;

    dget37(rmax, lmax, ninfo, &knt);

    if (rmax[0] > THRESH || rmax[1] > THRESH ||
        ninfo[0] != 0 || ninfo[1] != 0 || ninfo[2] != 0) {
        fprintf(stderr, " Error in DTRSNA: RMAX = %.3e %.3e %.3e\n"
                " LMAX = %d %d %d NINFO= %d %d %d KNT= %d\n",
                rmax[0], rmax[1], rmax[2],
                lmax[0], lmax[1], lmax[2],
                ninfo[0], ninfo[1], ninfo[2], knt);
    }
    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  DTRSEN — reorder and condition estimate for Schur clusters         */
/* ------------------------------------------------------------------ */

static void test_dget38_dtrsen(void** state)
{
    (void)state;
    f64 rmax[3];
    int lmax[3], ninfo[3], knt;

    dget38(rmax, lmax, ninfo, &knt);

    if (rmax[0] > THRESH || rmax[1] > THRESH ||
        ninfo[0] != 0 || ninfo[1] != 0 || ninfo[2] != 0) {
        fprintf(stderr, " Error in DTRSEN: RMAX = %.3e %.3e %.3e\n"
                " LMAX = %d %d %d NINFO= %d %d %d KNT= %d\n",
                rmax[0], rmax[1], rmax[2],
                lmax[0], lmax[1], lmax[2],
                ninfo[0], ninfo[1], ninfo[2], knt);
    }
    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  DLAQTR — quasi-triangular solve for condition estimation           */
/* ------------------------------------------------------------------ */

static void test_dget39_dlaqtr(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo, knt;

    dget39(&rmax, &lmax, &ninfo, &knt);

    if (rmax > THRESH) {
        fprintf(stderr, " Error in DLAQTR: RMAX = %.3e\n LMAX = %d NINFO= %d KNT= %d\n",
                rmax, lmax, ninfo, knt);
    }
    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  DTGEXC — generalized Schur block swap                             */
/* ------------------------------------------------------------------ */

static void test_dget40_dtgexc(void** state)
{
    (void)state;
    f64 rmax;
    int lmax, ninfo, knt;

    dget40(&rmax, &lmax, &ninfo, &knt);

    if (rmax > THRESH) {
        fprintf(stderr, " Error in DTGEXC: RMAX = %.3e\n LMAX = %d NINFO= %d KNT= %d\n",
                rmax, lmax, ninfo, knt);
    }
    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  Main — CMocka test registration                                   */
/* ------------------------------------------------------------------ */

int main(void)
{
    f64 eps = dlamch("P");
    f64 sfmin = dlamch("S");
    fprintf(stderr, " Tests of the Nonsymmetric eigenproblem condition estimation routines\n"
            " DLALN2, DLASY2, DLANV2, DLAEXC, DTRSYL, DTREXC, DTRSNA, DTRSEN, DLAQTR, DTGEXC\n\n"
            " Relative machine precision (EPS) = %16.6e\n"
            " Safe minimum (SFMIN)             = %16.6e\n\n"
            " Routines pass computational tests if test ratio is less than%8.2f\n\n",
            eps, sfmin, THRESH);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_derrec),
        cmocka_unit_test(test_dget31_dlaln2),
        cmocka_unit_test(test_dget32_dlasy2),
        cmocka_unit_test(test_dget33_dlanv2),
        cmocka_unit_test(test_dget34_dlaexc),
        cmocka_unit_test(test_dget35_dtrsyl),
        cmocka_unit_test(test_dsyl01_dtrsyl3),
        cmocka_unit_test(test_dget36_dtrexc),
        cmocka_unit_test(test_dget37_dtrsna),
        cmocka_unit_test(test_dget38_dtrsen),
        cmocka_unit_test(test_dget39_dlaqtr),
        cmocka_unit_test(test_dget40_dtgexc),
    };
    return cmocka_run_group_tests_name("dchkec", tests, NULL, NULL);
}
