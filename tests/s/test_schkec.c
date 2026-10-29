/**
 * @file test_schkec.c
 * @brief Port of LAPACK TESTING/EIG/dchkec.f — eigencondition estimation tests.
 *
 * Tests eigen-condition estimation routines:
 *   SLALN2, SLASY2, SLANV2, SLAQTR, SLAEXC,
 *   STRSYL, STRSYL3, STREXC, STRSNA, STRSEN, STGEXC
 *
 * Each sub-routine runs through a fixed set of numerical examples,
 * subjects them to various tests, and compares the test results to
 * a threshold THRESH. STREXC, STRSNA, STRSEN, and STGEXC also use
 * precomputed examples (embedded as static data in the C port).
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 20.0f

/* ------------------------------------------------------------------ */
/*  Error exit tests                                                  */
/* ------------------------------------------------------------------ */

static void test_serrec(void** state)
{
    (void)state;
    INT ok, nt;
    serrec("SEC", &ok, &nt);
    assert_true(ok);
}

/* ------------------------------------------------------------------ */
/*  SLALN2 — small quasi-triangular system solver                     */
/* ------------------------------------------------------------------ */

static void test_sget31_slaln2(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo[2], knt;

    sget31(&rmax, &lmax, ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo[0], 0);
}

/* ------------------------------------------------------------------ */
/*  SLASY2 — Sylvester-like equation solver                           */
/* ------------------------------------------------------------------ */

static void test_sget32_slasy2(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    sget32(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  SLANV2 — 2x2 real Schur form standardization                     */
/* ------------------------------------------------------------------ */

static void test_sget33_slanv2(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    sget33(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo, 0);
}

/* ------------------------------------------------------------------ */
/*  SLAEXC — swap adjacent diagonal blocks in Schur form              */
/* ------------------------------------------------------------------ */

static void test_sget34_slaexc(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo[2], knt;

    sget34(&rmax, &lmax, ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_int_equal(ninfo[1], 0);
}

/* ------------------------------------------------------------------ */
/*  STRSYL — Sylvester equation solver                                */
/* ------------------------------------------------------------------ */

static void test_sget35_strsyl(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    sget35(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  STRSYL3 — Sylvester equation solver (level 3)                     */
/* ------------------------------------------------------------------ */

static void test_ssyl01_strsyl3(void** state)
{
    (void)state;
    INT ftrsyl[3];
    f32 rtrsyl[2];
    INT itrsyl[2];
    INT ktrsyl3;

    ssyl01(THRESH, ftrsyl, rtrsyl, itrsyl, &ktrsyl3);

    assert_true(ktrsyl3 >= 0);
    assert_int_equal(ftrsyl[0], 0);
    assert_int_equal(ftrsyl[1], 0);
    assert_int_equal(ftrsyl[2], 0);
}

/* ------------------------------------------------------------------ */
/*  STREXC — reorder Schur form                                       */
/* ------------------------------------------------------------------ */

static void test_sget36_strexc(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo[3], knt;

    sget36(&rmax, &lmax, ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_true(ninfo[2] <= 0);
}

/* ------------------------------------------------------------------ */
/*  STRSNA — condition numbers for eigenvalues/eigenvectors            */
/* ------------------------------------------------------------------ */

static void test_sget37_strsna(void** state)
{
    (void)state;
    f32 rmax[3];
    INT lmax[3], ninfo[3], knt;

    sget37(rmax, lmax, ninfo, &knt);

    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  STRSEN — reorder and condition estimate for Schur clusters         */
/* ------------------------------------------------------------------ */

static void test_sget38_strsen(void** state)
{
    (void)state;
    f32 rmax[3];
    INT lmax[3], ninfo[3], knt;

    sget38(rmax, lmax, ninfo, &knt);

    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  SLAQTR — quasi-triangular solve for condition estimation           */
/* ------------------------------------------------------------------ */

static void test_sget39_slaqtr(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    sget39(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  STGEXC — generalized Schur block swap                             */
/* ------------------------------------------------------------------ */

static void test_sget40_stgexc(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo[2], knt;

    sget40(&rmax, &lmax, ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  Main — CMocka test registration                                   */
/* ------------------------------------------------------------------ */

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_serrec),
        cmocka_unit_test(test_sget31_slaln2),
        cmocka_unit_test(test_sget32_slasy2),
        cmocka_unit_test(test_sget33_slanv2),
        cmocka_unit_test(test_sget34_slaexc),
        cmocka_unit_test(test_sget35_strsyl),
        cmocka_unit_test(test_ssyl01_strsyl3),
        cmocka_unit_test(test_sget36_strexc),
        cmocka_unit_test(test_sget37_strsna),
        cmocka_unit_test(test_sget38_strsen),
        cmocka_unit_test(test_sget39_slaqtr),
        cmocka_unit_test(test_sget40_stgexc),
    };
    /* Threshold from sec.in; ISPECs 1 and 12-16 as SCHKEE sets them for SEC */
    xlaenv(1, 1);
    xlaenv(12, 11);
    xlaenv(13, 2);
    xlaenv(14, 0);
    xlaenv(15, 2);
    xlaenv(16, 2);
    (void)cmocka_run_group_tests_name("schkec", tests, NULL, NULL);
    return 0;
}
