/**
 * @file test_zchkec.c
 * @brief Port of LAPACK TESTING/EIG/zchkec.f — eigencondition estimation tests.
 *
 * Tests eigen-condition estimation routines:
 *   ZTRSYL, ZTRSYL3, ZTREXC, ZTRSNA, ZTRSEN
 *
 * Each sub-routine runs through a fixed set of numerical examples,
 * subjects them to various tests, and compares the test results to
 * a threshold THRESH. ZTRSNA and ZTRSEN also use precomputed examples
 * (embedded as static data in the C port).
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0

/* ------------------------------------------------------------------ */
/*  ZTRSYL — Sylvester equation solver                                */
/* ------------------------------------------------------------------ */

static void test_zget35_ztrsyl(void** state)
{
    (void)state;
    f64 rmax;
    INT lmax, ninfo, knt;

    zget35(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  ZTRSYL3 — Sylvester equation solver (level 3)                     */
/* ------------------------------------------------------------------ */

static void test_zsyl01_ztrsyl3(void** state)
{
    (void)state;
    INT ftrsyl[3];
    f64 rtrsyl[2];
    INT itrsyl[2];
    INT ktrsyl3;

    zsyl01(THRESH, ftrsyl, rtrsyl, itrsyl, &ktrsyl3);

    assert_int_equal(ftrsyl[0], 0);
    assert_int_equal(ftrsyl[1], 0);
    assert_int_equal(ftrsyl[2], 0);
}

/* ------------------------------------------------------------------ */
/*  ZTREXC — reorder Schur form                                       */
/* ------------------------------------------------------------------ */

static void test_zget36_ztrexc(void** state)
{
    (void)state;
    f64 rmax;
    INT lmax, ninfo, knt;

    zget36(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_true(ninfo <= 0);
}

/* ------------------------------------------------------------------ */
/*  ZTRSNA — condition numbers for eigenvalues/eigenvectors            */
/* ------------------------------------------------------------------ */

static void test_zget37_ztrsna(void** state)
{
    (void)state;
    f64 rmax[3];
    INT lmax[3], ninfo[3], knt;

    zget37(rmax, lmax, ninfo, &knt);

    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  ZTRSEN — reorder and condition estimate for Schur clusters         */
/* ------------------------------------------------------------------ */

static void test_zget38_ztrsen(void** state)
{
    (void)state;
    f64 rmax[3];
    INT lmax[3], ninfo[3], knt;

    zget38(rmax, lmax, ninfo, &knt);

    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  Main — CMocka test registration                                   */
/* ------------------------------------------------------------------ */

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zget35_ztrsyl),
        cmocka_unit_test(test_zsyl01_ztrsyl3),
        cmocka_unit_test(test_zget36_ztrexc),
        cmocka_unit_test(test_zget37_ztrsna),
        cmocka_unit_test(test_zget38_ztrsen),
    };
    return cmocka_run_group_tests_name("zchkec", tests, NULL, NULL);
}
