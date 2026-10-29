/**
 * @file test_cchkec.c
 * @brief Port of LAPACK TESTING/EIG/zchkec.f — eigencondition estimation tests.
 *
 * Tests eigen-condition estimation routines:
 *   CTRSYL, CTRSYL3, CTREXC, CTRSNA, CTRSEN
 *
 * Each sub-routine runs through a fixed set of numerical examples,
 * subjects them to various tests, and compares the test results to
 * a threshold THRESH. CTRSNA and CTRSEN also use precomputed examples
 * (embedded as static data in the C port).
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 20.0f

/* ------------------------------------------------------------------ */
/*  Error exit tests                                                  */
/* ------------------------------------------------------------------ */

static void test_cerrec(void** state)
{
    (void)state;
    INT ok, nt;
    cerrec("CEC", &ok, &nt);
    assert_true(ok);
}

/* ------------------------------------------------------------------ */
/*  CTRSYL — Sylvester equation solver                                */
/* ------------------------------------------------------------------ */

static void test_cget35_ctrsyl(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    cget35(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
}

/* ------------------------------------------------------------------ */
/*  CTRSYL3 — Sylvester equation solver (level 3)                     */
/* ------------------------------------------------------------------ */

static void test_csyl01_ctrsyl3(void** state)
{
    (void)state;
    INT ftrsyl[3];
    f32 rtrsyl[2];
    INT itrsyl[2];
    INT ktrsyl3;

    csyl01(THRESH, ftrsyl, rtrsyl, itrsyl, &ktrsyl3);

    assert_true(ktrsyl3 >= 0);
    assert_int_equal(ftrsyl[0], 0);
    assert_int_equal(ftrsyl[1], 0);
    assert_int_equal(ftrsyl[2], 0);
}

/* ------------------------------------------------------------------ */
/*  CTREXC — reorder Schur form                                       */
/* ------------------------------------------------------------------ */

static void test_cget36_ctrexc(void** state)
{
    (void)state;
    f32 rmax;
    INT lmax, ninfo, knt;

    cget36(&rmax, &lmax, &ninfo, &knt);

    assert_true(rmax <= THRESH);
    assert_true(ninfo <= 0);
}

/* ------------------------------------------------------------------ */
/*  CTRSNA — condition numbers for eigenvalues/eigenvectors            */
/* ------------------------------------------------------------------ */

static void test_cget37_ctrsna(void** state)
{
    (void)state;
    f32 rmax[3];
    INT lmax[3], ninfo[3], knt;

    cget37(rmax, lmax, ninfo, &knt);

    assert_true(rmax[0] <= THRESH);
    assert_true(rmax[1] <= THRESH);
    assert_int_equal(ninfo[0], 0);
    assert_int_equal(ninfo[1], 0);
    assert_int_equal(ninfo[2], 0);
}

/* ------------------------------------------------------------------ */
/*  CTRSEN — reorder and condition estimate for Schur clusters         */
/* ------------------------------------------------------------------ */

static void test_cget38_ctrsen(void** state)
{
    (void)state;
    f32 rmax[3];
    INT lmax[3], ninfo[3], knt;

    cget38(rmax, lmax, ninfo, &knt);

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
        cmocka_unit_test(test_cerrec),
        cmocka_unit_test(test_cget35_ctrsyl),
        cmocka_unit_test(test_csyl01_ctrsyl3),
        cmocka_unit_test(test_cget36_ctrexc),
        cmocka_unit_test(test_cget37_ctrsna),
        cmocka_unit_test(test_cget38_ctrsen),
    };
    /* Threshold from cec.in; ISPECs 1 and 12 as CCHKEE sets them for CEC */
    xlaenv(1, 1);
    xlaenv(12, 1);
    (void)cmocka_run_group_tests_name("cchkec", tests, NULL, NULL);
    return 0;
}
