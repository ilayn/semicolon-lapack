/**
 * @file test_harness.h
 * @brief CMocka integration macros for LAPACK-style residual testing.
 *
 * This header bridges CMocka's assertion framework with LAPACK's
 * normalized residual methodology.
 *
 * Each test file must define its own THRESH constant before using
 * assert_residual_ok. The threshold should match what LAPACK uses
 * for that specific test suite (see dtest.in, dchkge.f, etc.).
 *
 * Usage:
 * @code
 *   #include "test_harness.h"
 *   #define THRESH 30.0
 *
 *   static void test_dgetrf_square(void **state) {
 *       double resid;
 *       dget01(m, n, A, lda, AFAC, lda, ipiv, rwork, &resid);
 *       assert_residual_ok(resid);
 *   }
 * @endcode
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

/* CMocka requires these headers in this exact order */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>

/* Standard includes for test code */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "semicolon_lapack/types.h"

/*
 * Test context tracking for better error messages.
 * Before running each parameterized test, set this to identify the test case.
 * On failure, the context will be printed along with the assertion details.
 */
static const char* g_test_context = NULL;

/**
 * Set the current test context for better failure messages.
 * Call this at the start of each parameterized test with a description.
 *
 * Example:
 *   set_test_context("dchkpo n=50 uplo=U imat=3 TEST 2 (inverse)");
 */
static inline void set_test_context(const char* context) {
    g_test_context = context;
}

/**
 * Clear the test context (call at end of test or in teardown).
 */
static inline void clear_test_context(void) {
    g_test_context = NULL;
}

/**
 * Assert that a normalized residual is below the test's THRESH.
 *
 * @param resid  Normalized residual from verification routine (dget01, etc.)
 *
 * Requires THRESH to be defined in the test file.
 */
#define assert_residual_ok(resid) \
    _assert_residual_below(resid, THRESH, #resid, __FILE__, __LINE__)

/**
 * Assert that a normalized residual is below a custom threshold.
 *
 * @param resid   Normalized residual
 * @param thresh  Custom threshold
 */
#define assert_residual_below(resid, thresh) \
    _assert_residual_below(resid, thresh, #resid, __FILE__, __LINE__)

/**
 * Internal implementation of residual assertion.
 * Do not call directly; use assert_residual_ok or assert_residual_below.
 */
static inline void _assert_residual_below(
    double resid,
    double thresh,
    const char *resid_expr,
    const char *file,
    int line)
{
    if (resid >= thresh) {
        if (g_test_context) {
            print_error("\n*** FAILED: %s\n", g_test_context);
        }
        print_error("    %s = %.6e >= threshold %.1f\n", resid_expr, resid, thresh);
        _fail(file, line);
    }
}

/**
 * Assert that INFO return value indicates success.
 *
 * @param info  INFO parameter from LAPACK routine
 *
 * INFO meaning:
 *   = 0  : Success
 *   < 0  : Illegal argument (-info is the argument number)
 *   > 0  : Algorithm-specific failure (e.g., singular matrix)
 */
#define assert_info_success(info) \
    _assert_info_success(info, #info, __FILE__, __LINE__)

static inline void _assert_info_success(
    int info,
    const char *info_expr,
    const char *file,
    int line)
{
    if (info != 0) {
        if (g_test_context) {
            print_error("\n*** FAILED: %s\n", g_test_context);
        }
        if (info < 0) {
            print_error("    %s = %d: illegal argument %d\n", info_expr, info, -info);
        } else {
            print_error("    %s = %d: algorithm failure\n", info_expr, info);
        }
        _fail(file, line);
    }
}

/**
 * Assert that INFO indicates a singular matrix (expected for certain tests).
 *
 * @param info  INFO parameter from LAPACK routine
 *
 * For singular matrix test types (5-7 in dlatb4), INFO > 0 is expected.
 */
#define assert_info_singular(info) \
    _assert_info_singular(info, #info, __FILE__, __LINE__)

static inline void _assert_info_singular(
    int info,
    const char *info_expr,
    const char *file,
    int line)
{
    if (info <= 0) {
        if (g_test_context) {
            print_error("\n*** FAILED: %s\n", g_test_context);
        }
        print_error("    %s = %d: expected singular (info > 0)\n", info_expr, info);
        _fail(file, line);
    }
}

/**
 * Skip test with a message.
 * Use for tests that don't apply to certain configurations.
 */
#define skip_test(msg) \
    do { \
        print_message("SKIP: %s\n", msg); \
        skip(); \
    } while (0)

/*
 * XLAENV - Set LAPACK tuning parameters for testing.
 *
 * This mirrors LAPACK's TESTING/LIN/xlaenv.f, allowing tests to override
 * block sizes and other tuning parameters.
 *
 * @param ispec  Parameter to set:
 *               1 = NB (block size)
 *               2 = NBMIN (minimum block size)
 *               3 = NX (crossover point)
 * @param nvalue Value to set
 *
 * Usage:
 *     xlaenv(1, 3);    // Set NB=3 for all routines
 *     dgetrf(...);     // Will use NB=3
 *     xlaenv_reset();  // Restore defaults
 */
extern void xlaenv(int ispec, int nvalue);
extern void xlaenv_reset(void);

/*
 * =============================================================================
 * CMocka Configuration Helpers
 * =============================================================================
 *
 * CMocka 2.0 supports several useful features via environment variables:
 *
 * TEST FILTERING (useful for debugging specific failures):
 *   CMOCKA_TEST_FILTER="pattern"   - Only run tests matching pattern
 *   CMOCKA_SKIP_FILTER="pattern"   - Skip tests matching pattern
 *
 *   Pattern wildcards:
 *     *  - matches zero or more characters
 *     ?  - matches exactly one character
 *
 *   Examples:
 *     CMOCKA_TEST_FILTER="*n50*"           - Run only n=50 tests
 *     CMOCKA_TEST_FILTER="*_U_*"           - Run only upper triangular tests
 *     CMOCKA_TEST_FILTER="dchksy_n5_U_type1*"  - Run specific test group
 *     CMOCKA_SKIP_FILTER="*type10*"        - Skip type 10 (scaled) tests
 *
 * OUTPUT FORMATS (useful for CI integration):
 *   CMOCKA_MESSAGE_OUTPUT=STANDARD  - Default human-readable output
 *   CMOCKA_MESSAGE_OUTPUT=TAP       - Test Anything Protocol (TAP v14)
 *   CMOCKA_MESSAGE_OUTPUT=XML       - JUnit-compatible XML
 *   CMOCKA_MESSAGE_OUTPUT=SUBUNIT   - Subunit format
 *
 *   Multiple formats can be combined:
 *     CMOCKA_MESSAGE_OUTPUT=STANDARD,XML
 *
 * PROGRAMMATIC ACCESS:
 *   These can also be set in code before running tests:
 *     cmocka_set_test_filter("*n50*");
 *     cmocka_set_skip_filter("*type10*");
 *     cmocka_set_message_output(CM_OUTPUT_TAP);
 */

/**
 * Initialize CMocka output based on environment or defaults.
 * Call this at the start of main() if you want programmatic control.
 *
 * Example:
 *   int main(void) {
 *       // Enable XML output for CI if CMOCKA_XML is set
 *       if (getenv("CMOCKA_XML")) {
 *           cmocka_set_message_output(CM_OUTPUT_XML);
 *       }
 *       // ... build and run tests ...
 *   }
 */

#endif /* TEST_HARNESS_H */
