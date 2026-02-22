/**
 * @file lapack_tuning.c
 * @brief Production implementation of LAPACK tuning parameter lookup.
 *
 * This file provides the production version of lapack_get_nb(),
 * lapack_get_nbmin(), and lapack_get_nx(). These functions use
 * a simple lookup table to return optimal block sizes.
 *
 * For testing with different block sizes, the test suite uses
 * tests/verify/lapack_tuning_test.c instead, which provides
 * the xlaenv() function to override parameters at runtime.
 */

#define LAPACK_TUNING_IMPLEMENTATION
#include "lapack_tuning.h"
