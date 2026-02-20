/**
 * @file lapack_tuning_test.c
 * @brief Test implementation of LAPACK tuning parameter lookup.
 *
 * This file provides the test version of lapack_get_nb(),
 * lapack_get_nbmin(), and lapack_get_nx(). Unlike the production
 * version, this one supports runtime override via xlaenv(),
 * mirroring LAPACK's TESTING/LIN/ilaenv.f mechanism.
 *
 * Usage in tests:
 *     xlaenv(1, 3);           // Set NB=3 for all routines
 *     dgetrf(...);            // Will use NB=3
 *     xlaenv_reset();         // Restore table defaults
 *
 * This allows testing blocked algorithms with various block sizes
 * (e.g., NB=1 for unblocked path, NB=3 for small blocks, NB=20
 * for medium blocks) without recompiling.
 */

#define LAPACK_TUNING_IMPLEMENTATION
#define LAPACK_TUNING_TEST
#include "lapack_tuning.h"
