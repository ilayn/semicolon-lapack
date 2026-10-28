/**
 * @file lapack_tuning_override.c
 * @brief The tuning override hook that the test suite sets.
 *
 * Kept out of lapack_tuning.c so that linking the hook does not also pull in
 * the production tuning functions, which a test build replaces with its own.
 */

#include <stddef.h>
#include "semicolon_lapack_auxiliary.h"

const lapack_tuning_override_t* lapack_tuning_override = NULL;
