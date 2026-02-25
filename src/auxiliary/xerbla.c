// xerbla.c
// Error handler for LAPACK routines

#include <stdio.h>
#include "semicolon_lapack_auxiliary.h"

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define XERBLA_WEAK __attribute__((weak))
#else
#define XERBLA_WEAK
#endif

XERBLA_WEAK void xerbla(const char *srname, INT info) {
    fprintf(stderr, " ** On entry to %s parameter number %d had an illegal value\n",
            srname, info);
}
