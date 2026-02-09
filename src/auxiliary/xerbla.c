// xerbla.c
// Error handler for LAPACK routines

#include <stdio.h>
#include "semicolon_lapack_auxiliary.h"

void xerbla(const char *srname, int info) {
    fprintf(stderr, " ** On entry to %s parameter number %d had an illegal value\n",
            srname, info);
}
