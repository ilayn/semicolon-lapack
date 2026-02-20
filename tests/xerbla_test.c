/**
 * @file xerbla_test.c
 * @brief Test version of xerbla with flag-setting for error exit tests.
 *
 * This provides a strong xerbla symbol that overrides the weak production
 * version in src/auxiliary/xerbla.c. Instead of printing to stderr, it
 * sets global flags that derrec/chkxer can inspect.
 *
 * Replaces the Fortran COMMON /INFOC/ and /SRNAMC/ mechanism.
 *
 * When xerbla_srnamt is empty (not in error-testing mode), xerbla
 * behaves like the production version (prints to stderr only).
 * When xerbla_srnamt is set, it validates the caller and parameter number.
 */

#include <stdio.h>
#include <string.h>
#include "semicolon_lapack_auxiliary.h"

/* Global state (replaces Fortran COMMON blocks) */
int    xerbla_infot  = 0;
int    xerbla_nout   = 0;
int    xerbla_ok     = 1;
int    xerbla_lerr   = 0;
char   xerbla_srnamt[33] = "";

void xerbla(const char* srname, int info) {
    /* When not in error-testing mode, behave like production xerbla */
    if (xerbla_srnamt[0] == '\0') {
        fprintf(stderr, " ** On entry to %s parameter number %d had an illegal value\n",
                srname, info);
        return;
    }

    xerbla_lerr = 1;
    if (info != xerbla_infot) {
        fprintf(stderr, " *** XERBLA was called with INFO = %d instead of %d ***\n",
                info, xerbla_infot);
        fprintf(stderr, " *** routine = %s ***\n", srname);
        xerbla_ok = 0;
    }
    if (strcmp(srname, xerbla_srnamt) != 0) {
        fprintf(stderr, " *** XERBLA was called with SRNAME = %s instead of %s ***\n",
                srname, xerbla_srnamt);
        xerbla_ok = 0;
    }
}
