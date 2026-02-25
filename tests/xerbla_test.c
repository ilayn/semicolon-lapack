/**
 * @file xerbla_test.c
 * @brief Test version of xerbla with flag-setting for error exit tests.
 *
 * This provides a strong xerbla symbol that overrides the weak production
 * version in src/auxiliary/xerbla.c via ELF symbol interposition (Linux).
 *
 * On Windows, DLLs do not participate in symbol interposition, so the
 * DLL's internal xerbla is never overridden by this strong symbol.
 * Instead, we set the xerbla_override function pointer (exported from
 * the DLL) to redirect the DLL's xerbla to our test handler.
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
INT    xerbla_infot  = 0;
INT    xerbla_nout   = 0;
INT    xerbla_ok     = 1;
INT    xerbla_lerr   = 0;
char   xerbla_srnamt[33] = "";

void xerbla(const char* srname, INT info) {
    /* When not in error-testing mode, behave like production xerbla */
    if (xerbla_srnamt[0] == '\0') {
        fprintf(stderr, " ** On entry to %s parameter number %lld had an illegal value\n",
                srname, (long long)info);
        return;
    }

    xerbla_lerr = 1;
    if (info != xerbla_infot) {
        fprintf(stderr, " *** XERBLA was called with INFO = %lld instead of %lld ***\n",
                (long long)info, (long long)xerbla_infot);
        fprintf(stderr, " *** routine = %s ***\n", srname);
        xerbla_ok = 0;
    }
    if (strcmp(srname, xerbla_srnamt) != 0) {
        fprintf(stderr, " *** XERBLA was called with SRNAME = %s instead of %s ***\n",
                srname, xerbla_srnamt);
        xerbla_ok = 0;
    }
}

/*
 * On Windows DLLs, the strong xerbla above is never called by the library
 * (DLLs don't do ELF-style symbol interposition). Set xerbla_override so
 * the DLL's xerbla delegates to our test handler.
 *
 * On Linux this is redundant (weak symbol override already works) but harmless.
 */
static void __attribute__((constructor)) xerbla_test_init(void) {
    xerbla_override = xerbla;
}
