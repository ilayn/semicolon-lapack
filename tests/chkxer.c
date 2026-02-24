/**
 * @file chkxer.c
 * @brief Port of LAPACK TESTING/EIG/chkxer.f
 *
 * Checks that xerbla was called (LERR flag set). If not, prints
 * a diagnostic and marks the overall test as failed.
 */

#include <stdio.h>
#include "verify.h"

void chkxer(const char* srnamt, INT infot, INT* lerr, INT* ok) {
    if (!(*lerr)) {
        fprintf(stderr, " *** Illegal value of parameter number %lld not detected by %s ***\n",
                (long long)infot, srnamt);
        *ok = 0;
    }
    *lerr = 0;
}
