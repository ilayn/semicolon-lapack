/**
 * @file chkxer.c
 * @brief Port of LAPACK TESTING/EIG/chkxer.f
 *
 * Checks that xerbla was called (LERR flag set). If not, prints
 * a diagnostic and marks the overall test as failed.
 */

#include <stdio.h>
#include "verify.h"

void chkxer(const char* srnamt, int infot, int* lerr, int* ok) {
    if (!(*lerr)) {
        fprintf(stderr, " *** Illegal value of parameter number %d not detected by %s ***\n",
                infot, srnamt);
        *ok = 0;
    }
    *lerr = 0;
}
