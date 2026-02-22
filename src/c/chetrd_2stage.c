/**
 * @file chetrd_2stage.c
 * @brief CHETRD_2STAGE reduces a complex Hermitian matrix to tridiagonal form using 2-stage algorithm.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

void chetrd_2stage(const char* vect, const char* uplo, const INT n,
                   c64* A, const INT lda,
                   f32* D, f32* E, c64* tau,
                   c64* hous2, const INT lhous2,
                   c64* work, const INT lwork, INT* info)
{
    INT lquery, upper;
    INT kd, ib, lwmin, lhmin, lwrk, ldab, wpos, abpos;

    *info = 0;
    (void)(vect[0] == 'V' || vect[0] == 'v');  /* wantq set but not used */
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1) || (lhous2 == -1);

    kd = ilaenv2stage(1, "CHETRD_2STAGE", vect, n, -1, -1, -1);
    ib = ilaenv2stage(2, "CHETRD_2STAGE", vect, n, kd, -1, -1);
    if (n == 0) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "CHETRD_2STAGE", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "CHETRD_2STAGE", vect, n, kd, ib, -1);
    }

    if (!(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -5;
    } else if (lhous2 < lhmin && !lquery) {
        *info = -10;
    } else if (lwork < lwmin && !lquery) {
        *info = -12;
    }

    if (*info == 0) {
        hous2[0] = CMPLXF((f32)lhmin, 0.0f);
        work[0] = CMPLXF((f32)lwmin, 0.0f);
    }

    if (*info != 0) {
        xerbla("CHETRD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    ldab = kd + 1;
    lwrk = lwork - ldab * n;
    abpos = 0;
    wpos = abpos + ldab * n;

    chetrd_he2hb(uplo, n, kd, A, lda, &work[abpos], ldab,
                 tau, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("CHETRD_HE2HB", -(*info));
        return;
    }

    chetrd_hb2st("Y", vect, uplo, n, kd,
                 &work[abpos], ldab, D, E,
                 hous2, lhous2, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("CHETRD_HB2ST", -(*info));
        return;
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
