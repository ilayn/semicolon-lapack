/**
 * @file ssytrd_2stage.c
 * @brief SSYTRD_2STAGE reduces a real symmetric matrix to tridiagonal form using 2-stage algorithm.
 */

#include "semicolon_lapack_single.h"

void ssytrd_2stage(const char* vect, const char* uplo, const INT n,
                   f32* A, const INT lda,
                   f32* D, f32* E, f32* tau,
                   f32* hous2, const INT lhous2,
                   f32* work, const INT lwork, INT* info)
{
    INT lquery, upper;
    INT kd, ib, lwmin, lhmin, lwrk, ldab, wpos, abpos;

    *info = 0;
    (void)(vect[0] == 'V' || vect[0] == 'v');  /* wantq set but not used */
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1) || (lhous2 == -1);

    kd = ilaenv2stage(1, "SSYTRD_2STAGE", vect, n, -1, -1, -1);
    ib = ilaenv2stage(2, "SSYTRD_2STAGE", vect, n, kd, -1, -1);
    if (n == 0) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "SSYTRD_2STAGE", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "SSYTRD_2STAGE", vect, n, kd, ib, -1);
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
        hous2[0] = (f32)lhmin;
        work[0] = (f32)lwmin;
    }

    if (*info != 0) {
        xerbla("SSYTRD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        work[0] = 1.0f;
        return;
    }

    ldab = kd + 1;
    lwrk = lwork - ldab * n;
    abpos = 0;
    wpos = abpos + ldab * n;

    ssytrd_sy2sb(uplo, n, kd, A, lda, &work[abpos], ldab,
                 tau, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("SSYTRD_SY2SB", -(*info));
        return;
    }

    ssytrd_sb2st("Y", vect, uplo, n, kd,
                 &work[abpos], ldab, D, E,
                 hous2, lhous2, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("SSYTRD_SB2ST", -(*info));
        return;
    }

    work[0] = (f32)lwmin;
}
