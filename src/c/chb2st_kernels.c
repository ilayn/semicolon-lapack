/**
 * @file chb2st_kernels.c
 * @brief CHB2ST_KERNELS is an internal routine used by the CHETRD_HB2ST subroutine.
 */

#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHB2ST_KERNELS is an internal routine used by the CHETRD_HB2ST
 * subroutine.
 *
 * @param[in]     uplo   'U' or 'L'
 * @param[in]     wantz  Indicates if eigenvectors are requested.
 * @param[in]     ttype  Internal type parameter.
 * @param[in]     st     Internal index parameter (0-based).
 * @param[in]     ed     Internal index parameter (0-based).
 * @param[in]     sweep  Internal index parameter (0-based).
 * @param[in]     n      The order of the matrix A.
 * @param[in]     nb     The size of the band.
 * @param[in]     ib     Internal block size parameter.
 * @param[in,out] A      Single complex array. A pointer to the matrix A.
 * @param[in]     lda    The leading dimension of the matrix A.
 * @param[out]    V      Single complex array, dimension 2*n.
 * @param[out]    TAU    Single complex array, dimension 2*n.
 * @param[in]     ldvt   Leading dimension parameter for V/TAU storage.
 * @param[out]    WORK   Single complex array. Workspace of size nb.
 */
void chb2st_kernels(
    const char* uplo,
    const INT wantz,
    const INT ttype,
    const INT st,
    const INT ed,
    const INT sweep,
    const INT n,
    const INT nb,
    const INT ib,
    c64* restrict A,
    const INT lda,
    c64* restrict V,
    c64* restrict TAU,
    const INT ldvt,
    c64* restrict WORK)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    (void)wantz; (void)ib; (void)ldvt;

    INT upper;
    INT i, j1, j2, lm, ln, vpos, taupos, dpos, ofdpos;
    c64 ctmp;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        dpos   = 2 * nb;
        ofdpos = 2 * nb - 1;
    } else {
        dpos   = 0;
        ofdpos = 1;
    }

    /*
     * Upper case
     */
    if (upper) {

        vpos   = (sweep % 2) * n + st;
        taupos = (sweep % 2) * n + st;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = ONE;
            for (i = 1; i < lm; i++) {
                V[vpos + i]                      = conjf(A[(ofdpos - i) + (st + i) * lda]);
                A[(ofdpos - i) + (st + i) * lda] = ZERO;
            }
            ctmp = conjf(A[ofdpos + st * lda]);
            clarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
            A[ofdpos + st * lda] = ctmp;

            lm = ed - st + 1;
            clarfy(uplo, lm, &V[vpos], 1,
                   conjf(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);
        }

        if (ttype == 3) {

            lm = ed - st + 1;
            clarfy(uplo, lm, &V[vpos], 1,
                   conjf(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n - 1;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;
            if (lm > 0) {
                clarfx("Left", ln, lm, &V[vpos],
                       conjf(TAU[taupos]),
                       &A[(dpos - nb) + j1 * lda], lda - 1, WORK);

                vpos   = (sweep % 2) * n + j1;
                taupos = (sweep % 2) * n + j1;

                V[vpos] = ONE;
                for (i = 1; i < lm; i++) {
                    V[vpos + i]                           = conjf(A[(dpos - nb - i) + (j1 + i) * lda]);
                    A[(dpos - nb - i) + (j1 + i) * lda]   = ZERO;
                }
                ctmp = conjf(A[(dpos - nb) + j1 * lda]);
                clarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
                A[(dpos - nb) + j1 * lda] = ctmp;

                clarfx("Right", ln - 1, lm, &V[vpos],
                       TAU[taupos],
                       &A[(dpos - nb + 1) + j1 * lda], lda - 1, WORK);
            }
        }

    /*
     * Lower case
     */
    } else {

        vpos   = (sweep % 2) * n + st;
        taupos = (sweep % 2) * n + st;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = ONE;
            for (i = 1; i < lm; i++) {
                V[vpos + i]                          = A[(ofdpos + i) + (st - 1) * lda];
                A[(ofdpos + i) + (st - 1) * lda]     = ZERO;
            }
            clarfg(lm, &A[ofdpos + (st - 1) * lda], &V[vpos + 1], 1,
                   &TAU[taupos]);

            lm = ed - st + 1;

            clarfy(uplo, lm, &V[vpos], 1,
                   conjf(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);

        }

        if (ttype == 3) {
            lm = ed - st + 1;

            clarfy(uplo, lm, &V[vpos], 1,
                   conjf(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);

        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n - 1;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;

            if (lm > 0) {
                clarfx("Right", lm, ln, &V[vpos],
                       TAU[taupos], &A[(dpos + nb) + st * lda],
                       lda - 1, WORK);

                vpos   = (sweep % 2) * n + j1;
                taupos = (sweep % 2) * n + j1;

                V[vpos] = ONE;
                for (i = 1; i < lm; i++) {
                    V[vpos + i]                      = A[(dpos + nb + i) + st * lda];
                    A[(dpos + nb + i) + st * lda]     = ZERO;
                }
                clarfg(lm, &A[(dpos + nb) + st * lda], &V[vpos + 1], 1,
                       &TAU[taupos]);

                clarfx("Left", lm, ln - 1, &V[vpos],
                       conjf(TAU[taupos]),
                       &A[(dpos + nb - 1) + (st + 1) * lda], lda - 1, WORK);

            }
        }
    }

    return;
}
