/**
 * @file zhb2st_kernels.c
 * @brief ZHB2ST_KERNELS is an internal routine used by the ZHETRD_HB2ST subroutine.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHB2ST_KERNELS is an internal routine used by the ZHETRD_HB2ST
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
 * @param[in,out] A      Double complex array. A pointer to the matrix A.
 * @param[in]     lda    The leading dimension of the matrix A.
 * @param[out]    V      Double complex array, dimension 2*n.
 * @param[out]    TAU    Double complex array, dimension 2*n.
 * @param[in]     ldvt   Leading dimension parameter for V/TAU storage.
 * @param[out]    WORK   Double complex array. Workspace of size nb.
 */
void zhb2st_kernels(
    const char* uplo,
    const int wantz,
    const int ttype,
    const int st,
    const int ed,
    const int sweep,
    const int n,
    const int nb,
    const int ib,
    c128* const restrict A,
    const int lda,
    c128* const restrict V,
    c128* const restrict TAU,
    const int ldvt,
    c128* const restrict WORK)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE = CMPLX(1.0, 0.0);

    (void)wantz; (void)ib; (void)ldvt;

    int upper;
    int i, j1, j2, lm, ln, vpos, taupos, dpos, ofdpos;
    c128 ctmp;

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
                V[vpos + i]                      = conj(A[(ofdpos - i) + (st + i) * lda]);
                A[(ofdpos - i) + (st + i) * lda] = ZERO;
            }
            ctmp = conj(A[ofdpos + st * lda]);
            zlarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
            A[ofdpos + st * lda] = ctmp;

            lm = ed - st + 1;
            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);
        }

        if (ttype == 3) {

            lm = ed - st + 1;
            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n - 1;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;
            if (lm > 0) {
                zlarfx("Left", ln, lm, &V[vpos],
                       conj(TAU[taupos]),
                       &A[(dpos - nb) + j1 * lda], lda - 1, WORK);

                vpos   = (sweep % 2) * n + j1;
                taupos = (sweep % 2) * n + j1;

                V[vpos] = ONE;
                for (i = 1; i < lm; i++) {
                    V[vpos + i]                           = conj(A[(dpos - nb - i) + (j1 + i) * lda]);
                    A[(dpos - nb - i) + (j1 + i) * lda]   = ZERO;
                }
                ctmp = conj(A[(dpos - nb) + j1 * lda]);
                zlarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
                A[(dpos - nb) + j1 * lda] = ctmp;

                zlarfx("Right", ln - 1, lm, &V[vpos],
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
            zlarfg(lm, &A[ofdpos + (st - 1) * lda], &V[vpos + 1], 1,
                   &TAU[taupos]);

            lm = ed - st + 1;

            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);

        }

        if (ttype == 3) {
            lm = ed - st + 1;

            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + st * lda], lda - 1, WORK);

        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n - 1;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;

            if (lm > 0) {
                zlarfx("Right", lm, ln, &V[vpos],
                       TAU[taupos], &A[(dpos + nb) + st * lda],
                       lda - 1, WORK);

                vpos   = (sweep % 2) * n + j1;
                taupos = (sweep % 2) * n + j1;

                V[vpos] = ONE;
                for (i = 1; i < lm; i++) {
                    V[vpos + i]                      = A[(dpos + nb + i) + st * lda];
                    A[(dpos + nb + i) + st * lda]     = ZERO;
                }
                zlarfg(lm, &A[(dpos + nb) + st * lda], &V[vpos + 1], 1,
                       &TAU[taupos]);

                zlarfx("Left", lm, ln - 1, &V[vpos],
                       conj(TAU[taupos]),
                       &A[(dpos + nb - 1) + (st + 1) * lda], lda - 1, WORK);

            }
        }
    }

    return;
}
