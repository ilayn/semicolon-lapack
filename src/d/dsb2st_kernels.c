/**
 * @file dsb2st_kernels.c
 * @brief DSB2ST_KERNELS is an internal routine used by DSYTRD_SB2ST.
 */

#include "semicolon_lapack_double.h"

/**
 * DSB2ST_KERNELS is an internal routine used by the DSYTRD_SB2ST subroutine.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangle is stored.
 * @param[in]     wantz Indicates if eigenvalues only are requested, or both
 *                      eigenvalues and eigenvectors.
 * @param[in]     ttype Internal parameter.
 * @param[in]     st    Internal parameter for indices.
 * @param[in]     ed    Internal parameter for indices.
 * @param[in]     sweep Internal parameter for indices.
 * @param[in]     n     The order of the matrix `A`.
 * @param[in]     nb    The size of the band.
 * @param[in]     ib    Internal parameter.
 * @param[in,out] A     A pointer to the matrix A.
 * @param[in]     lda   The leading dimension of the matrix `A`.
 * @param[out]    V     Array of dimension `2*n` if eigenvalues only are requested or
 *                      to be queried for vectors.
 * @param[out]    tau   Array of dimension (`2*n`). The scalar factors of the
 *                      Householder reflectors are stored in this array.
 * @param[in]     ldvt  Internal parameter.
 * @param[out]    work  Workspace of size `nb`.
 *
 * @par Further Details:
 * @rst
 * Implemented by Azzam Haidar.
 *
 * All details are available on technical report, SC11, SC13 papers.
 *
 * Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
 * Parallel reduction to condensed forms for symmetric eigenvalue problems
 * using aggregated fine-grained and memory-aware kernels. In Proceedings
 * of 2011 International Conference for High Performance Computing,
 * Networking, Storage and Analysis (SC '11), New York, NY, USA,
 * Article 8 , 11 pages.
 * https://doi.org/10.1145/2063384.2063394
 *
 * A. Haidar, J. Kurzak, P. Luszczek, 2013.
 * An improved parallel singular value algorithm and its implementation
 * for multicore hardware, In Proceedings of 2013 International Conference
 * for High Performance Computing, Networking, Storage and Analysis (SC '13).
 * Denver, Colorado, USA, 2013.
 * Article 90, 12 pages.
 * https://doi.org/10.1145/2503210.2503292
 *
 * A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
 * A novel hybrid CPU-GPU generalized eigensolver for electronic structure
 * calculations based on fine-grained memory aware tasks.
 * International Journal of High Performance Computing Applications.
 * Volume 28 Issue 2, Pages 196-209, May 2014.
 * https://doi.org/10.1177/1094342013502097
 * @endrst
 */
void dsb2st_kernels(const char* uplo, const INT wantz, const INT ttype,
                    const INT st, const INT ed, const INT sweep,
                    const INT n, const INT nb, const INT ib,
                    f64* A, const INT lda,
                    f64* V, f64* tau, const INT ldvt,
                    f64* work)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    INT upper;
    INT i, j1, j2, lm, ln, vpos, taupos, dpos, ofdpos;
    f64 ctmp;

    (void)ib;
    (void)ldvt;
    (void)wantz;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        dpos = 2 * nb;
        ofdpos = 2 * nb - 1;
    } else {
        dpos = 0;
        ofdpos = 1;
    }

    if (upper) {
        vpos = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = one;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i] = A[(ofdpos - i) + (st - 1 + i) * lda];
                A[(ofdpos - i) + (st - 1 + i) * lda] = zero;
            }
            ctmp = A[ofdpos + (st - 1) * lda];
            dlarfg(lm, &ctmp, &V[vpos + 1], 1, &tau[taupos]);
            A[ofdpos + (st - 1) * lda] = ctmp;

            lm = ed - st + 1;
            dlarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 3) {
            lm = ed - st + 1;
            dlarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;
            if (lm > 0) {
                dlarfx("L", ln, lm, &V[vpos], tau[taupos],
                       &A[(dpos - nb) + (j1 - 1) * lda], lda - 1, work);

                vpos = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = one;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i] = A[(dpos - nb - i) + (j1 - 1 + i) * lda];
                    A[(dpos - nb - i) + (j1 - 1 + i) * lda] = zero;
                }
                ctmp = A[(dpos - nb) + (j1 - 1) * lda];
                dlarfg(lm, &ctmp, &V[vpos + 1], 1, &tau[taupos]);
                A[(dpos - nb) + (j1 - 1) * lda] = ctmp;

                dlarfx("R", ln - 1, lm, &V[vpos], tau[taupos],
                       &A[(dpos - nb + 1) + (j1 - 1) * lda], lda - 1, work);
            }
        }

    } else {
        vpos = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = one;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i] = A[(ofdpos + i) + (st - 2) * lda];
                A[(ofdpos + i) + (st - 2) * lda] = zero;
            }
            dlarfg(lm, &A[ofdpos + (st - 2) * lda], &V[vpos + 1], 1, &tau[taupos]);

            lm = ed - st + 1;

            dlarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 3) {
            lm = ed - st + 1;

            dlarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;

            if (lm > 0) {
                dlarfx("R", lm, ln, &V[vpos], tau[taupos],
                       &A[(dpos + nb) + (st - 1) * lda], lda - 1, work);

                vpos = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = one;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i] = A[(dpos + nb + i) + (st - 1) * lda];
                    A[(dpos + nb + i) + (st - 1) * lda] = zero;
                }
                dlarfg(lm, &A[(dpos + nb) + (st - 1) * lda], &V[vpos + 1], 1, &tau[taupos]);

                dlarfx("L", lm, ln - 1, &V[vpos], tau[taupos],
                       &A[(dpos + nb - 1) + st * lda], lda - 1, work);
            }
        }
    }
}
