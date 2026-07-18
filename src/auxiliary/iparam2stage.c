/**
 * @file iparam2stage.c
 * @brief IPARAM2STAGE sets problem and machine dependent parameters for 2-stage algorithms.
 */

#include "semicolon_lapack_auxiliary.h"
#include "lapack_tuning.h"
#include <string.h>

/**
 * This program sets problem and machine dependent parameters
 * useful for xHETRD_2STAGE, xHETRD_HE2HB, xHETRD_HB2ST,
 * xGEBRD_2STAGE, xGEBRD_GE2GB, xGEBRD_GB2BD
 * and related subroutines for eigenvalue problems.
 *
 * It is called whenever ILAENV is called with `17<=ispec<=21`.
 * It is called whenever ILAENV2STAGE is called with `1<=ispec<=5`
 * with a direct conversion `ispec+16`.
 *
 * @param[in] ispec `ispec` specifies which tunable parameter IPARAM2STAGE should
 *                  return.
 *                  `ispec=17`: the optimal blocksize nb for the reduction to BAND.
 *                  `ispec=18`: the optimal blocksize ib for the eigenvectors
 *                  singular vectors update routine.
 *                  `ispec=19`: The length of the array that store the Housholder
 *                  representation for the second stage Band to Tridiagonal or
 *                  Bidiagonal.
 *                  `ispec=20`: The workspace needed for the routine in input.
 *                  `ispec=21`: For future release.
 * @param[in] name  Name of the calling subroutine.
 * @param[in] opts  The character options to the subroutine `name`, concatenated
 *                  into a single character string. For example, `uplo='U'`,
 *                  `trans='T'`, and `diag='N'` for a triangular routine would
 *                  be specified as `opts="UTN"`.
 * @param[in] ni    The size of the matrix.
 * @param[in] nbi   Used in the reduction, (e.g., the size of the band), needed to
 *                  compute workspace and LHOUS2.
 * @param[in] ibi   Represents the IB of the reduction, needed to compute workspace
 *                  and LHOUS2.
 * @param[in] nxi   Needed in the future release.
 *
 * @return The value of the tunable parameter selected by `ispec`, or `-1` if
 *         `ispec` is out of range or the precision prefix of `name` is not one
 *         of `'S'`, `'D'`, `'C'`, `'Z'`.
 *
 * @par Further Details:
 * @rst
 * Implemented by Azzam Haidar.
 *
 * All detail are available on technical report, SC11, SC13 papers.
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
INT iparam2stage(const INT ispec, const char* name, const char* opts,
                 const INT ni, const INT nbi, const INT ibi, const INT nxi)
{
    INT kd, ib, lhous, lwork, nthreads;
    INT factoptnb, qroptnb, lqoptnb;
    INT rprec, cprec;
    char prec, algo[4], stag[6], vect;
    char subnam[13];
    INT i, ic;

    if (ispec < 17 || ispec > 21) {
        return -1;
    }

    nthreads = 1;

    if (ispec != 19) {
        strncpy(subnam, name, 12);
        subnam[12] = '\0';

        for (i = 0; i < 12 && subnam[i] != '\0'; i++) {
            ic = (unsigned char)subnam[i];
            if (ic >= 97 && ic <= 122) {
                subnam[i] = (char)(ic - 32);
            }
        }

        prec = subnam[0];
        algo[0] = subnam[3];
        algo[1] = subnam[4];
        algo[2] = subnam[5];
        algo[3] = '\0';
        stag[0] = subnam[7];
        stag[1] = subnam[8];
        stag[2] = subnam[9];
        stag[3] = subnam[10];
        stag[4] = subnam[11];
        stag[5] = '\0';

        rprec = (prec == 'S' || prec == 'D');
        cprec = (prec == 'C' || prec == 'Z');

        if (!(rprec || cprec)) {
            return -1;
        }
    }

    if (ispec == 17 || ispec == 18) {
        if (nthreads > 4) {
            if (cprec) {
                kd = 128;
                ib = 32;
            } else {
                kd = 160;
                ib = 40;
            }
        } else if (nthreads > 1) {
            /* LAPACK has identical branches */
            kd = 64;
            ib = 32;
        } else {
            if (cprec) {
                kd = 16;
                ib = 16;
            } else {
                kd = 32;
                ib = 16;
            }
        }
        if (ispec == 17) return kd;
        if (ispec == 18) return ib;

    } else if (ispec == 19) {
        vect = opts[0];
        if (vect == 'N' || vect == 'n') {
            lhous = (4 * ni > 1) ? 4 * ni : 1;
        } else {
            lhous = ((4 * ni > 1) ? 4 * ni : 1) + ibi;
        }
        if (lhous >= 0) {
            return lhous;
        } else {
            return -1;
        }

    } else if (ispec == 20) {
        lwork = -1;
        qroptnb = lapack_get_nb("GEQRF");
        lqoptnb = lapack_get_nb("GELQF");
        factoptnb = (qroptnb > lqoptnb) ? qroptnb : lqoptnb;

        if (strcmp(algo, "TRD") == 0) {
            if (strcmp(stag, "2STAG") == 0) {
                INT tmp1 = 2 * nbi * nbi;
                INT tmp2 = nbi * nthreads;
                INT maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
                INT factmax = (nbi + 1 > factoptnb) ? nbi + 1 : factoptnb;
                lwork = ni * nbi + ni * factmax + maxval + (nbi + 1) * ni;
            } else if (strcmp(stag, "HE2HB") == 0 || strcmp(stag, "SY2SB") == 0) {
                INT factmax = (nbi > factoptnb) ? nbi : factoptnb;
                lwork = ni * nbi + ni * factmax + 2 * nbi * nbi;
            } else if (strcmp(stag, "HB2ST") == 0 || strcmp(stag, "SB2ST") == 0) {
                lwork = (2 * nbi + 1) * ni + nbi * nthreads;
            }
        } else if (strcmp(algo, "BRD") == 0) {
            if (strcmp(stag, "2STAG") == 0) {
                INT tmp1 = 2 * nbi * nbi;
                INT tmp2 = nbi * nthreads;
                INT maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
                INT factmax = (nbi + 1 > factoptnb) ? nbi + 1 : factoptnb;
                lwork = 2 * ni * nbi + ni * factmax + maxval + (nbi + 1) * ni;
            } else if (strcmp(stag, "GE2GB") == 0) {
                INT factmax = (nbi > factoptnb) ? nbi : factoptnb;
                lwork = ni * nbi + ni * factmax + 2 * nbi * nbi;
            } else if (strcmp(stag, "GB2BD") == 0) {
                lwork = (3 * nbi + 1) * ni + nbi * nthreads;
            }
        }
        lwork = (lwork > 1) ? lwork : 1;

        if (lwork > 0) {
            return lwork;
        } else {
            return -1;
        }

    } else if (ispec == 21) {
        return nxi;
    }

    return -1;
}
