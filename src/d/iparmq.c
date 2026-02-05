/**
 * @file iparmq.c
 * @brief IPARMQ returns problem and machine dependent parameters for HSEQR.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * IPARMQ returns problem and machine dependent parameters useful for
 * xHSEQR and related subroutines for eigenvalue problems.
 *
 * @param[in] ispec   Specifies which tunable parameter to return:
 *                    = 12: (INMIN) Matrices of order nmin or less are sent
 *                          directly to xLAHQR. NMIN must be at least 11.
 *                    = 13: (INWIN) Size of the deflation window.
 *                    = 14: (INIBL) Determines when to stop nibbling and
 *                          invest in an (expensive) multi-shift QR sweep.
 *                    = 15: (NSHFTS) The number of simultaneous shifts in
 *                          a multi-shift QR iteration.
 *                    = 16: (IACC22) Whether to accumulate reflections before
 *                          updating far-from-diagonal elements (0, 1, or 2).
 *                    = 17: (ICOST) Relative cost of flops within the
 *                          near-the-diagonal shift chase vs BLAS calls.
 * @param[in] name    Name of the calling subroutine.
 * @param[in] opts    Concatenation of the string arguments to TTQRE.
 * @param[in] n       Order of the Hessenberg matrix H.
 * @param[in] ilo     It is assumed that H is already upper triangular
 *                    in rows and columns 0:ilo-1 (0-based).
 * @param[in] ihi     It is assumed that H is already upper triangular
 *                    in rows and columns ihi+1:n-1 (0-based).
 * @param[in] lwork   The amount of workspace available.
 *
 * @return The requested parameter value, or -1 if ispec is invalid.
 */
int iparmq(
    const int ispec,
    const char* name,
    const char* opts,
    const int n,
    const int ilo,
    const int ihi,
    const int lwork)
{
    /* Parameters */
    const int INMIN = 12;
    const int INWIN = 13;
    const int INIBL = 14;
    const int ISHFTS = 15;
    const int IACC22 = 16;
    const int ICOST = 17;

    /* Default values from LAPACK iparmq.f */
    const int NMIN = 75;      /* Crossover point for xLAHQR vs xLAQR0 */
    const int K22MIN = 14;    /* Min size for 2x2 block structure */
    const int KACMIN = 14;    /* Min size for accumulation */
    const int NIBBLE = 14;    /* Nibble crossover point */
    const int KNWSWP = 500;   /* Deflation window size threshold */
    const int RCOST = 10;     /* Relative cost (percentage) */
    const double TWO = 2.0;

    int nh = 0, ns = 0;
    int result;

    /* Unused parameters (matching Fortran interface) */
    (void)opts;
    (void)n;
    (void)lwork;

    /* Compute NH = active matrix size and NS = number of shifts */
    if (ispec == ISHFTS || ispec == INWIN || ispec == IACC22) {
        nh = ihi - ilo + 1;
        ns = 2;
        if (nh >= 30) {
            ns = 4;
        }
        if (nh >= 60) {
            ns = 10;
        }
        if (nh >= 150) {
            int log2_nh = (int)(log((double)nh) / log(TWO) + 0.5);
            ns = nh / log2_nh;
            if (ns < 10) {
                ns = 10;
            }
        }
        if (nh >= 590) {
            ns = 64;
        }
        if (nh >= 3000) {
            ns = 128;
        }
        if (nh >= 6000) {
            ns = 256;
        }
        /* Make NS even */
        ns = ns - (ns % 2);
        if (ns < 2) {
            ns = 2;
        }
    }

    if (ispec == INMIN) {
        /* Matrices of order smaller than NMIN get sent to xLAHQR */
        result = NMIN;

    } else if (ispec == INIBL) {
        /* Nibble crossover point */
        result = NIBBLE;

    } else if (ispec == ISHFTS) {
        /* Number of simultaneous shifts */
        result = ns;

    } else if (ispec == INWIN) {
        /* Deflation window size */
        if (nh <= KNWSWP) {
            result = ns;
        } else {
            result = 3 * ns / 2;
        }

    } else if (ispec == IACC22) {
        /* Whether to accumulate reflections */
        result = 0;

        /* Convert first 6 characters of NAME to uppercase for comparison */
        char subnam[7];
        int i;
        for (i = 0; i < 6 && name[i] != '\0'; i++) {
            char c = name[i];
            /* Convert lowercase to uppercase (ASCII) */
            if (c >= 'a' && c <= 'z') {
                subnam[i] = c - 32;
            } else {
                subnam[i] = c;
            }
        }
        for (; i < 6; i++) {
            subnam[i] = ' ';
        }
        subnam[6] = '\0';

        /* Check for specific routines */
        /* GGHRD or GGHD3 */
        if ((subnam[1] == 'G' && subnam[2] == 'G' && subnam[3] == 'H' &&
             subnam[4] == 'R' && subnam[5] == 'D') ||
            (subnam[1] == 'G' && subnam[2] == 'G' && subnam[3] == 'H' &&
             subnam[4] == 'D' && subnam[5] == '3')) {
            result = 1;
            if (nh >= K22MIN) {
                result = 2;
            }
        }
        /* xEXC (DTREXC, etc.) - check positions 3-5 */
        else if (subnam[3] == 'E' && subnam[4] == 'X' && subnam[5] == 'C') {
            if (nh >= KACMIN) {
                result = 1;
            }
            if (nh >= K22MIN) {
                result = 2;
            }
        }
        /* HSEQR or xLAQR */
        else if ((subnam[1] == 'H' && subnam[2] == 'S' && subnam[3] == 'E' &&
                  subnam[4] == 'Q' && subnam[5] == 'R') ||
                 (subnam[1] == 'L' && subnam[2] == 'A' && subnam[3] == 'Q' &&
                  subnam[4] == 'R')) {
            if (ns >= KACMIN) {
                result = 1;
            }
            if (ns >= K22MIN) {
                result = 2;
            }
        }

    } else if (ispec == ICOST) {
        /* Relative cost of near-the-diagonal chase vs BLAS updates */
        result = RCOST;

    } else {
        /* Invalid value of ispec */
        result = -1;
    }

    return result;
}
