/**
 * @file dlasrt.c
 * @brief DLASRT sorts numbers in increasing or decreasing order.
 */

#include "semicolon_lapack_double.h"

/**
 * Sort the numbers in D in increasing order (if ID = 'I') or
 * in decreasing order (if ID = 'D').
 *
 * Uses Quick Sort, reverting to Insertion sort on arrays of
 * size <= 20. Dimension of STACK limits N to about 2**32.
 *
 * @param[in]     id    = 'I': sort D in increasing order;
 *                       = 'D': sort D in decreasing order.
 * @param[in]     n     The length of the array D.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the array to be sorted.
 *                      On exit, D has been sorted into increasing order
 *                      (D[0] <= ... <= D[n-1]) or into decreasing order
 *                      (D[0] >= ... >= D[n-1]), depending on ID.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dlasrt(const char* id, const INT n, f64* restrict D, INT* info)
{
    const INT SELECT = 20;

    INT dir, endd, i, j, start, stkpnt;
    f64 d1, d2, d3, dmnmx, tmp;
    INT stack[2][32];

    /* Test the input parameters */
    *info = 0;
    dir = -1;
    if (id[0] == 'D' || id[0] == 'd') {
        dir = 0;
    } else if (id[0] == 'I' || id[0] == 'i') {
        dir = 1;
    }
    if (dir == -1) {
        *info = -1;
        return;
    } else if (n < 0) {
        *info = -2;
        return;
    }

    /* Quick return if possible */
    if (n <= 1) return;

    stkpnt = 1;
    stack[0][0] = 0;       /* start (0-based) */
    stack[1][0] = n - 1;   /* end (0-based) */

    while (stkpnt > 0) {
        start = stack[0][stkpnt - 1];
        endd = stack[1][stkpnt - 1];
        stkpnt--;

        if (endd - start <= SELECT && endd - start > 0) {
            /* Do Insertion sort on D[start:endd] */
            if (dir == 0) {
                /* Sort into decreasing order */
                for (i = start + 1; i <= endd; i++) {
                    for (j = i; j >= start + 1; j--) {
                        if (D[j] > D[j - 1]) {
                            dmnmx = D[j];
                            D[j] = D[j - 1];
                            D[j - 1] = dmnmx;
                        } else {
                            break;
                        }
                    }
                }
            } else {
                /* Sort into increasing order */
                for (i = start + 1; i <= endd; i++) {
                    for (j = i; j >= start + 1; j--) {
                        if (D[j] < D[j - 1]) {
                            dmnmx = D[j];
                            D[j] = D[j - 1];
                            D[j - 1] = dmnmx;
                        } else {
                            break;
                        }
                    }
                }
            }
        } else if (endd - start > SELECT) {
            /* Partition D[start:endd] and stack parts, largest one first */

            /* Choose partition entry as median of 3 */
            d1 = D[start];
            d2 = D[endd];
            i = (start + endd) / 2;
            d3 = D[i];
            if (d1 < d2) {
                if (d3 < d1) {
                    dmnmx = d1;
                } else if (d3 < d2) {
                    dmnmx = d3;
                } else {
                    dmnmx = d2;
                }
            } else {
                if (d3 < d2) {
                    dmnmx = d2;
                } else if (d3 < d1) {
                    dmnmx = d3;
                } else {
                    dmnmx = d1;
                }
            }

            if (dir == 0) {
                /* Sort into decreasing order */
                i = start - 1;
                j = endd + 1;
                for (;;) {
                    do { j--; } while (D[j] < dmnmx);
                    do { i++; } while (D[i] > dmnmx);
                    if (i < j) {
                        tmp = D[i];
                        D[i] = D[j];
                        D[j] = tmp;
                    } else {
                        break;
                    }
                }
                if (j - start > endd - j - 1) {
                    stkpnt++;
                    stack[0][stkpnt - 1] = start;
                    stack[1][stkpnt - 1] = j;
                    stkpnt++;
                    stack[0][stkpnt - 1] = j + 1;
                    stack[1][stkpnt - 1] = endd;
                } else {
                    stkpnt++;
                    stack[0][stkpnt - 1] = j + 1;
                    stack[1][stkpnt - 1] = endd;
                    stkpnt++;
                    stack[0][stkpnt - 1] = start;
                    stack[1][stkpnt - 1] = j;
                }
            } else {
                /* Sort into increasing order */
                i = start - 1;
                j = endd + 1;
                for (;;) {
                    do { j--; } while (D[j] > dmnmx);
                    do { i++; } while (D[i] < dmnmx);
                    if (i < j) {
                        tmp = D[i];
                        D[i] = D[j];
                        D[j] = tmp;
                    } else {
                        break;
                    }
                }
                if (j - start > endd - j - 1) {
                    stkpnt++;
                    stack[0][stkpnt - 1] = start;
                    stack[1][stkpnt - 1] = j;
                    stkpnt++;
                    stack[0][stkpnt - 1] = j + 1;
                    stack[1][stkpnt - 1] = endd;
                } else {
                    stkpnt++;
                    stack[0][stkpnt - 1] = j + 1;
                    stack[1][stkpnt - 1] = endd;
                    stkpnt++;
                    stack[0][stkpnt - 1] = start;
                    stack[1][stkpnt - 1] = j;
                }
            }
        }
    }
}
