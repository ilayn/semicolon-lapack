/**
 * @file dlamrg.c
 * @brief DLAMRG creates a permutation list to merge the entries of two
 *        independently sorted sets into a single set sorted in ascending order.
 */

#include "semicolon_lapack_double.h"

/**
 * DLAMRG will create a permutation list which will merge the elements
 * of A (which is composed of two independently sorted sets) into a
 * single set which is sorted in ascending order.
 *
 * @param[in]     n1     The length of the first sorted list.
 * @param[in]     n2     The length of the second sorted list.
 * @param[in]     A      Double precision array, dimension (n1+n2).
 *                       The first n1 elements contain a list sorted in
 *                       ascending or descending order. Likewise for the
 *                       final n2 elements.
 * @param[in]     dtrd1  Stride through the first subset of A.
 *                       1 = ascending, -1 = descending.
 * @param[in]     dtrd2  Stride through the second subset of A.
 *                       1 = ascending, -1 = descending.
 * @param[out]    index  Integer array, dimension (n1+n2).
 *                       On exit, a permutation with 0-based indices such that if
 *                       B[i] = A[index[i]] for i=0,...,n1+n2-1,
 *                       then B is sorted in ascending order.
 */
void dlamrg(const INT n1, const INT n2, const f64* A,
            const INT dtrd1, const INT dtrd2, INT* index)
{
    INT n1sv = n1;
    INT n2sv = n2;
    INT ind1, ind2;

    if (dtrd1 > 0) {
        ind1 = 0;
    } else {
        ind1 = n1 - 1;
    }
    if (dtrd2 > 0) {
        ind2 = n1;
    } else {
        ind2 = n1 + n2 - 1;
    }

    INT i = 0;

    while (n1sv > 0 && n2sv > 0) {
        if (A[ind1] <= A[ind2]) {
            index[i] = ind1;
            i++;
            ind1 += dtrd1;
            n1sv--;
        } else {
            index[i] = ind2;
            i++;
            ind2 += dtrd2;
            n2sv--;
        }
    }

    if (n1sv == 0) {
        for (INT j = 0; j < n2sv; j++) {
            index[i] = ind2;
            i++;
            ind2 += dtrd2;
        }
    } else {
        for (INT j = 0; j < n1sv; j++) {
            index[i] = ind1;
            i++;
            ind1 += dtrd1;
        }
    }
}
