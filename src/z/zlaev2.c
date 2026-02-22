/**
 * @file zlaev2.c
 * @brief ZLAEV2 computes the eigenvalues and eigenvectors of a 2-by-2
 *        Hermitian matrix.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAEV2 computes the eigendecomposition of a 2-by-2 Hermitian matrix
 *    [  A         B  ]
 *    [  CONJG(B)  C  ].
 * On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
 * eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
 * eigenvector for RT1, giving the decomposition
 *
 * [ CS1  CONJG(SN1) ] [    A     B ] [ CS1 -CONJG(SN1) ] = [ RT1  0  ]
 * [-SN1     CS1     ] [ CONJG(B) C ] [ SN1     CS1     ]   [  0  RT2 ].
 *
 * @param[in]  a    The (1,1) element of the 2-by-2 matrix.
 * @param[in]  b    The (1,2) element and the conjugate of the (2,1) element
 *                  of the 2-by-2 matrix.
 * @param[in]  c    The (2,2) element of the 2-by-2 matrix.
 * @param[out] rt1  The eigenvalue of larger absolute value.
 * @param[out] rt2  The eigenvalue of smaller absolute value.
 * @param[out] cs1  The cosine of the rotation.
 * @param[out] sn1  The vector (CS1, SN1) is a unit right eigenvector for RT1.
 */
void zlaev2(const c128 a, const c128 b,
            const c128 c, f64* rt1, f64* rt2,
            f64* cs1, c128* sn1)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    f64 t;
    c128 w;

    if (cabs(b) == ZERO) {
        w = CMPLX(ONE, 0.0);
    } else {
        w = conj(b) / cabs(b);
    }
    dlaev2(creal(a), cabs(b), creal(c), rt1, rt2, cs1, &t);
    *sn1 = w * t;
}
