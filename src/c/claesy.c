/**
 * @file claesy.c
 * @brief CLAESY computes the eigenvalues and eigenvectors of a 2-by-2
 *        complex symmetric matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>

/**
 * CLAESY computes the eigendecomposition of a 2-by-2 symmetric matrix
 *    ( ( A, B );( B, C ) )
 * provided the norm of the matrix of eigenvectors is larger than
 * some threshold value.
 *
 * RT1 is the eigenvalue of larger absolute value, and RT2 of
 * smaller absolute value.  If the eigenvectors are computed, then
 * on return ( CS1, SN1 ) is the unit eigenvector for RT1, hence
 *
 * [  CS1     SN1   ] . [ A  B ] . [ CS1    -SN1   ] = [ RT1  0  ]
 * [ -SN1     CS1   ]   [ B  C ]   [ SN1     CS1   ]   [  0  RT2 ]
 *
 * @param[in]     a       The ( 1, 1 ) element of input matrix.
 * @param[in]     b       The ( 1, 2 ) element of input matrix.  The ( 2, 1 )
 *                        element is also given by B, since the 2-by-2 matrix
 *                        is symmetric.
 * @param[in]     c       The ( 2, 2 ) element of input matrix.
 * @param[out]    rt1     The eigenvalue of larger modulus.
 * @param[out]    rt2     The eigenvalue of smaller modulus.
 * @param[out]    evscal  The complex value by which the eigenvector matrix was
 *                        scaled to make it orthonormal.  If EVSCAL is zero,
 *                        the eigenvectors were not computed.  This means one of
 *                        two things:  the 2-by-2 matrix could not be
 *                        diagonalized, or the norm of the matrix of
 *                        eigenvectors before scaling was larger than the
 *                        threshold value THRESH (set below).
 * @param[out]    cs1     If EVSCAL .NE. 0, ( CS1, SN1 ) is the unit right
 *                        eigenvector for RT1.
 * @param[out]    sn1     If EVSCAL .NE. 0, ( CS1, SN1 ) is the unit right
 *                        eigenvector for RT1.
 */
void claesy(
    const c64 a,
    const c64 b,
    const c64 c,
    c64* rt1,
    c64* rt2,
    c64* evscal,
    c64* cs1,
    c64* sn1)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 HALF = 0.5f;
    const f32 THRESH = 0.1f;

    f32 babs, evnorm, tabs, z;
    c64 s, t, tmp;

    /* Special case:  The matrix is actually diagonal.
       To avoid divide by zero later, we treat this case separately. */

    if (cabsf(b) == ZERO) {
        *rt1 = a;
        *rt2 = c;
        if (cabsf(*rt1) < cabsf(*rt2)) {
            tmp = *rt1;
            *rt1 = *rt2;
            *rt2 = tmp;
            *cs1 = CMPLXF(ZERO, 0.0f);
            *sn1 = CMPLXF(ONE, 0.0f);
        } else {
            *cs1 = CMPLXF(ONE, 0.0f);
            *sn1 = CMPLXF(ZERO, 0.0f);
        }
    } else {

        /* Compute the eigenvalues and eigenvectors.
           The characteristic equation is
              lambda **2 - (A+C) lambda + (A*C - B*B)
           and we solve it using the quadratic formula. */

        s = (a + c) * HALF;
        t = (a - c) * HALF;

        /* Take the square root carefully to avoid over/under flow. */

        babs = cabsf(b);
        tabs = cabsf(t);
        z = fmaxf(babs, tabs);
        if (z > ZERO) {
            t = z * csqrtf((t / z) * (t / z) + (b / z) * (b / z));
        }

        /* Compute the two eigenvalues.  RT1 and RT2 are exchanged
           if necessary so that RT1 will have the greater magnitude. */

        *rt1 = s + t;
        *rt2 = s - t;
        if (cabsf(*rt1) < cabsf(*rt2)) {
            tmp = *rt1;
            *rt1 = *rt2;
            *rt2 = tmp;
        }

        /* Choose CS1 = 1 and SN1 to satisfy the first equation, then
           scale the components of this eigenvector so that the matrix
           of eigenvectors X satisfies  X * X**T = I .  (No scaling is
           done if the norm of the eigenvalue matrix is less than THRESH.) */

        *sn1 = (*rt1 - a) / b;
        tabs = cabsf(*sn1);
        if (tabs > ONE) {
            t = tabs * csqrtf((ONE / tabs) * (ONE / tabs) + (*sn1 / tabs) * (*sn1 / tabs));
        } else {
            t = csqrtf(CONE + (*sn1) * (*sn1));
        }
        evnorm = cabsf(t);
        if (evnorm >= THRESH) {
            *evscal = CONE / t;
            *cs1 = *evscal;
            *sn1 = (*sn1) * (*evscal);
        } else {
            *evscal = CMPLXF(ZERO, 0.0f);
        }
    }
}
