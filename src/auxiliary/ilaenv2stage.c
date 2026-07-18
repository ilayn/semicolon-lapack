/**
 * @file ilaenv2stage.c
 * @brief ILAENV2STAGE chooses problem-dependent parameters for 2-stage algorithms.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * ILAENV2STAGE is called from the LAPACK routines to choose problem-dependent
 * parameters for the local environment. See `ispec` for a description of
 * the parameters.
 *
 * It sets problem and machine dependent parameters useful for `*_2STAGE` and
 * related subroutines.
 *
 * This version provides a set of parameters which should give good,
 * but not optimal, performance on many of the currently available
 * computers for the 2-stage solvers. Users are encouraged to modify this
 * subroutine to set the tuning parameters for their particular machine using
 * the option and problem size information in the arguments.
 *
 * @param[in] ispec Specifies the parameter to be returned as the value of
 *                  ILAENV2STAGE.
 *                  `ispec=1`: the optimal blocksize nb for the reduction to BAND.
 *                  `ispec=2`: the optimal blocksize ib for the eigenvectors
 *                  singular vectors update routine.
 *                  `ispec=3`: The length of the array that store the Housholder
 *                  representation for the second stage Band to Tridiagonal or
 *                  Bidiagonal.
 *                  `ispec=4`: The workspace needed for the routine in input.
 *                  `ispec=5`: For future release.
 * @param[in] name  The name of the calling subroutine, in either upper case or
 *                  lower case.
 * @param[in] opts  The character options to the subroutine `name`, concatenated
 *                  into a single character string. For example, `uplo='U'`,
 *                  `trans='T'`, and `diag='N'` for a triangular routine would
 *                  be specified as `opts="UTN"`.
 * @param[in] n1    Problem dimension for the subroutine `name`; this may not be
 *                  required.
 * @param[in] n2    Problem dimension for the subroutine `name`; this may not be
 *                  required.
 * @param[in] n3    Problem dimension for the subroutine `name`; this may not be
 *                  required.
 * @param[in] n4    Problem dimension for the subroutine `name`; this may not be
 *                  required.
 *
 * @return If the return value `>=0`, it is the value of the parameter specified
 *         by `ispec`. If the return value `<0`, then if the value is `-k`, the
 *         k-th argument had an illegal value.
 *
 * @par Further Details:
 * @rst
 * The following conventions have been used when calling ILAENV2STAGE
 * from the LAPACK routines:
 *
 * 1) OPTS is a concatenation of all of the character options to
 *    subroutine NAME, in the same order that they appear in the
 *    argument list for NAME, even if they are not used in determining
 *    the value of the parameter specified by ISPEC.
 *
 * 2) The problem dimensions N1, N2, N3, N4 are specified in the order
 *    that they appear in the argument list for NAME.  N1 is used
 *    first, N2 second, and so on, and unused problem dimensions are
 *    passed a value of -1.
 *
 * 3) The parameter value returned by ILAENV2STAGE is checked for validity in
 *    the calling subroutine.
 * @endrst
 */
INT ilaenv2stage(const INT ispec, const char* name, const char* opts,
                 const INT n1, const INT n2, const INT n3, const INT n4)
{
    INT iispec;

    if (ispec < 1 || ispec > 5) {
        return -1;
    }

    iispec = 16 + ispec;
    return iparam2stage(iispec, name, opts, n1, n2, n3, n4);
}
