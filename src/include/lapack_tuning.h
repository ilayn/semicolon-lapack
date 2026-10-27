/**
 * @file lapack_tuning.h
 * @brief Block size and tuning parameters for LAPACK routines.
 *
 * This header replaces LAPACK's ILAENV mechanism with a simple lookup table.
 * It uses the "single-header library" pattern (STB-style):
 *
 *   - Include normally for declarations only
 *   - Define LAPACK_TUNING_IMPLEMENTATION in exactly ONE .c file to get
 *     the implementation
 *   - For tests, also define LAPACK_TUNING_TEST to enable runtime override
 *     of block sizes (equivalent to LAPACK's XLAENV/ILAENV test mechanism)
 *
 * Production usage (in src/lapack_tuning.c):
 * @code
 *     #define LAPACK_TUNING_IMPLEMENTATION
 *     #include "lapack_tuning.h"
 * @endcode
 *
 * Test usage (in tests/testutils/lapack_tuning_test.c):
 * @code
 *     #define LAPACK_TUNING_IMPLEMENTATION
 *     #define LAPACK_TUNING_TEST
 *     #include "lapack_tuning.h"
 * @endcode
 *
 * The test version provides xlaenv() and allows overriding NB values,
 * mirroring LAPACK's TESTING/LIN/ilaenv.f vs SRC/ilaenv.f approach.
 *
 * Supported ISPEC values (matching LAPACK's ILAENV):
 *   ISPEC=1:  NB      - Block size
 *   ISPEC=2:  NBMIN   - Minimum block size for blocking to be used
 *   ISPEC=3:  NX      - Crossover point (use unblocked if N < NX)
 *   ISPEC=4:  6       - Number of shifts for xHSEQR (DEPRECATED, handled by IPARMQ)
 *   ISPEC=5:  2       - Minimum column dimension (xTGSYL), lapack_get_nbcol()
 *   ISPEC=6:  MNTHR   - SVD crossover point = INT(REAL(MIN(M,N))*1.6E0)
 *   ISPEC=7:  1       - Number of processors (not used)
 *   ISPEC=8:  50      - Crossover point for multishift QR (DEPRECATED, handled by IPARMQ)
 *   ISPEC=9:  SMLSIZ  - Max size of D&C subproblems (LAPACK_SMLSIZ below)
 *   ISPEC=10: 1       - IEEE NaN arithmetic can be trusted (assumed yes)
 *   ISPEC=11: 1       - IEEE infinity arithmetic can be trusted (assumed yes)
 *   ISPEC=12..17:      Handled by iparmq(), through lapack_get_iparmq()
 *   ISPEC=17..21:      Handled by iparam2stage(), through lapack_get_iparam2stage()
 */

#ifndef LAPACK_TUNING_H
#define LAPACK_TUNING_H

#include <stddef.h>  /* for NULL */
#include <stdint.h>

/*
 * ISPEC=9: Maximum size of the subproblems at the bottom of the
 * divide-and-conquer computation tree (used by xGELSD, xGESDD, xLAED0,
 * xSTEDC, xBDSDC).  From ilaenv.f line 718.
 */
#define LAPACK_SMLSIZ 25

typedef struct {
    const char* routine;  /* Name passed to ILAENV, without the precision prefix */
    int nb;               /* Block size (ISPEC=1) */
    int nbmin;            /* Minimum block size (ISPEC=2) */
    int nx;               /* Crossover point (ISPEC=3) */
} lapack_tuning_t;

/*
 * Function declarations - always available
 */
int lapack_get_nb(const char* routine);
int lapack_get_nbmin(const char* routine);
int lapack_get_nx(const char* routine);
int lapack_get_mnthr(const char* routine, int m, int n);

/*
 * DGEQR/DGELQ block size functions
 *
 * These routines use separate MB (row block) and NB (column block) parameters.
 * From ilaenv.f lines 293-338:
 *   GEQR with N3=1: MB = min(M, 32768/N) for large matrices, else M
 *   GEQR with N3=2: NB = 1
 *   GELQ with N3=1: NB = 1
 *   GELQ with N3=2: NB = min(M, 32768/N) for large matrices, else M
 */
int lapack_get_geqr_mb(int m, int n);
int lapack_get_geqr_nb(int m, int n);
int lapack_get_gelq_mb(int m, int n);
int lapack_get_gelq_nb(int m, int n);

/*
 * TRSYL block size (ISPEC=1 for xTRSYL)
 *
 * Unlike other routines, TRSYL uses a dynamic block size that depends on the
 * problem dimensions and precision.  From ilaenv.f lines 477-485:
 *   Real:    NB = MIN(MAX(48, INT(MIN(N1,N2)*16/100)), 240)
 *   Complex: NB = MIN(MAX(24, INT(MIN(N1,N2)*8/100)),  80)
 */
int lapack_get_trsyl_nb(int n1, int n2, int is_complex);

/*
 * Block sizes that depend on a dimension or on the precision.
 * From ilaenv.f:
 *   xGBTRF, ILAENV(1, 'xGBTRF', ' ', M, N, KL, KU): NB = 1 if KU <= 64, else 32
 *   xPBTRF, ILAENV(1, 'xPBTRF', UPLO, N, KD, -1, -1): NB = 1 if KD <= 64, else 32
 *   xTZRZF, as queried by xGELSY: NB = 64 for real, 32 for complex
 */
int lapack_get_gbtrf_nb(int ku);
int lapack_get_pbtrf_nb(int kd);
int lapack_get_tzrzf_nb(int is_complex);

/*
 * ISPEC=5, the minimum column dimension, read by xTGSYL as its column block
 * size. ilaenv.f returns 2 for every routine.
 */
int lapack_get_nbcol(const char* routine);

/*
 * ISPEC=12..17 where the Fortran calls ILAENV for xHSEQR, xLAQR0, xLAQR3,
 * xLAQR4, xGGHD3 and xLAQZ0, and ISPEC=17..21 where ILAENV2STAGE calls
 * IPARAM2STAGE. The production versions return iparmq() and iparam2stage().
 * The test versions return the xlaenv value when one is set, as LAPACK's
 * TESTING/EIG/ilaenv.f does: ISPEC=12..16 from XLAENV and ISPEC=17 = NB.
 */
int lapack_get_iparmq(int ispec, const char* name, const char* opts,
                      int64_t n, int64_t ilo, int64_t ihi, int64_t lwork);
int lapack_get_iparam2stage(int ispec, const char* name, const char* opts,
                            int64_t ni, int64_t nbi, int64_t ibi, int64_t nxi);

#ifdef LAPACK_TUNING_TEST
/*
 * Test-only functions (equivalent to LAPACK's XLAENV)
 * These are only declared when LAPACK_TUNING_TEST is defined.
 */
void xlaenv(int ispec, int nvalue);
void xlaenv_reset(void);
#endif

#endif /* LAPACK_TUNING_H */


/*
 * =============================================================================
 * Implementation section - include in exactly ONE .c file
 * =============================================================================
 */
#if defined(LAPACK_TUNING_IMPLEMENTATION) && !defined(LAPACK_TUNING_IMPLEMENTATION_INCLUDED)
#define LAPACK_TUNING_IMPLEMENTATION_INCLUDED

#include "semicolon_lapack_auxiliary.h"

/*
 * Tuning table: what SRC/ilaenv.f returns for ISPEC=1, 2 and 3, one row per
 * name that a Fortran routine passes to ILAENV, without the precision prefix.
 * Callers pass that same name. A name without a row gets the sentinel, which
 * is ilaenv.f's default (NB=1, NBMIN=2, NX=0).
 */
static const lapack_tuning_t lapack_tuning_table[] = {
    /* Routine               NB  NBMIN   NX */
    {"GETRF",                64,     2,   0},
    {"GETRI",                64,     2,   0},
    {"GEQRF",                32,     2, 128},
    {"GEQLF",                32,     2, 128},
    {"GERQF",                32,     2, 128},
    {"GELQF",                32,     2, 128},
    {"GEHRD",                32,     2, 128},
    {"GEBRD",                32,     2, 128},
    /*
     * ilaenv.f compares SUBNAM(4:7) with 'QP3RK', which can never match,
     * so it returns 1, 2, 0 for xGEQP3RK. These are the values that branch
     * was written to return.
     */
    {"GEQP3RK",              32,     2, 128},
    {"GELST",                 1,     2,   0},
    {"POTRF",                64,     2,   0},
    {"SYTRF",                64,     8,   0},
    {"SYTRF_RK",             64,     8,   0},
    {"SYTRF_ROOK",           64,     8,   0},
    {"SYTRF_AA",             64,     8,   0},
    {"SYTRF_AA_2STAGE",     192,     8,   0},
    {"SYTRI2",                1,     2,   0},
    {"SYTRI_3",               1,     2,   0},
    {"SYTRD",                32,     2,  32},
    {"SYGST",                64,     2,   0},
    {"HETRF",                64,     2,   0},
    {"HETRF_RK",             64,     2,   0},
    {"HETRF_ROOK",           64,     2,   0},
    {"HETRF_AA",             64,     2,   0},
    {"HETRF_AA_2STAGE",     192,     2,   0},
    {"HETRI_3",               1,     2,   0},
    {"HETRD",                32,     2,  32},
    {"HEGST",                64,     2,   0},
    {"ORGQR",                32,     2, 128},
    {"ORGQL",                32,     2, 128},
    {"ORGRQ",                32,     2, 128},
    {"ORGLQ",                32,     2, 128},
    {"ORGHR",                32,     2, 128},
    {"ORMQR",                32,     2,   0},
    {"ORMQL",                32,     2,   0},
    {"ORMRQ",                32,     2,   0},
    {"ORMLQ",                32,     2,   0},
    {"ORMBR",                32,     2,   0},
    {"ORMTR",                32,     2,   0},
    {"ORMRZ",                64,     2,   0},
    {"UNGQR",                32,     2, 128},
    {"UNGQL",                32,     2, 128},
    {"UNGRQ",                32,     2, 128},
    {"UNGLQ",                32,     2, 128},
    {"UNGHR",                32,     2, 128},
    {"UNMQR",                32,     2,   0},
    {"UNMQL",                32,     2,   0},
    {"UNMRQ",                32,     2,   0},
    {"UNMLQ",                32,     2,   0},
    {"UNMBR",                32,     2,   0},
    {"UNMTR",                32,     2,   0},
    {"UNMRZ",                32,     2,   0},
    {"GGHD3",                32,     2, 128},
    {"TRTRI",                64,     2,   0},
    {"TREVC",                64,     2,   0},
    {"TGSYL",                 1,     2,   0},
    {"LAUUM",                64,     2,  64},
    {"LATRS",                32,     2,   0},
    {"LARFT",                 1,     2,  64},
    {"LAORHR_COL_GETRFNP",   32,     2,   0},
    {"STEBZ",                 1,     2,   0},
    {NULL,                    1,     2,   0}   /* Sentinel / default */
};

/* Helper: look up entry in tuning table */
static const lapack_tuning_t* lapack_tuning_lookup(const char* routine)
{
    for (int i = 0; lapack_tuning_table[i].routine != NULL; i++) {
        const char* a = lapack_tuning_table[i].routine;
        const char* b = routine;
        while (*a != '\0' && *a == *b) {
            a++;
            b++;
        }
        if (*a == *b) {
            return &lapack_tuning_table[i];
        }
    }
    return &lapack_tuning_table[sizeof(lapack_tuning_table)/sizeof(lapack_tuning_table[0]) - 1];
}

/* ISPEC=6 from ilaenv.f: INT( REAL( MIN( N1, N2 ) )*1.6E0 ) */
static int lapack_tuning_mnthr(int m, int n)
{
    int minmn = (m < n) ? m : n;
    return (int)((float)minmn * 1.6f);
}

static int lapack_tuning_gbtrf_nb(int ku)
{
    return (ku <= 64) ? 1 : 32;
}

static int lapack_tuning_pbtrf_nb(int kd)
{
    return (kd <= 64) ? 1 : 32;
}

static int lapack_tuning_tzrzf_nb(int is_complex)
{
    return is_complex ? 32 : 64;
}

#ifdef LAPACK_TUNING_TEST
/*
 * Test version: uses overridable parameters (like LAPACK's TESTING/LIN/ilaenv.f)
 *
 * IPARMS array (matching LAPACK's CLAENV common block):
 *   IPARMS[0] = NB      (ISPEC=1)
 *   IPARMS[1] = NBMIN   (ISPEC=2)
 *   IPARMS[2] = NX      (ISPEC=3)
 *   IPARMS[3] = unused  (ISPEC=4, number of shifts for xHSEQR)
 *   IPARMS[4] = NBCOL   (ISPEC=5, minimum column dimension, read by xTGSYL)
 *   IPARMS[5] = MNTHR override (ISPEC=6, if >= 0 use this instead of mult*min(m,n))
 *   ... up to IPARMS[8] for ISPEC=1..9
 *   IPARMS[11..15] = ISPEC=12..16 for lapack_get_iparmq (INMIN, INWIN,
 *                    INIBL, ISHFTS, IACC22)
 *
 * If IPARMS[i] < 0 (the initial state, restored by xlaenv_reset), fall back
 * to table lookup, so a test can set any value from 0 up, including NX = 0.
 * A negative value given to xlaenv restores the table value of that ISPEC.
 */
#define LAPACK_TUNING_NPARMS 16
static int iparms[LAPACK_TUNING_NPARMS] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

void xlaenv(int ispec, int nvalue)
{
    if (ispec >= 1 && ispec <= LAPACK_TUNING_NPARMS) {
        iparms[ispec - 1] = nvalue;
    }
}

void xlaenv_reset(void)
{
    for (int i = 0; i < LAPACK_TUNING_NPARMS; i++) {
        iparms[i] = -1;
    }
}

int lapack_get_nbcol(const char* routine)
{
    (void)routine;
    if (iparms[4] >= 0) {
        return iparms[4];
    }
    return 2;
}

int lapack_get_iparmq(int ispec, const char* name, const char* opts,
                      int64_t n, int64_t ilo, int64_t ihi, int64_t lwork)
{
    if (ispec >= 12 && ispec <= 16 && iparms[ispec - 1] >= 0) {
        return iparms[ispec - 1];
    }
    if (ispec == 17 && iparms[0] >= 0) {
        return iparms[0];
    }
    return (int)iparmq(ispec, name, opts, (INT)n, (INT)ilo, (INT)ihi, (INT)lwork);
}

int lapack_get_iparam2stage(int ispec, const char* name, const char* opts,
                            int64_t ni, int64_t nbi, int64_t ibi, int64_t nxi)
{
    if (ispec == 17 && iparms[0] >= 0) {
        return iparms[0];
    }
    return (int)iparam2stage(ispec, name, opts, (INT)ni, (INT)nbi, (INT)ibi, (INT)nxi);
}

int lapack_get_nb(const char* routine)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    return lapack_tuning_lookup(routine)->nb;
}

int lapack_get_nbmin(const char* routine)
{
    if (iparms[1] >= 0) {
        return iparms[1];
    }
    return lapack_tuning_lookup(routine)->nbmin;
}

int lapack_get_nx(const char* routine)
{
    if (iparms[2] >= 0) {
        return iparms[2];
    }
    return lapack_tuning_lookup(routine)->nx;
}

int lapack_get_mnthr(const char* routine, int m, int n)
{
    (void)routine;
    /* Test override: if IPARMS[5] >= 0, use it directly */
    if (iparms[5] >= 0) {
        return iparms[5];
    }
    return lapack_tuning_mnthr(m, n);
}

int lapack_get_gbtrf_nb(int ku)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    return lapack_tuning_gbtrf_nb(ku);
}

int lapack_get_pbtrf_nb(int kd)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    return lapack_tuning_pbtrf_nb(kd);
}

int lapack_get_tzrzf_nb(int is_complex)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    return lapack_tuning_tzrzf_nb(is_complex);
}

int lapack_get_geqr_mb(int m, int n)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    if ((m * n <= 131072) || (m <= 8192)) {
        return m;
    }
    return 32768 / n;
}

int lapack_get_geqr_nb(int m, int n)
{
    (void)m; (void)n;
    if (iparms[1] >= 0) {
        return iparms[1];
    }
    return 1;
}

int lapack_get_gelq_mb(int m, int n)
{
    (void)m; (void)n;
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    return 1;
}

int lapack_get_gelq_nb(int m, int n)
{
    if (iparms[1] >= 0) {
        return iparms[1];
    }
    if ((m * n <= 131072) || (m <= 8192)) {
        return m;
    }
    return 32768 / n;
}

int lapack_get_trsyl_nb(int n1, int n2, int is_complex)
{
    if (iparms[0] >= 0) {
        return iparms[0];
    }
    int min_mn = (n1 < n2) ? n1 : n2;
    if (is_complex) {
        /* ilaenv.f line 483: NB = MIN(MAX(24, INT(MIN(N1,N2)*8/100)), 80) */
        int nb = (min_mn * 8) / 100;
        if (nb < 24) nb = 24;
        if (nb > 80) nb = 80;
        return nb;
    } else {
        /* ilaenv.f line 480: NB = MIN(MAX(48, INT(MIN(N1,N2)*16/100)), 240) */
        int nb = (min_mn * 16) / 100;
        if (nb < 48) nb = 48;
        if (nb > 240) nb = 240;
        return nb;
    }
}

#else
/*
 * Production version: direct table lookup, no override mechanism
 *
 * These are marked as weak symbols so that the test version (which provides
 * strong symbols) takes precedence when both are linked. This allows tests
 * to override the block size via xlaenv() without requiring separate builds.
 *
 * GCC/Clang: __attribute__((weak))
 * MSVC: would need #pragma comment(linker, "/alternatename:...")
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define LAPACK_WEAK __attribute__((weak))
#else
#define LAPACK_WEAK
#endif

LAPACK_WEAK int lapack_get_nb(const char* routine)
{
    return lapack_tuning_lookup(routine)->nb;
}

LAPACK_WEAK int lapack_get_nbmin(const char* routine)
{
    return lapack_tuning_lookup(routine)->nbmin;
}

LAPACK_WEAK int lapack_get_nx(const char* routine)
{
    return lapack_tuning_lookup(routine)->nx;
}

LAPACK_WEAK int lapack_get_mnthr(const char* routine, int m, int n)
{
    (void)routine;
    return lapack_tuning_mnthr(m, n);
}

LAPACK_WEAK int lapack_get_gbtrf_nb(int ku)
{
    return lapack_tuning_gbtrf_nb(ku);
}

LAPACK_WEAK int lapack_get_pbtrf_nb(int kd)
{
    return lapack_tuning_pbtrf_nb(kd);
}

LAPACK_WEAK int lapack_get_tzrzf_nb(int is_complex)
{
    return lapack_tuning_tzrzf_nb(is_complex);
}

LAPACK_WEAK int lapack_get_geqr_mb(int m, int n)
{
    if ((m * n <= 131072) || (m <= 8192)) {
        return m;
    }
    return 32768 / n;
}

LAPACK_WEAK int lapack_get_geqr_nb(int m, int n)
{
    (void)m; (void)n;
    return 1;
}

LAPACK_WEAK int lapack_get_gelq_mb(int m, int n)
{
    (void)m; (void)n;
    return 1;
}

LAPACK_WEAK int lapack_get_gelq_nb(int m, int n)
{
    if ((m * n <= 131072) || (m <= 8192)) {
        return m;
    }
    return 32768 / n;
}

LAPACK_WEAK int lapack_get_trsyl_nb(int n1, int n2, int is_complex)
{
    int min_mn = (n1 < n2) ? n1 : n2;
    if (is_complex) {
        /* ilaenv.f line 483: NB = MIN(MAX(24, INT(MIN(N1,N2)*8/100)), 80) */
        int nb = (min_mn * 8) / 100;
        if (nb < 24) nb = 24;
        if (nb > 80) nb = 80;
        return nb;
    } else {
        /* ilaenv.f line 480: NB = MIN(MAX(48, INT(MIN(N1,N2)*16/100)), 240) */
        int nb = (min_mn * 16) / 100;
        if (nb < 48) nb = 48;
        if (nb > 240) nb = 240;
        return nb;
    }
}

LAPACK_WEAK int lapack_get_nbcol(const char* routine)
{
    (void)routine;
    return 2;
}

LAPACK_WEAK int lapack_get_iparmq(int ispec, const char* name, const char* opts,
                                  int64_t n, int64_t ilo, int64_t ihi, int64_t lwork)
{
    return (int)iparmq(ispec, name, opts, (INT)n, (INT)ilo, (INT)ihi, (INT)lwork);
}

LAPACK_WEAK int lapack_get_iparam2stage(int ispec, const char* name, const char* opts,
                                        int64_t ni, int64_t nbi, int64_t ibi, int64_t nxi)
{
    return (int)iparam2stage(ispec, name, opts, (INT)ni, (INT)nbi, (INT)ibi, (INT)nxi);
}

#undef LAPACK_WEAK

#endif /* LAPACK_TUNING_TEST */

#endif /* LAPACK_TUNING_IMPLEMENTATION */
