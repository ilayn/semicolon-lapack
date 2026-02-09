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
 *   ISPEC=1: NB      - Block size
 *   ISPEC=2: NBMIN   - Minimum block size for blocking to be used
 *   ISPEC=3: NX      - Crossover point (use unblocked if N < NX)
 *   ISPEC=6: MNTHR   - SVD crossover point = mnthr_mult * min(M,N)
 */

#ifndef LAPACK_TUNING_H
#define LAPACK_TUNING_H

#include <stddef.h>  /* for NULL */

typedef struct {
    const char* routine;  /* Routine name (uppercase, without precision prefix) */
    int nb;               /* Block size (ISPEC=1) */
    int nbmin;            /* Minimum block size (ISPEC=2) */
    int nx;               /* Crossover point (ISPEC=3) */
    double mnthr_mult;    /* SVD crossover multiplier (ISPEC=6): mnthr = mult * min(m,n) */
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
#ifdef LAPACK_TUNING_IMPLEMENTATION

/*
 * Tuning table - values based on LAPACK's ilaenv defaults
 *
 * The mnthr_mult field is only used by SVD routines (GESVD, GESDD, GELSS).
 * From ilaenv.f: ILAENV(6,...) = INT(MIN(N1,N2) * 1.6)
 * A value of 0.0 means ISPEC=6 is not applicable to this routine.
 */
static const lapack_tuning_t lapack_tuning_table[] = {
    /* Routine    NB  NBMIN  NX  MNTHR_MULT */
    {"GETRF",     64,    2,   0,  0.0},
    {"GBTRF",     64,    2,   0,  0.0},
    {"GETRS",     64,    2,   0,  0.0},
    {"GETRI",     64,    2,   0,  0.0},
    {"GEQRF",     32,    2, 128,  0.0},
    {"GEQLF",     32,    2, 128,  0.0},
    {"GERQF",     32,    2, 128,  0.0},
    {"GELQF",     32,    2, 128,  0.0},
    {"ORGQR",     32,    2, 128,  0.0},
    {"ORGLQ",     32,    2, 128,  0.0},
    {"ORGQL",     32,    2, 128,  0.0},
    {"ORGRQ",     32,    2, 128,  0.0},
    {"ORMQR",     32,    2, 128,  0.0},
    {"ORMLQ",     32,    2, 128,  0.0},
    {"ORMQL",     32,    2, 128,  0.0},
    {"ORMRQ",     32,    2, 128,  0.0},
    {"POTRF",     64,    2,   0,  0.0},
    {"LAUUM",     64,    2,   0,  0.0},
    {"SYTRF",     64,    2,   0,  0.0},
    {"SYGST",     64,    2,   0,  0.0},  /* Generalized eigenproblem reduction */
    {"HETRF",     64,    2,   0,  0.0},
    {"TRTRI",     64,    2,   0,  0.0},
    {"SYTRI2",    64,    2,   0,  0.0},
    {"GEBRD",     32,    2, 128,  0.0},
    {"SYTRD",     32,    2,   0,  0.0},
    {"HETRD",     32,    2,   0,  0.0},
    {"GEHRD",     32,    2, 128,  0.0},
    {"GGHRD",     32,    2, 128,  0.0},  /* GG routines */
    {"GGHD3",     32,    2, 128,  0.0},  /* Blocked variant */
    {"STEQR",      0,    0,   0,  0.0},  /* Not blocked */
    {"TREVC",     64,    2,   0,  0.0},
    {"GESVD",     32,    2,   0,  1.6},  /* SVD crossover: mnthr = 1.6 * min(m,n) */
    {"GESDD",     32,    2,   0,  1.6},  /* SVD crossover: mnthr = 1.6 * min(m,n) */
    {"GELSS",     32,    2,   0,  1.6},  /* SVD crossover: mnthr = 1.6 * min(m,n) */
    {NULL,         1,    2,   0,  0.0}   /* Sentinel / default */
};

/* Helper: look up entry in tuning table */
static const lapack_tuning_t* lapack_tuning_lookup(const char* routine)
{
    for (int i = 0; lapack_tuning_table[i].routine != NULL; i++) {
        if (lapack_tuning_table[i].routine[0] == routine[0] &&
            lapack_tuning_table[i].routine[1] == routine[1] &&
            lapack_tuning_table[i].routine[2] == routine[2] &&
            lapack_tuning_table[i].routine[3] == routine[3] &&
            lapack_tuning_table[i].routine[4] == routine[4]) {
            return &lapack_tuning_table[i];
        }
    }
    return &lapack_tuning_table[sizeof(lapack_tuning_table)/sizeof(lapack_tuning_table[0]) - 1];
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
 *   IPARMS[4] = unused  (ISPEC=5)
 *   IPARMS[5] = MNTHR override (ISPEC=6, if > 0 use this instead of mult*min(m,n))
 *   ... up to IPARMS[8] for ISPEC=1..9
 *
 * If IPARMS[i] == 0, fall back to table lookup.
 */
static int iparms[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

void xlaenv(int ispec, int nvalue)
{
    if (ispec >= 1 && ispec <= 9) {
        iparms[ispec - 1] = nvalue;
    }
}

void xlaenv_reset(void)
{
    for (int i = 0; i < 9; i++) {
        iparms[i] = 0;
    }
}

int lapack_get_nb(const char* routine)
{
    if (iparms[0] > 0) {
        return iparms[0];
    }
    return lapack_tuning_lookup(routine)->nb;
}

int lapack_get_nbmin(const char* routine)
{
    if (iparms[1] > 0) {
        return iparms[1];
    }
    return lapack_tuning_lookup(routine)->nbmin;
}

int lapack_get_nx(const char* routine)
{
    if (iparms[2] > 0) {
        return iparms[2];
    }
    return lapack_tuning_lookup(routine)->nx;
}

int lapack_get_mnthr(const char* routine, int m, int n)
{
    /* Test override: if IPARMS[5] > 0, use it directly */
    if (iparms[5] > 0) {
        return iparms[5];
    }
    /* Otherwise compute from multiplier */
    int minmn = (m < n) ? m : n;
    double mult = lapack_tuning_lookup(routine)->mnthr_mult;
    if (mult <= 0.0) {
        /* Routine doesn't use MNTHR, return minmn (no crossover benefit) */
        return minmn;
    }
    return (int)(mult * minmn);
}

int lapack_get_geqr_mb(int m, int n)
{
    if (iparms[0] > 0) {
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
    if (iparms[1] > 0) {
        return iparms[1];
    }
    return 1;
}

int lapack_get_gelq_mb(int m, int n)
{
    (void)m; (void)n;
    if (iparms[0] > 0) {
        return iparms[0];
    }
    return 1;
}

int lapack_get_gelq_nb(int m, int n)
{
    if (iparms[1] > 0) {
        return iparms[1];
    }
    if ((m * n <= 131072) || (m <= 8192)) {
        return m;
    }
    return 32768 / n;
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
#if defined(__GNUC__) || defined(__clang__)
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
    int minmn = (m < n) ? m : n;
    double mult = lapack_tuning_lookup(routine)->mnthr_mult;
    if (mult <= 0.0) {
        /* Routine doesn't use MNTHR, return minmn (no crossover benefit) */
        return minmn;
    }
    return (int)(mult * minmn);
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

#undef LAPACK_WEAK

#endif /* LAPACK_TUNING_TEST */

#endif /* LAPACK_TUNING_IMPLEMENTATION */
