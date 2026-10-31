/**
 * @file test_iparam2stage_case.c
 * @brief IPARAM2STAGE regression test for the case of NAME.
 *
 * The reference IPARAM2STAGE converts NAME to upper case only when its
 * first character is lower case, and then takes ALGO = NAME(4:6) and
 * STAG = NAME(8:12). With an upper case first character and a lower case
 * ALGO or STAG, no ISPEC = 20 workspace formula matches and LWORK = 1 is
 * returned. Each case compares ISPEC = 17, 18 and 20 with the value for the
 * same NAME in upper case, or with 1 where the reference returns 1.
 */

#include "test_harness.h"

typedef struct {
    const char* name;
    const char* upper;
    INT lwork_is_one;
} case_t;

static void test_name_case(void** state)
{
    (void)state;
    static const case_t cases[] = {
        {"dsytrd_2stage", "DSYTRD_2STAGE", 0},
        {"dSYTRD_2STAGE", "DSYTRD_2STAGE", 0},
        {"Dsytrd_2stage", "DSYTRD_2STAGE", 1},
        {"DSYTRD_2stage", "DSYTRD_2STAGE", 1},
        {"zhetrd_he2hb", "ZHETRD_HE2HB", 0},
        {"Zhetrd_he2hb", "ZHETRD_HE2HB", 1},
        {"dgebrd_2stage", "DGEBRD_2STAGE", 0},
        {"Dgebrd_2STAGE", "DGEBRD_2STAGE", 1},
        {"DGEbrd_GB2BD", "DGEBRD_GB2BD", 1},
        {"zhetrd_HB2ST", "ZHETRD_HB2ST", 0},
        {"ZHETRD_hb2st", "ZHETRD_HB2ST", 1},
        {"cGEBRD_GE2GB", "CGEBRD_GE2GB", 0},
    };
    const INT specs[3] = {17, 18, 20};
    INT nbad = 0;

    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (INT j = 0; j < 3; j++) {
            INT got = iparam2stage(specs[j], cases[k].name, "N", 100, 16, 8, -1);
            INT ref = iparam2stage(specs[j], cases[k].upper, "N", 100, 16, 8, -1);
            assert_true(ref > 1);
            INT expected = (specs[j] == 20 && cases[k].lwork_is_one) ? 1 : ref;
            if (got != expected) {
                print_message("IPARAM2STAGE(%d, '%s') = %d, expected %d\n",
                              (int)specs[j], cases[k].name, (int)got,
                              (int)expected);
                nbad++;
            }
        }
    }
    assert_int_equal(nbad, 0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_name_case),
    };
    (void)cmocka_run_group_tests_name("iparam2stage_case", tests, NULL, NULL);
    return 0;
}
