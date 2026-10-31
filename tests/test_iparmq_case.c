/**
 * @file test_iparmq_case.c
 * @brief IPARMQ regression test for the case of NAME.
 *
 * The reference IPARMQ converts NAME to upper case only when its first
 * character is lower case; a NAME that starts with an upper case letter is
 * compared as given. Each case calls IPARMQ with ISPEC = 16 (IACC22) for
 * NH = 10, 20 and 200 and checks the value the reference returns.
 */

#include "test_harness.h"

typedef struct {
    const char* name;
    INT expected[3];
} case_t;

static void test_iacc22_case(void** state)
{
    (void)state;
    static const case_t cases[] = {
        {"DHSEQR", {0, 0, 2}},
        {"dhseqr", {0, 0, 2}},
        {"dHSEQR", {0, 0, 2}},
        {"DhseqR", {0, 0, 0}},
        {"Dlaqr0", {0, 0, 0}},
        {"dlaqr0", {0, 0, 2}},
        {"zGGHRD", {1, 2, 2}},
        {"ZggHRD", {0, 0, 0}},
        {"zgghd3", {1, 2, 2}},
        {"DTREXC", {0, 2, 2}},
        {"dtrexc", {0, 2, 2}},
        {"Dtrexc", {0, 0, 0}},
        {"xyzexc", {0, 2, 2}},
        {"XYZexc", {0, 0, 0}},
    };
    const INT nhs[3] = {10, 20, 200};
    INT nbad = 0;

    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (INT j = 0; j < 3; j++) {
            INT got = iparmq(16, cases[k].name, "EV", nhs[j], 1, nhs[j], 1000);
            if (got != cases[k].expected[j]) {
                print_message("IPARMQ(16, '%s', NH=%d) = %d, expected %d\n",
                              cases[k].name, (int)nhs[j], (int)got,
                              (int)cases[k].expected[j]);
                nbad++;
            }
        }
    }
    assert_int_equal(nbad, 0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_iacc22_case),
    };
    (void)cmocka_run_group_tests_name("iparmq_case", tests, NULL, NULL);
    return 0;
}
