#include "gtest/gtest.h"
#include "template_example.h"

using namespace cp;

TEST(Template, CPU) {
    EXPECT_STREQ(afunc<target::CPU>().c_str(), "cpu");
}

TEST(Template, GPU) {
    EXPECT_STREQ(afunc<target::GPU>().c_str(), "gpu");
}

TEST(Template, Default) {
    EXPECT_STREQ(afunc().c_str(), "cpu");
}


TEST(IfConstexpr, CPU) {
    EXPECT_STREQ(anotherfunc<target::CPU>().c_str(), "cpu");
}

TEST(IfConstexpr, GPU) {
    EXPECT_STREQ(anotherfunc<target::GPU>().c_str(), "gpu");
}

TEST(IfConstexpr, Default) {
    EXPECT_STREQ(anotherfunc().c_str(), "cpu");
}
