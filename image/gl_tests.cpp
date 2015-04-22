#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

using namespace LFL;

GTEST_API_ int main(int argc, const char **argv) {
    testing::InitGoogleTest(&argc, (char**)argv);
    LFL::FLAGS_default_font = LFL::FakeFontEngine::Filename();
    CHECK_EQ(LFL::app->Create(argc, argv, __FILE__), 0);
    return RUN_ALL_TESTS();
}

TEST(GLTest, Texture) {
}

