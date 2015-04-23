#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

using namespace LFL;

GTEST_API_ int main(int argc, const char **argv) {
    FLAGS_lfapp_video = true;
    testing::InitGoogleTest(&argc, (char**)argv);
    CHECK_EQ(0, LFL::app->Create(argc, argv, __FILE__));
    CHECK_EQ(0, LFL::app->Init());
    return RUN_ALL_TESTS();
}

TEST(GLTest, Texture) {
    static const int dim=32, psize=4, lsize=dim*psize, size=lsize*dim;
    unsigned char pdata[size];
    for (unsigned char *p = pdata, *e = pdata + size; p != e; ++p) *p = (p-pdata) % 256;
    Texture tex;
    tex.Resize(dim, dim, Pixel::RGBA, Texture::Flag::CreateGL);
    EXPECT_NE(0, tex.ID);

    tex.LoadBuffer(pdata, point(dim, dim), Pixel::RGBA, lsize);
    // PngWriter::Write("gl_tests_00.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));

    tex.UpdateGL(tex.buf, Box(dim, dim));
    tex.DumpGL(tex.ID);
    // PngWriter::Write("gl_tests_01.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));
}

