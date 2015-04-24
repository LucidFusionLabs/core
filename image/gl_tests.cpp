#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

namespace LFL {
int CompareTextureToBuffer(int tex_id, const unsigned char *buf, int dim, int size, const char *fn, bool debug=0) {
    Texture test;
    test.DumpGL(tex_id);
    if (debug) {
        printf("CompareTextureToBuffer: %s\n", fn);
        for (int i=0; i<dim; i++) printf("%s\n", Vec<unsigned char>::Str(test.buf + dim*i, dim, "%d").c_str());
        PngWriter::Write(fn, test);
    }
    return memcmp(buf, test.buf, size);
}

}; // namespace LFL
using namespace LFL;

GTEST_API_ int main(int argc, const char **argv) {
    FLAGS_lfapp_video = true;
    testing::InitGoogleTest(&argc, (char**)argv);
    CHECK_EQ(0, LFL::app->Create(argc, argv, __FILE__));
    CHECK_EQ(0, LFL::app->Init());
    return RUN_ALL_TESTS();
}

TEST(GLTest, Texture) {
    static const int dim=16, psize=4, lsize=dim*psize, size=lsize*dim;
    unsigned char pdata[size];
    for (unsigned char *p = pdata, *e = pdata + size; p != e; ++p) *p = (p-pdata) % 256;
    // printf("reference\n"); for (int i=0; i<dim; i++) printf("%s\n", Vec<unsigned char>::Str(pdata   + dim*i, dim, "%d").c_str());

    Texture tex;
    tex.Resize(dim, dim, Texture::preferred_pf, Texture::Flag::CreateGL);
    EXPECT_NE(0, tex.ID);

    tex.LoadBuffer(pdata, point(dim, dim), Texture::preferred_pf, lsize);
    // PngWriter::Write("gl_tests_01.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));

    tex.UpdateGL(tex.buf, Box(dim, dim));
    EXPECT_EQ(0, CompareTextureToBuffer(tex.ID, pdata, dim, size, "gl_tests_02.png"));

    FrameBuffer fb;
    fb.Create(dim, dim, FrameBuffer::Flag::CreateTexture);
    screen->gd->EnableLayering();
    screen->gd->DisableBlend();
    screen->gd->Clear();
    tex.Bind();
    Box(dim, dim).Draw(tex.coord);
    EXPECT_EQ(0, CompareTextureToBuffer(fb.tex.ID, pdata, dim, size, "gl_tests_03.png"));

    screen->gd->Clear();
    tex.Bind();
    Box(dim, dim).DrawCrimped(tex.coord, 0);
    EXPECT_EQ(0, CompareTextureToBuffer(fb.tex.ID, pdata, dim, size, "gl_tests_04.png"));

    fb.Release();
    screen->gd->Clear();
    Box(dim, dim).Draw(tex.coord);
    tex.ScreenshotBox(Box(dim, dim), 0);
    // PngWriter::Write("gl_tests_05.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));
}

TEST(GLTest, Font) {
}
