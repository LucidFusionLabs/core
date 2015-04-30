#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

namespace LFL {
int CompareTextureToBuffer(int tex_id, const unsigned char *buf, int h, int linesize, const char *fn, bool debug=0) {
    Texture test;
    test.DumpGL(tex_id);
    if (debug) {
        printf("CompareTextureToBuffer: %s\n%s\n", fn, test.HexDump().c_str());
        PngWriter::Write(fn, test);
    }
    CHECK_LE(h,        test.height);
    CHECK_LE(linesize, test.LineSize());
    for (int i=0, cmp; i<h; i++)
        if ((cmp = memcmp(buf + i*linesize, test.buf + i*test.LineSize(), linesize))) return cmp;
    return 0;
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

    Texture tex;
    tex.Resize(dim, dim, Texture::preferred_pf, Texture::Flag::CreateGL);
    EXPECT_NE(0, tex.ID);

    tex.LoadBuffer(pdata, point(dim, dim), Texture::preferred_pf, lsize);
    // PngWriter::Write("gl_tests_01.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));

    tex.UpdateGL(tex.buf, Box(dim, dim));
    EXPECT_EQ(0, CompareTextureToBuffer(tex.ID, pdata, dim, lsize, "gl_tests_02.png"));

    FrameBuffer fb;
    fb.Create(dim, dim, FrameBuffer::Flag::CreateTexture);
    screen->gd->EnableLayering();
    screen->gd->DisableBlend();
    screen->gd->Clear();
    tex.Bind();
    Box(dim, dim).Draw(tex.coord);
    EXPECT_EQ(0, CompareTextureToBuffer(fb.tex.ID, pdata, dim, lsize, "gl_tests_03.png"));

    screen->gd->Clear();
    tex.Bind();
    Box(dim, dim).DrawCrimped(tex.coord, 0);
    EXPECT_EQ(0, CompareTextureToBuffer(fb.tex.ID, pdata, dim, lsize, "gl_tests_04.png"));

    fb.Release();
    screen->gd->Clear();
    Box(dim, dim).Draw(tex.coord);
    tex.ScreenshotBox(Box(dim, dim), 0);
    // PngWriter::Write("gl_tests_05.png", tex);
    EXPECT_EQ(0, memcmp(pdata, tex.buf, size));
}

TEST(GLTest, Font) {
}

#ifdef __APPLE__
#import <CoreText/CTFont.h>
#import <CoreGraphics/CGBitmapContext.h>
TEST(GLTest, CoreText) {
    Font *font = Fonts::Get("coretext://Monaco", "", 12, Color::red, Color::blue);
    CHECK_NE(nullptr, font);
    CoreTextFontEngine::Resource *resource = static_cast<CoreTextFontEngine::Resource*>(font->resource.get());
    CHECK_NE(nullptr, resource);
    Glyph *g = font->FindGlyph('a');
    CHECK_NE(nullptr, font);
    Texture ref;
    // for (int i=0; i<=1000; i++) for (int j=0; j<=1000; j++) {
    for (int i=0, j=0; j <= 1000; j ? j++ : i++) { if (i > 1000) { i=0; j=1; }
        g->tex.RenewBuffer();
        CHECK_NE(nullptr, g->tex.buf);
        CGGlyph cg = g->internal.coretext.id;
        CGPoint point = CGPointMake(-g->bearing_x + i/1000.0, -RoundLower(g->internal.coretext.origin_y) + j/1000.0);
        CGContextRef context = g->tex.CGBitMap();
        CGContextSetRGBFillColor(context, font->bg.r(), font->bg.g(), font->bg.b(), font->bg.a());
        CGContextFillRect(context, CGRectMake(0, 0, g->tex.width, g->tex.height));
        CGContextSetRGBFillColor(context, font->fg.r(), font->fg.g(), font->fg.b(), font->fg.a());
        CGContextSetFont(context, resource->cgfont);
        CGContextSetFontSize(context, font->size);
        CGContextShowGlyphsAtPositions(context, &cg, &point, 1);
        CGContextRelease(context);
        g->tex.FlipBufferY();
        if (!ref.buf) {
            // PngWriter::Write("gl_tests_10.png", g->tex);
            // printf("reference:\n%s", g->tex.HexDump().c_str());
            ref.buf = g->tex.ReleaseBuffer();
        } else {
            int cmp = memcmp(ref.buf, g->tex.buf, g->tex.BufferSize());
            if (i < 333 && j < 999) EXPECT_EQ(0, cmp);
            else                    EXPECT_NE(0, cmp);
        }
    }
    FrameBuffer fb;
    fb.Create(g->tex.width, g->tex.height, FrameBuffer::Flag::CreateTexture);
    font->Draw(string(1, 'a'), Box(-g->bearing_x, g->tex.height - g->bearing_y, g->tex.width, font->ascender));
    EXPECT_EQ(0, CompareTextureToBuffer(fb.tex.ID, ref.buf, g->tex.height, g->tex.LineSize(), "gl_tests_11.png"));
}
#endif
