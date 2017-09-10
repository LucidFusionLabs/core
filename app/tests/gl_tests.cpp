#include "gtest/gtest.h"
#include "core/app/app.h"

namespace LFL {
int CompareTextureToBuffer(const Texture &test, const unsigned char *buf, int h, int linesize, const char *fn=0, bool debug=0) {
  if (debug && fn) {
    printf("CompareTextureToBuffer: %s\n%s\n", fn, test.HexDump().c_str());
    PngWriter::Write(fn, test);
  }
  if (!(h        <= test.height))     { FATAL(BlankNull(fn), ": CHECK(", h,        " <= ", test.height,     ")"); }
  if (!(linesize <= test.LineSize())) { FATAL(BlankNull(fn), ": CHECK(", linesize, " <= ", test.LineSize(), ")"); }
  for (int i=0, cmp; i<h; i++)
    if ((cmp = memcmp(buf + i*linesize, test.buf + i*test.LineSize(), linesize))) return cmp;
  return 0;
}

int CompareTextureToBuffer(GraphicsDeviceHolder *gd, int tex_id, const unsigned char *buf, int h, int linesize, const char *fn=0, bool debug=0) {
#ifdef LFL_MOBILE
  INFO("skipping ", BlankNull(fn));
  return 0;
#else
  Texture test(gd);
  gd->GD()->DumpTexture(&test, tex_id);
  return CompareTextureToBuffer(test, buf, h, linesize, fn, debug);
#endif
}

Application *app = nullptr;
bool tests_done = false;
int tests_result = -1;

extern "C" int MyAppFrame(Window *W, unsigned clicks, int flag) {
  if (!tests_done) {
    tests_result = RUN_ALL_TESTS();
    tests_done = true;
    app->Shutdown();
  }
  return 0;
}

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
  FLAGS_enable_video = FLAGS_enable_input = true;
  app = new Application(argc, argv);
  app->focused = CreateWindow(app);
  app->focused->frame_cb = MyAppFrame;
  testing::InitGoogleTest(&argc, const_cast<char**>(argv));
  return app;
}

extern "C" int MyAppMain() {
  CHECK_EQ(0, app->Create(__FILE__));
  CHECK_EQ(0, app->Init());
  app->focused->gd->have_npot_textures = false;
  app->StartNewWindow(app->focused);
  return app->Main();
}

}; // namespace LFL
using namespace LFL;

TEST(GLTest, Texture) {
  static const int dim=16, psize=4, lsize=dim*psize, size=lsize*dim;
  unsigned char pdata[size], zero_pdata[size];
  for (unsigned char *p = pdata, *e = pdata + size; p != e; ++p) *p = (p-pdata) % 256;
  memzero(zero_pdata);

  auto w = app->focused;
  Texture tex(w), tex2(w);
  tex.Resize(dim, dim, Texture::preferred_pf, Texture::Flag::CreateGL);
  EXPECT_NE(0, tex.ID);

  tex.LoadBuffer(pdata, point(dim, dim), Texture::preferred_pf, lsize);
  // PngWriter::Write("gl_tests_01.png", tex);
  EXPECT_EQ(0, memcmp(pdata, tex.buf, size));

  tex.UpdateGL(tex.buf, Box(dim, dim));
  EXPECT_EQ(0, CompareTextureToBuffer(w, tex.ID, pdata, dim, lsize, "gl_tests_02.png"));

  GraphicsContext gc(w->gd);
  FrameBuffer fb(w);
  fb.Create(dim, dim, FrameBuffer::Flag::CreateTexture);
  gc.gd->EnableLayering();
  gc.gd->DisableBlend();
  gc.gd->Clear();
  tex.Bind();
  gc.DrawTexturedBox(Box(dim, dim), tex.coord);
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, pdata, dim, lsize, "gl_tests_03.png"));

  gc.gd->Clear();
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, zero_pdata, dim, lsize));

  gc.gd->Clear();
  tex.Bind();
  gc.DrawCrimpedBox(Box(dim, dim), tex.coord, 0);
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, pdata, dim, lsize, "gl_tests_04.png"));

  gc.gd->Clear();
  fb.Release(false);
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, zero_pdata, dim, lsize));

  fb.Attach(0, 0, false);
  gc.gd->Clear();
  tex.Bind();
  gc.DrawTexturedBox(Box(dim, dim), tex.coord);
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, pdata, dim, lsize, "gl_tests_05.png"));
  fb.Release(false);

  gc.gd->Clear();
  tex.Bind();
  gc.DrawTexturedBox(Box(dim, dim), tex.coord);
  gc.gd->ScreenshotBox(&tex2, Box(dim, dim), 0);
  EXPECT_EQ(0, CompareTextureToBuffer(tex2, pdata, dim, lsize, "gl_tests_06.png"));
  fb.Release(true);
}

TEST(GLTest, Fonts) {
  auto w = app->focused;
  GraphicsContext gc(w->gd);
  Color c(0xffcdef12);
  Texture tex(w, 37, 93), tex2(w);
  tex.RenewBuffer();
  SimpleVideoResampler::Fill(tex.buf, tex.width, tex.height, tex.pf, tex.LineSize(), 0, 0, c);
  EXPECT_EQ(4, Pixel::Size(tex.pf));
  for (int y = 0; y < tex.height; y++) {
    for (int x = 0; x < tex.width; x++) {
      const unsigned char *b = tex.buf + y * tex.LineSize() + x * Pixel::Size(tex.pf);
      if (Texture::preferred_pf == Pixel::RGBA) {
        EXPECT_EQ(c.R(), b[0]);
        EXPECT_EQ(c.G(), b[1]);
        EXPECT_EQ(c.B(), b[2]);
        EXPECT_EQ(c.A(), b[3]);
      } else if (Texture::preferred_pf == Pixel::BGRA) {
        EXPECT_EQ(c.B(), b[0]);
        EXPECT_EQ(c.G(), b[1]);
        EXPECT_EQ(c.R(), b[2]);
        EXPECT_EQ(c.A(), b[3]);
      } else EXPECT_EQ(0, 1);
    }
  }

#if 0 
  gc.gd->Clear();
  gc.gd->FillColor(c);
  Box(tex.width, tex.height).Draw(gc.gd);
  gc.gd->SetColor(Color::white);
  gc.gd->ScreenshotBox(&tex2, Box(tex.width, tex.height), 0);
  EXPECT_EQ(0, CompareTextureToBuffer(tex2, tex.buf, tex.height, tex.LineSize(), "gl_tests_10.png"));
#endif

  gc.gd->Clear();
  app->fonts->GetFillColor(c)->Draw(&gc, Box(tex.width, tex.height));
  gc.gd->ScreenshotBox(&tex2, Box(tex.width, tex.height), 0);
  EXPECT_EQ(0, CompareTextureToBuffer(tex2, tex.buf, tex.height, tex.LineSize(), "gl_tests_11.png"));
}

#ifdef __APPLE__
#import <CoreText/CTFont.h>
#import <CoreGraphics/CGBitmapContext.h>
TEST(GLTest, CoreText) {
  auto w = app->focused;
  Color fg = Color(0xfd, 0x00, 0x00), bg = Color(0x00, 0x00, 0xfc);
  Font *font = app->fonts->Get(w->gl_h, "coretext://Monaco", "", 12, fg, bg);
  CHECK_NE(nullptr, font);
  CoreTextFontEngine::Resource *resource = static_cast<CoreTextFontEngine::Resource*>(font->resource.get());
  CHECK_NE(nullptr, resource);
  Glyph *g = font->FindGlyph('a');
  CHECK_NE(nullptr, font);
  Texture ref(w);
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
      // PngWriter::Write("gl_tests_12.png", g->tex);
      // printf("reference:\n%s", g->tex.HexDump().c_str());
      ref.buf = g->tex.ReleaseBuffer();
    } else {
      int cmp = memcmp(ref.buf, g->tex.buf, g->tex.BufferSize());
      if (i < 333 && j < 999) EXPECT_EQ(0, cmp);
      else                    EXPECT_NE(0, cmp);
    }
  }
  GraphicsContext gc(w->gd);
  FrameBuffer fb(w);
  fb.Create(g->tex.width, g->tex.height, FrameBuffer::Flag::CreateTexture);
  font->Draw(gc.gd, string(1, 'a'), Box(-g->bearing_x, g->tex.height - g->bearing_y, g->tex.width, font->ascender));
  EXPECT_EQ(0, CompareTextureToBuffer(w, fb.tex.ID, ref.buf, g->tex.height, g->tex.LineSize(), "gl_tests_11.png"));
}
#endif
