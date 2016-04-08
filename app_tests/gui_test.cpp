/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gtest/gtest.h"
#include "core/app/gui.h"
#include "core/app/ipc.h"

namespace LFL {
struct LinesFrameBufferTest : public TextBox::LinesFrameBuffer {
  struct LineOp {
    point p; String16 text; int xo, wlo, wll;
    LineOp(TextBox::Line *L, int X, int O, int S) : p(L->p), text(L->Text16()), xo(X), wlo(O), wll(S) {}
    string DebugString() const { return StrCat("LineOp text(", text, ") ", xo, " ", wlo, " ", wll); }
  };
  struct PaintOp {
    String16 text; int lines; point p; Box b;
    PaintOp(TextBox::Line *L, const point &P, const Box &B) : text(L->Text16()), lines(L->Lines()), p(P), b(B) {}
    string DebugString() const { return StrCat("PaintOp text(", text, ") ", p, " ", b); }
  };
  static vector<LineOp> front, back;
  static vector<PaintOp> paint;
  LinesFrameBufferTest(GraphicsDevice *D) :
    TextBox::LinesFrameBuffer(D) { paint_cb = &LinesFrameBufferTest::PaintTest; }
  virtual void Clear(TextBox::Line *l)                              override { EXPECT_EQ(0, 1); }
  virtual void Update(TextBox::Line *l, int flag=0)                 override { EXPECT_EQ(0, 1); }
  virtual void Update(TextBox::Line *l, const point &p, int flag=0) override { EXPECT_EQ(0, 1); }
  virtual void PushFrontAndUpdateOffset(TextBox::Line *l, int lo)   override { EXPECT_EQ(0, 1); }
  virtual void PushBackAndUpdateOffset (TextBox::Line *l, int lo)   override { EXPECT_EQ(0, 1); }
  virtual int PushFrontAndUpdate(TextBox::Line *l, int xo=0, int wlo=0, int wll=0, int flag=0) override { int ret = TextBox::LinesFrameBuffer::PushFrontAndUpdate(l,xo,wlo,wll); front.emplace_back(l,xo,wlo,wll); return ret; }
  virtual int PushBackAndUpdate (TextBox::Line *l, int xo=0, int wlo=0, int wll=0, int flag=0) override { int ret = TextBox::LinesFrameBuffer::PushBackAndUpdate (l,xo,wlo,wll); back .emplace_back(l,xo,wlo,wll); return ret; }
  static point PaintTest(TextBox::Line *l, point lp, const Box &b)
  { l->Draw(lp + b.Position()); paint.emplace_back(l, lp, b); return point(lp.x, lp.y-b.h); }
};
vector<LinesFrameBufferTest::LineOp> LinesFrameBufferTest::front;
vector<LinesFrameBufferTest::LineOp> LinesFrameBufferTest::back;
vector<LinesFrameBufferTest::PaintOp> LinesFrameBufferTest::paint;

struct TextAreaTest : public TextArea {
  struct UpdateTokenOp {
    Line *l; String16 word; int type;
    UpdateTokenOp(Line *L, const String16 &W, int T) : l(L), word(W), type(T) {}
  };
  vector<UpdateTokenOp> token;
  LinesFrameBufferTest line_fb_test;
  TextAreaTest(GraphicsDevice *D, const FontRef &F, int S=200) : TextArea(D, F, S, 0), line_fb_test(D) {}
  virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
  virtual void UpdateToken(Line *l, int wo, int wl, int t, const TokenProcessor<DrawableBox>*) override {
    token.emplace_back(l, DrawableBoxRun(&l->data->glyphs[wo], wl).Text16(), t);
  }
};

struct EditorTest : public Editor {
  LinesFrameBufferTest line_fb_test;
  EditorTest(GraphicsDevice *D, const FontRef &F, File *I, bool Wrap=0) :
    Editor(D, F, I), line_fb_test(D) { line_fb.wrap=line_fb_test.wrap=Wrap; }
  virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
};

struct TerminalTest : public Terminal {
  struct UpdateTokenOp {
    Line *bl, *el; int bo, eo, type; string text;
    UpdateTokenOp(Line *BL, int BO, Line *EL, int EO, const string &X, int T) : bl(BL), el(EL), bo(BO), eo(EO), type(T), text(X) {}
  };
  vector<UpdateTokenOp> token;
  using Terminal::Terminal;
  void UpdateLongToken(Line *BL, int BO, Line *EL, int EO, const string &text, int T) {
    token.emplace_back(BL, BO, EL, EO, text, T);
  }
  void PrintLines(const char *header) {
    if (header) printf("%s\n", header);
    for (int i=1; i<=term_height; ++i) {
      Line *L = GetTermLine(i);
      printf("%s (%s)\n", String::ToUTF8(L->Text16()).c_str(), L->p.DebugString().c_str());
    }
  }
  void AssignLineTextToLineNumber() { for (int i=1; i<=term_height; ++i) Write(StrCat("\x1b[", i, "H", i, "\x1b[K")); }
};

TEST(GUITest, TextArea) { 
  {
    Font *font = app->fonts->Fake();
    TextAreaTest ta(screen->gd, font, 10);
    ta.line.InsertAt(-1)->AssignText("a");
    EXPECT_EQ(String16(u"a"), ta.line[-1].Text16());

    ta.line.InsertAt(-1)->AssignText("b");
    ta.line.InsertAt(-1)->AssignText("c");
    ta.line.InsertAt(-1)->AssignText("d");
    EXPECT_EQ(u"a", ta.line[-4].Text16()); EXPECT_EQ(-4, ta.line.IndexOf(&ta.line[-4]));
    EXPECT_EQ(u"b", ta.line[-3].Text16()); EXPECT_EQ(-3, ta.line.IndexOf(&ta.line[-3]));
    EXPECT_EQ(u"c", ta.line[-2].Text16()); EXPECT_EQ(-2, ta.line.IndexOf(&ta.line[-2]));
    EXPECT_EQ(u"d", ta.line[-1].Text16()); EXPECT_EQ(-1, ta.line.IndexOf(&ta.line[-1]));
    ta.line.InsertAt(-4)->AssignText("z");
    EXPECT_EQ(u"z", ta.line[-4].Text16());
    EXPECT_EQ(u"a", ta.line[-3].Text16());
    EXPECT_EQ(u"b", ta.line[-2].Text16());
    EXPECT_EQ(u"c", ta.line[-1].Text16());
    ta.line.InsertAt(-2)->AssignText("y");
    EXPECT_EQ(u"z", ta.line[-4].Text16());
    EXPECT_EQ(u"a", ta.line[-3].Text16());
    EXPECT_EQ(u"y", ta.line[-2].Text16());
    EXPECT_EQ(u"b", ta.line[-1].Text16());
    ta.line.PopBack(2);
    EXPECT_EQ(u"z", ta.line[-2].Text16());
    EXPECT_EQ(u"a", ta.line[-1].Text16());
    ta.line.PushFront()->AssignText("w");
    ta.line.PushFront()->AssignText("u");
    EXPECT_EQ(u"u", ta.line[-4].Text16());
    EXPECT_EQ(u"w", ta.line[-3].Text16());
    EXPECT_EQ(u"z", ta.line[-2].Text16());
    EXPECT_EQ(u"a", ta.line[-1].Text16());
    ta.line.PushFront()->AssignText("1");
    ta.line.PushFront()->AssignText("2");
    ta.line.PushFront()->AssignText("3");
    ta.line.PushFront()->AssignText("4");
    ta.line.PushFront()->AssignText("5");
    ta.line.PushFront()->AssignText("6");
    ta.line.PushFront()->AssignText("7");
    EXPECT_EQ(u"7", ta.line[-10].Text16()); EXPECT_EQ(-10, ta.line.IndexOf(&ta.line[-10]));
    EXPECT_EQ(u"6", ta.line[-9] .Text16()); EXPECT_EQ(-9,  ta.line.IndexOf(&ta.line[-9]));
    EXPECT_EQ(u"5", ta.line[-8] .Text16()); EXPECT_EQ(-8,  ta.line.IndexOf(&ta.line[-8]));
    EXPECT_EQ(u"4", ta.line[-7] .Text16()); EXPECT_EQ(-7,  ta.line.IndexOf(&ta.line[-7]));
    EXPECT_EQ(u"3", ta.line[-6] .Text16()); EXPECT_EQ(-6,  ta.line.IndexOf(&ta.line[-6]));
    EXPECT_EQ(u"2", ta.line[-5] .Text16()); EXPECT_EQ(-5,  ta.line.IndexOf(&ta.line[-5]));
    EXPECT_EQ(u"1", ta.line[-4] .Text16()); EXPECT_EQ(-4,  ta.line.IndexOf(&ta.line[-4]));
    EXPECT_EQ(u"u", ta.line[-3] .Text16()); EXPECT_EQ(-3,  ta.line.IndexOf(&ta.line[-3]));
    EXPECT_EQ(u"w", ta.line[-2] .Text16()); EXPECT_EQ(-2,  ta.line.IndexOf(&ta.line[-2]));
    EXPECT_EQ(u"z", ta.line[-1] .Text16()); EXPECT_EQ(-1,  ta.line.IndexOf(&ta.line[-1]));
  }

  for (int test_iter=0; test_iter<2; test_iter++) { 
    printf("TextAreaTest iter=%d reverse=%d\n", test_iter, test_iter);
    // 0: 122, 1: 222, 2: 223, 3: 234, 4: 344, 5: 445
    TextBox::Line *L;
    Font *font = app->fonts->Fake();
    TextAreaTest ta(screen->gd, font);
    ta.reverse_line_fb = test_iter;
    ta.line_fb.wrap = 1;
    (L = ta.line.PushFront())->AssignText("1");       L->Layout();
    (L = ta.line.PushFront())->AssignText("2\n2\n2"); L->Layout();
    (L = ta.line.PushFront())->AssignText("3");       L->Layout();
    (L = ta.line.PushFront())->AssignText("4\n4");    L->Layout();
    (L = ta.line.PushFront())->AssignText("5");       L->Layout();
    EXPECT_EQ(1, ta.line[-1].Lines());
    EXPECT_EQ(3, ta.line[-2].Lines());
    EXPECT_EQ(1, ta.line[-3].Lines());
    EXPECT_EQ(2, ta.line[-4].Lines());
    EXPECT_EQ(1, ta.line[-5].Lines());

    bool reverse = ta.reverse_line_fb;
    LinesFrameBufferTest *test_fb = &ta.line_fb_test;
    auto &test_fb_front = reverse ? test_fb->back  : test_fb->front;
    auto &test_fb_back  = reverse ? test_fb->front : test_fb->back;
    int fh = font->Height();
    Box b(128, 3*fh);
    ta.Draw(b, TextArea::DrawFlag::CheckResized);
    int w = test_fb->w;

    // 0: 122 : 1=1:0, 2=2:2, 3=2:1
    EXPECT_EQ(b.w, test_fb->w);          EXPECT_EQ(b.h, test_fb->h);
    EXPECT_EQ(0, ta.start_line);         EXPECT_EQ(1, ta.end_line);
    EXPECT_EQ(0, ta.start_line_adjust);  EXPECT_EQ(1, ta.start_line_cutoff);
    EXPECT_EQ(2, ta.end_line_adjust);    EXPECT_EQ(1, ta.end_line_cutoff);

    EXPECT_EQ(2, test_fb_front.size());
    EXPECT_NEAR(0, test_fb->scroll.y, 1e-5);
    EXPECT_EQ(point(0,0), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"1",          test_fb_front[0].text);
      EXPECT_EQ(0,             test_fb_front[0].wlo);
      EXPECT_EQ(3,             test_fb_front[0].wll);
      if (reverse) EXPECT_EQ(point(0,3*fh), test_fb_front[0].p);
      else         EXPECT_EQ(point(0,fh),   test_fb_front[0].p);
    }
    if (test_fb_front.size() > 1) {
      EXPECT_EQ(u"2\n2\n2",    test_fb_front[1].text);
      EXPECT_EQ(0,             test_fb_front[1].wlo);
      EXPECT_EQ(2,             test_fb_front[1].wll);
      if (reverse) EXPECT_EQ(point(0,2*fh), test_fb_front[1].p);
      else         EXPECT_EQ(point(0,4*fh), test_fb_front[1].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"1",             test_fb->paint[0].text);
      EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[0].b);
      if (reverse) EXPECT_EQ(point(0,3*fh),    test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,fh),      test_fb->paint[0].p);
    }
    if (test_fb->paint.size() > 1) {
      EXPECT_EQ(u"2\n2\n2",       test_fb->paint[1].text);
      if (reverse) EXPECT_EQ(point(0,2*fh),    test_fb->paint[1].p);
      else         EXPECT_EQ(point(0,3*fh),    test_fb->paint[1].p);
      if (reverse) EXPECT_EQ(Box(0,0, w,2*fh), test_fb->paint[1].b);
      else         EXPECT_EQ(Box(0,fh,w,2*fh), test_fb->paint[1].b);
    }
    test_fb->paint.clear();

    // 1: 222 : 2=2:2, 3=2:1, 1=2:0
    ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ(1, ta.start_line);        EXPECT_EQ(1, ta.end_line);
    EXPECT_EQ(0, ta.start_line_adjust); EXPECT_EQ(3, ta.start_line_cutoff);
    EXPECT_EQ(3, ta.end_line_adjust);   EXPECT_EQ(0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_front.size());
    EXPECT_NEAR(-1/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,2*fh), test_fb->p);
    else         EXPECT_EQ(point(0,  fh), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",  test_fb_front[0].text);
      EXPECT_EQ(2,           test_fb_front[0].wlo);
      EXPECT_EQ(1,           test_fb_front[0].wll);
      if (reverse) EXPECT_EQ(point(0,5*fh), test_fb_front[0].p);
      else         EXPECT_EQ(point(0,  fh), test_fb_front[0].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",       test_fb->paint[0].text);
      if (reverse) EXPECT_EQ(point(0,3*fh),       test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,  fh),       test_fb->paint[0].p);
      if (reverse) EXPECT_EQ(Box(0,2*fh,w,fh),    test_fb->paint[0].b);
      else         EXPECT_EQ(Box(0,0,   w,fh),    test_fb->paint[0].b);
    }
    test_fb->paint.clear();

    // 2: 223 : 3=2:1, 1=2:0, 2=3:0
    ta.v_scrolled = 2.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ( 1, ta.start_line);        EXPECT_EQ( 2, ta.end_line);  
    EXPECT_EQ(-1, ta.start_line_adjust); EXPECT_EQ( 2, ta.start_line_cutoff);
    EXPECT_EQ( 1, ta.end_line_adjust);   EXPECT_EQ( 0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_front.size());
    EXPECT_NEAR(-2/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,  fh), test_fb->p);
    else         EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"3",          test_fb_front[0].text);
      EXPECT_EQ(0,             test_fb_front[0].wlo);
      EXPECT_EQ(1,             test_fb_front[0].wll);
      EXPECT_EQ(point(0,fh*2), test_fb_front[0].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"3",          test_fb->paint[0].text);
      EXPECT_EQ(point(0,fh*2), test_fb->paint[0].p);
      EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    }
    test_fb->paint.clear();

    // 1: 222 : 2=2:2, 3=2:1, 1=2:0
    ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ(1, ta.start_line);         EXPECT_EQ(1, ta.end_line);   
    EXPECT_EQ(0, ta.start_line_adjust);  EXPECT_EQ(3, ta.start_line_cutoff);
    EXPECT_EQ(3, ta.end_line_adjust);    EXPECT_EQ(0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_back.size());
    EXPECT_NEAR(-1/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,2*fh), test_fb->p);
    else         EXPECT_EQ(point(0,fh),   test_fb->p);
    if (test_fb_back.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",    test_fb_back[0].text);
      EXPECT_EQ(2,             test_fb_back[0].wlo);
      EXPECT_EQ(1,             test_fb_back[0].wll);
      if (reverse) EXPECT_EQ(point(0,fh*2), test_fb_back[0].p);
      else         EXPECT_EQ(point(0,fh*4), test_fb_back[0].p);
    }
    test_fb_back.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",       test_fb->paint[0].text);
      EXPECT_EQ(point(0,fh*2),    test_fb->paint[0].p);
      if (reverse) EXPECT_EQ(Box(0,0,   w,fh), test_fb->paint[0].b);
      else         EXPECT_EQ(Box(0,fh*2,w,fh), test_fb->paint[0].b);
    }
    test_fb->paint.clear();

    // 0: 122 : 1=1:0, 2=2:2, 3=2:1
    ta.v_scrolled = 0;
    ta.UpdateScrolled();
    EXPECT_EQ(0, ta.start_line);         EXPECT_EQ(1, ta.end_line);   
    EXPECT_EQ(0, ta.start_line_adjust);  EXPECT_EQ(1, ta.start_line_cutoff);
    EXPECT_EQ(2, ta.end_line_adjust);    EXPECT_EQ(1, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_back.size());
    EXPECT_NEAR(0, test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,3*fh), test_fb->p);
    else         EXPECT_EQ(point(0,0),    test_fb->p);
    if (test_fb_back.size() > 0) {
      EXPECT_EQ(u"1",        test_fb_back[0].text);
      EXPECT_EQ(0,           test_fb_back[0].wlo);
      EXPECT_EQ(1,           test_fb_back[0].wll);
      if (reverse) EXPECT_EQ(point(0,3*fh), test_fb_back[0].p);
      else         EXPECT_EQ(point(0,fh),   test_fb_back[0].p);
    }
    test_fb_back.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"1",          test_fb->paint[0].text);
      EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
      if (reverse) EXPECT_EQ(point(0,3*fh), test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,fh),   test_fb->paint[0].p);
    }
    test_fb->paint.clear();

    // 2: 223 : 3=2:1, 1=2:0, 2=3:0
    ta.v_scrolled = 2.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ( 1, ta.start_line);         EXPECT_EQ( 2, ta.end_line);   
    EXPECT_EQ(-1, ta.start_line_adjust);  EXPECT_EQ( 2, ta.start_line_cutoff);
    EXPECT_EQ( 1, ta.end_line_adjust);    EXPECT_EQ( 0, ta.end_line_cutoff);

    EXPECT_EQ(2, test_fb_front.size());
    EXPECT_NEAR(-2/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,  fh), test_fb->p);
    else         EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",    test_fb_front[0].text);
      EXPECT_EQ(2,             test_fb_front[0].wlo);
      EXPECT_EQ(1,             test_fb_front[0].wll);
      if (reverse) EXPECT_EQ(point(0,5*fh),   test_fb_front[0].p);
      else         EXPECT_EQ(point(0,fh),     test_fb_front[0].p);
    }
    if (test_fb_front.size() > 1) {
      EXPECT_EQ(u"3",          test_fb_front[1].text);
      EXPECT_EQ(0,             test_fb_front[1].wlo);
      EXPECT_EQ(1,             test_fb_front[1].wll);
      EXPECT_EQ(point(0,fh*2), test_fb_front[1].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",    test_fb->paint[0].text);
      if (reverse) EXPECT_EQ(point(0,3*fh),   test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,fh),     test_fb->paint[0].p);
      if (reverse) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[0].b);
      else         EXPECT_EQ(Box(0,0,   w,fh), test_fb->paint[0].b);
    }
    if (test_fb->paint.size() > 1) {
      EXPECT_EQ(u"3",          test_fb->paint[1].text);
      EXPECT_EQ(point(0,2*fh), test_fb->paint[1].p);
      EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[1].b);
    }
    test_fb->paint.clear();

    // 4: 344 : 2=3:0, 3=4:1, 1=4:0
    ta.v_scrolled = 4.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ( 2, ta.start_line);         EXPECT_EQ( 3, ta.end_line);   
    EXPECT_EQ( 0, ta.start_line_adjust);  EXPECT_EQ( 1, ta.start_line_cutoff);
    EXPECT_EQ( 2, ta.end_line_adjust);    EXPECT_EQ( 0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_front.size());
    EXPECT_NEAR(-1/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,2*fh), test_fb->p);
    else         EXPECT_EQ(point(0,  fh), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"4\n4",     test_fb_front[0].text);
      EXPECT_EQ(0,           test_fb_front[0].wlo);
      EXPECT_EQ(2,           test_fb_front[0].wll);
      if (reverse) EXPECT_EQ(point(0,4*fh), test_fb_front[0].p);
      else         EXPECT_EQ(point(0,fh),   test_fb_front[0].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"4\n4",         test_fb->paint[0].text);
      EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
      if (reverse) EXPECT_EQ(point(0,  fh),   test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,4*fh),   test_fb->paint[0].p);
    }
    if (test_fb->paint.size() > 1) {
      EXPECT_EQ(u"4\n4",         test_fb->paint[1].text);
      EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
      if (reverse) EXPECT_EQ(point(0,4*fh),     test_fb->paint[1].p);
      else         EXPECT_EQ(point(0,  fh),     test_fb->paint[1].p);
    }
    test_fb->paint.clear();

    // 5: 445 : 3=4:1, 1=4:0, 2=5:0
    ta.v_scrolled = 5.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ( 3, ta.start_line);         EXPECT_EQ( 4, ta.end_line);   
    EXPECT_EQ( 0, ta.start_line_adjust);  EXPECT_EQ( 2, ta.start_line_cutoff);
    EXPECT_EQ( 1, ta.end_line_adjust);    EXPECT_EQ( 0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_front.size());
    EXPECT_NEAR(-2/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,1*fh), test_fb->p);
    else         EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb_front.size() > 0) {
      EXPECT_EQ(u"5",          test_fb_front[0].text);
      EXPECT_EQ(0,             test_fb_front[0].wlo);
      EXPECT_EQ(1,             test_fb_front[0].wll);
      EXPECT_EQ(point(0,2*fh), test_fb_front[0].p);
    }
    test_fb_front.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"5",          test_fb->paint[0].text);
      EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
      EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    }
    test_fb->paint.clear();

    // 3: 234 : 1=2:0, 2=3:0, 3=4:1
    ta.v_scrolled = 3.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ( 1, ta.start_line);         EXPECT_EQ( 3, ta.end_line);   
    EXPECT_EQ(-2, ta.start_line_adjust);  EXPECT_EQ( 1, ta.start_line_cutoff);
    EXPECT_EQ( 1, ta.end_line_adjust);    EXPECT_EQ( 1, ta.end_line_cutoff);

    EXPECT_EQ(2, test_fb_back.size());
    EXPECT_NEAR(0, test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,3*fh), test_fb->p);
    else         EXPECT_EQ(point(0,0),    test_fb->p);
    if (test_fb_back.size() > 0) {
      EXPECT_EQ(u"3",          test_fb_back[0].text);
      EXPECT_EQ(0,             test_fb_back[0].wlo);
      EXPECT_EQ(2,             test_fb_back[0].wll);
      EXPECT_EQ(point(0,2*fh), test_fb_back[0].p);
    }
    if (test_fb_back.size() > 1) {
      EXPECT_EQ(u"2\n2\n2",    test_fb_back[1].text);
      EXPECT_EQ(0,             test_fb_back[1].wlo);
      EXPECT_EQ(1,             test_fb_back[1].wll);
      if (reverse) EXPECT_EQ(point(0,5*fh),   test_fb_back[1].p);
      else         EXPECT_EQ(point(0,  fh),   test_fb_back[1].p);
    }
    test_fb_back.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"3",          test_fb->paint[0].text);
      EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
      EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    }
    if (test_fb->paint.size() > 1) {
      EXPECT_EQ(u"2\n2\n2",    test_fb->paint[1].text);
      if (reverse) EXPECT_EQ(point(0,3*fh),    test_fb->paint[1].p);
      else         EXPECT_EQ(point(0,  fh),    test_fb->paint[1].p);
      if (reverse) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[1].b);
      else         EXPECT_EQ(Box(0,0,   w,fh), test_fb->paint[1].b);
    }
    test_fb->paint.clear();

    // 1: 222 : 2=2:2, 3=2:1, 1=2:0
    ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
    ta.UpdateScrolled();
    EXPECT_EQ(1, ta.start_line);         EXPECT_EQ(1, ta.end_line);   
    EXPECT_EQ(0, ta.start_line_adjust);  EXPECT_EQ(3, ta.start_line_cutoff);
    EXPECT_EQ(3, ta.end_line_adjust);    EXPECT_EQ(0, ta.end_line_cutoff);

    EXPECT_EQ(1, test_fb_back.size());
    EXPECT_NEAR(2/3.0*(reverse?-1:1), test_fb->scroll.y, 1e-5);
    if (reverse) EXPECT_EQ(point(0,2*fh), test_fb->p);
    else         EXPECT_EQ(point(0,fh),   test_fb->p);
    if (test_fb_back.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",    test_fb_back[0].text);
      EXPECT_EQ(1,             test_fb_back[0].wlo);
      EXPECT_EQ(2,             test_fb_back[0].wll);
      if (reverse) EXPECT_EQ(point(0,fh*2), test_fb_back[0].p);
      else         EXPECT_EQ(point(0,fh*4), test_fb_back[0].p);
    }
    test_fb_back.clear();
    if (reverse) EXPECT_EQ(2, test_fb->paint.size());
    else         EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) {
      EXPECT_EQ(u"2\n2\n2",       test_fb->paint[0].text);
      if (reverse) EXPECT_EQ(point(0,fh*5),    test_fb->paint[0].p);
      else         EXPECT_EQ(point(0,fh*3),    test_fb->paint[0].p);
      if (reverse) EXPECT_EQ(Box(0,0, w,fh*2), test_fb->paint[0].b);
      else         EXPECT_EQ(Box(0,fh,w,fh*2), test_fb->paint[0].b);
    }
    if (test_fb->paint.size() > 1) {
      EXPECT_EQ(u"2\n2\n2",        test_fb->paint[1].text);
      EXPECT_EQ(point(0,2*fh),   test_fb->paint[1].p);
      EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
    }
    test_fb->paint.clear();
  }
}

TEST(GUITest, Editor) {
  Font *font = app->fonts->Fake();
  int fh = font->Height(), w = font->fixed_width;
  EXPECT_NE(0, fh); EXPECT_NE(0, w);
  EditorTest e(screen->gd, font, new BufferFile(string("1\n2 2 2\n3\n4 4\n5\n")), true);
  LinesFrameBufferTest *test_fb = &e.line_fb_test;
  Box b(w, 3*fh);
  e.Draw(b, TextArea::DrawFlag::CheckResized);

  // 0: 122 : 3=1:0, 2=2:0, 1=2:1
  EXPECT_EQ(0, e.start_line_adjust); EXPECT_EQ(1, e.start_line_cutoff);
  EXPECT_EQ(2, e.end_line_adjust);   EXPECT_EQ(1, e.end_line_cutoff);
  EXPECT_EQ(2, e.line.Size());
  EXPECT_EQ(4, e.fb_wrapped_lines);
  EXPECT_EQ(2, test_fb->back.size());
  if (test_fb->back.size() > 0) EXPECT_EQ(u"1",          test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(3,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->back[0].p);
  if (test_fb->back.size() > 1) EXPECT_EQ(u"2 2 2",      test_fb->back[1].text);
  if (test_fb->back.size() > 1) EXPECT_EQ(0,             test_fb->back[1].wlo);
  if (test_fb->back.size() > 1) EXPECT_EQ(2,             test_fb->back[1].wll);
  if (test_fb->back.size() > 1) EXPECT_EQ(point(0,2*fh), test_fb->back[1].p);
  test_fb->back.clear();
  EXPECT_EQ(2, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"1",            test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),   test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),   test_fb->paint[0].b);
  if (test_fb->paint.size() > 1) EXPECT_EQ(u"2 2 2",        test_fb->paint[1].text);
  if (test_fb->paint.size() > 1) EXPECT_EQ(3,               test_fb->paint[1].lines);
  if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),   test_fb->paint[1].p);
  if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
  test_fb->paint.clear();

  // 1: 222 : 2=2:0, 1=2:1, 3=2:2
  e.v_scrolled = 1.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(0, e.start_line_adjust);  EXPECT_EQ(3, e.start_line_cutoff);
  EXPECT_EQ(3, e.end_line_adjust);    EXPECT_EQ(0, e.end_line_cutoff);
  EXPECT_EQ(1, e.line.Size());
  EXPECT_EQ(3, e.fb_wrapped_lines);
  EXPECT_EQ(1, test_fb->back.size());
  EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,2*fh), test_fb->p);
  if (test_fb->back.size() > 0) EXPECT_EQ(u"2 2 2",      test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,5*fh), test_fb->back[0].p);
  test_fb->back.clear();
  EXPECT_EQ(1, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"2 2 2",         test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),    test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[0].b);
  test_fb->paint.clear();

  // 2: 223 : 1=2:1, 3=2:2, 2=3:0
  e.v_scrolled = 2.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(-1, e.start_line_adjust);  EXPECT_EQ( 2, e.start_line_cutoff);
  EXPECT_EQ( 1, e.end_line_adjust);    EXPECT_EQ( 0, e.end_line_cutoff);
  EXPECT_EQ( 2, e.line.Size());
  EXPECT_EQ( 4, e.fb_wrapped_lines);
  EXPECT_EQ( 1, test_fb->back.size());
  EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,fh), test_fb->p);
  if (test_fb->back.size() > 0) EXPECT_EQ(u"3",          test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->back[0].p);
  test_fb->back.clear();
  EXPECT_EQ(1, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"3",          test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
  test_fb->paint.clear();

  // 1: 222 : 2=2:0, 1=2:1, 3=2:2
  e.v_scrolled = 1.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(0, e.start_line_adjust);  EXPECT_EQ(3, e.start_line_cutoff);
  EXPECT_EQ(3, e.end_line_adjust);    EXPECT_EQ(0, e.end_line_cutoff);
  EXPECT_EQ(1, e.line.Size());
  EXPECT_EQ(3, e.fb_wrapped_lines);
  EXPECT_EQ(1, test_fb->front.size());
  EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,2*fh), test_fb->p);
  if (test_fb->front.size() > 0) EXPECT_EQ(u"2 2 2",      test_fb->front[0].text);
  if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wlo);
  if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
  if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
  test_fb->front.clear();
  EXPECT_EQ(1, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"2 2 2",      test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
  test_fb->paint.clear();

  // 0: 122 : 3=1:0, 2=2:0, 1=2:1
  e.v_scrolled = 0;
  e.UpdateScrolled();
  EXPECT_EQ(0, e.start_line_adjust);  EXPECT_EQ(1, e.start_line_cutoff);
  EXPECT_EQ(2, e.end_line_adjust);    EXPECT_EQ(1, e.end_line_cutoff);
  EXPECT_EQ(2, e.line.Size());
  EXPECT_EQ(4, e.fb_wrapped_lines);
  EXPECT_EQ(1, test_fb->front.size());
  EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,3*fh), test_fb->p);
  if (test_fb->front.size() > 0) EXPECT_EQ(u"1",          test_fb->front[0].text);
  if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
  if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
  if (test_fb->front.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->front[0].p);
  test_fb->front.clear();
  EXPECT_EQ(1, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"1",          test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
  test_fb->paint.clear();

  // 2: 223 : 1=2:1, 3=2:2, 2=3:0
  e.v_scrolled = 2.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(-1, e.start_line_adjust);  EXPECT_EQ( 2, e.start_line_cutoff);
  EXPECT_EQ( 1, e.end_line_adjust);    EXPECT_EQ( 0, e.end_line_cutoff);
  EXPECT_EQ( 2, e.line.Size());
  EXPECT_EQ( 4, e.fb_wrapped_lines);
  EXPECT_EQ( 2, test_fb->back.size());
  EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,fh), test_fb->p);
  if (test_fb->back.size() > 0) EXPECT_EQ(u"2 2 2",      test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,5*fh), test_fb->back[0].p);
  if (test_fb->back.size() > 1) EXPECT_EQ(u"3",          test_fb->back[1].text);
  if (test_fb->back.size() > 1) EXPECT_EQ(0,             test_fb->back[1].wlo);
  if (test_fb->back.size() > 1) EXPECT_EQ(1,             test_fb->back[1].wll);
  if (test_fb->back.size() > 1) EXPECT_EQ(point(0,2*fh), test_fb->back[1].p);
  test_fb->back.clear();
  EXPECT_EQ(2, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"2 2 2",         test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),    test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[0].b);
  if (test_fb->paint.size() > 1) EXPECT_EQ(u"3",             test_fb->paint[1].text);
  if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),    test_fb->paint[1].p);
  if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[1].b);
  test_fb->paint.clear();

  // 4: 344 : 2=3:0, 1=4:0, 3=4:1
  e.v_scrolled = 4.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ( 0, e.start_line_adjust);  EXPECT_EQ( 1, e.start_line_cutoff);
  EXPECT_EQ( 2, e.end_line_adjust);    EXPECT_EQ( 0, e.end_line_cutoff);
  EXPECT_EQ( 2, e.line.Size());
  EXPECT_EQ( 3, e.fb_wrapped_lines);
  EXPECT_EQ( 1, test_fb->back.size());
  EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,2*fh), test_fb->p);
  if (test_fb->back.size() > 0) EXPECT_EQ(u"4 4",        test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,4*fh), test_fb->back[0].p);
  test_fb->back.clear();
  EXPECT_EQ(2, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"4 4",          test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),     test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
  if (test_fb->paint.size() > 1) EXPECT_EQ(u"4 4",          test_fb->paint[1].text);
  if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,4*fh),   test_fb->paint[1].p);
  if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
  test_fb->paint.clear();

  // 5: 445 : 1=4:0, 3=4:1, 2=5:0
  e.v_scrolled = 5.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ( 0, e.start_line_adjust);  EXPECT_EQ( 2, e.start_line_cutoff);
  EXPECT_EQ( 1, e.end_line_adjust);    EXPECT_EQ( 0, e.end_line_cutoff);
  EXPECT_EQ( 2, e.line.Size());
  EXPECT_EQ( 3, e.fb_wrapped_lines);
  EXPECT_EQ( 1, test_fb->back.size());
  EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,1*fh), test_fb->p);
  if (test_fb->back.size() > 0) EXPECT_EQ(u"5",          test_fb->back[0].text);
  if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
  if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
  if (test_fb->back.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->back[0].p);
  test_fb->back.clear();
  EXPECT_EQ(1, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"5",          test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
  test_fb->paint.clear();

  // 3: 234 : 3=2:2, 2=3:0, 1=4:0
  e.v_scrolled = 3.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(-2, e.start_line_adjust);  EXPECT_EQ( 1, e.start_line_cutoff);
  EXPECT_EQ( 1, e.end_line_adjust);    EXPECT_EQ( 1, e.end_line_cutoff);
  EXPECT_EQ( 3, e.line.Size());
  EXPECT_EQ( 6, e.fb_wrapped_lines);
  EXPECT_EQ( 2, test_fb->front.size());
  EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,3*fh), test_fb->p);
  if (test_fb->front.size() > 0) EXPECT_EQ(u"3",          test_fb->front[0].text);
  if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
  if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wll);
  if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
  if (test_fb->front.size() > 1) EXPECT_EQ(u"2 2 2",      test_fb->front[1].text);
  if (test_fb->front.size() > 1) EXPECT_EQ(0,             test_fb->front[1].wlo);
  if (test_fb->front.size() > 1) EXPECT_EQ(1,             test_fb->front[1].wll);
  if (test_fb->front.size() > 1) EXPECT_EQ(point(0,5*fh), test_fb->front[1].p);
  test_fb->front.clear();
  EXPECT_EQ(2, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"3",             test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh),    test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[0].b);
  if (test_fb->paint.size() > 1) EXPECT_EQ(u"2 2 2",         test_fb->paint[1].text);
  if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,3*fh),    test_fb->paint[1].p);
  if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[1].b);
  test_fb->paint.clear();

  // 1: 222 : 2=2:0, 1=2:1, 3=2:2
  e.v_scrolled = 1.0/(e.WrappedLines()-1);
  e.UpdateScrolled();
  EXPECT_EQ(0, e.start_line_adjust);  EXPECT_EQ(3, e.start_line_cutoff);
  EXPECT_EQ(3, e.end_line_adjust);    EXPECT_EQ(0, e.end_line_cutoff);
  EXPECT_EQ(1, e.line.Size());
  EXPECT_EQ(3, e.fb_wrapped_lines);
  EXPECT_EQ(1, test_fb->front.size());
  EXPECT_NEAR(-2/3.0, test_fb->scroll.y, 1e-6);
  EXPECT_EQ(point(0,2*fh), test_fb->p);
  if (test_fb->front.size() > 0) EXPECT_EQ(u"2 2 2",      test_fb->front[0].text);
  if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wlo);
  if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wll);
  if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
  test_fb->back.clear();
  EXPECT_EQ(2, test_fb->paint.size());
  if (test_fb->paint.size() > 0) EXPECT_EQ(u"2 2 2",        test_fb->paint[0].text);
  if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,5*fh),   test_fb->paint[0].p);
  if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
  if (test_fb->paint.size() > 1) EXPECT_EQ(u"2 2 2",        test_fb->paint[1].text);
  if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),   test_fb->paint[1].p);
  if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
  test_fb->paint.clear();
}

TEST(GUITest, Terminal) {
  int cursor_attr = 0;
  cursor_attr = Terminal::Style::SetFGColorIndex(cursor_attr, 7);
  EXPECT_EQ(7,  Terminal::Style::GetFGColorIndex(cursor_attr));
  cursor_attr = Terminal::Style::SetFGColorIndex(cursor_attr, 2);
  EXPECT_EQ(2,  Terminal::Style::GetFGColorIndex(cursor_attr));
  cursor_attr = Terminal::Style::SetBGColorIndex(cursor_attr, 3);
  EXPECT_EQ(3,  Terminal::Style::GetBGColorIndex(cursor_attr));
  EXPECT_EQ(2,  Terminal::Style::GetFGColorIndex(cursor_attr));
  cursor_attr |= Terminal::Style::Bold;
  EXPECT_EQ(10, Terminal::Style::GetFGColorIndex(cursor_attr));
  EXPECT_EQ(3,  Terminal::Style::GetBGColorIndex(cursor_attr));
  cursor_attr = Terminal::Style::SetFGColorIndex(cursor_attr, 4);
  cursor_attr = Terminal::Style::SetBGColorIndex(cursor_attr, 0);
  EXPECT_EQ(12, Terminal::Style::GetFGColorIndex(cursor_attr));
  EXPECT_EQ(0,  Terminal::Style::GetBGColorIndex(cursor_attr));
  cursor_attr &= ~Terminal::Style::Bold;
  EXPECT_EQ(4,  Terminal::Style::GetFGColorIndex(cursor_attr));

  TerminalTest ta(nullptr, screen->gd, app->fonts->Fake(), point(80, 25));
  int tw = ta.term_width, th = ta.term_height, fw = ta.style.font->FixedWidth(), fh = ta.style.font->Height();
  ta.CheckResized(Box(tw*fw, th*fh));
  EXPECT_EQ(80, ta.term_width);
  EXPECT_EQ(25, ta.term_height);
  EXPECT_EQ(25, ta.line.Size());
  EXPECT_EQ(25, ta.GetPrimaryFrameBuffer()->lines);
  EXPECT_EQ(1, ta.term_cursor.x);
  EXPECT_EQ(1, ta.term_cursor.y);
  ta.Write("\x1b[13;33H"); EXPECT_EQ(33, ta.term_cursor.x); EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[Z");      EXPECT_EQ(25, ta.term_cursor.x); EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[Z");      EXPECT_EQ(17, ta.term_cursor.x); EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[Z");      EXPECT_EQ(9,  ta.term_cursor.x); EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[Z");      EXPECT_EQ(1,  ta.term_cursor.x); EXPECT_EQ(13, ta.term_cursor.y);

  ta.AssignLineTextToLineNumber();
  for (int i=1; i<=th; ++i) EXPECT_EQ(&ta.line[-th + i-1], ta.GetTermLine(i));
  for (int i=1; i<=th; ++i) EXPECT_EQ(Str16Cat(i), ta.GetTermLine(i)->Text16());
  for (int i=1; i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+1)), ta.GetTermLine(i)->p);

  ta.Write("\x1b[1;23r"); EXPECT_EQ(1, ta.scroll_region_beg); EXPECT_EQ(23, ta.scroll_region_end);
  ta.Write("\x1b[7;1H");  EXPECT_EQ(1, ta.term_cursor.x);     EXPECT_EQ(7,  ta.term_cursor.y);
  ta.Write("\x1b[M");
  for (int i=1;  i<=23; ++i) EXPECT_EQ(i<23 ? Str16Cat(i+(i>=7)) : u"", ta.GetTermLine(i)->Text16());
  for (int i=24; i<=th; ++i) EXPECT_EQ(Str16Cat(i),                     ta.GetTermLine(i)->Text16());
  for (int i=1;  i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+1)),            ta.GetTermLine(i)->p);

  ta.Write("\x1b[13;1H");  EXPECT_EQ(1, ta.term_cursor.x);    EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[M");
  for (int i=1;  i<=23; ++i) EXPECT_EQ(i<22 ? Str16Cat(i+(i>=7)+(i>=13)) : u"", ta.GetTermLine(i)->Text16());
  for (int i=24; i<=th; ++i) EXPECT_EQ(Str16Cat(i),                             ta.GetTermLine(i)->Text16());
  for (int i=1;  i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+1)),                    ta.GetTermLine(i)->p);

  ta.AssignLineTextToLineNumber();
  for (int i=1; i<=th; ++i) EXPECT_EQ(Str16Cat(i), ta.GetTermLine(i)->Text16());
  for (int i=1; i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+1)), ta.GetTermLine(i)->p);

  ta.Write("\x1b[7;23r"); EXPECT_EQ(7, ta.scroll_region_beg); EXPECT_EQ(23, ta.scroll_region_end);
  ta.Write("\x1b[23;1H"); EXPECT_EQ(1, ta.term_cursor.x);     EXPECT_EQ(23, ta.term_cursor.y);

  ta.Write("\n");
  for (int i=1;  i<=23; ++i) EXPECT_EQ(i<23 ? Str16Cat(i+(i>=7)) : u"",    ta.GetTermLine(i)->Text16());
  for (int i=24; i<=th; ++i) EXPECT_EQ(Str16Cat(i),                        ta.GetTermLine(i)->Text16());
  for (int i=1;  i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+((i<7)||i>=24))),  ta.GetTermLine(i)->p);

  ta.Write("\x1b[13;1H");  EXPECT_EQ(1, ta.term_cursor.x);    EXPECT_EQ(13, ta.term_cursor.y);
  ta.Write("\x1b[M");
  for (int i=1;  i<=23; ++i) EXPECT_EQ(i<22 ? Str16Cat(i+(i>=7)+(i>=13)) : u"", ta.GetTermLine(i)->Text16());
  for (int i=24; i<=th; ++i) EXPECT_EQ(Str16Cat(i),                             ta.GetTermLine(i)->Text16());
  for (int i=1;  i<=th; ++i) EXPECT_EQ(point(0,fh*(th-i+((i<7)||i>=24))),       ta.GetTermLine(i)->p);
}

TEST(GUITest, LineTokenProcessor) {
  TextAreaTest ta(screen->gd, app->fonts->Fake(), 10);
  ta.token_processing = 1;
  TextBox::Line *L = ta.line.InsertAt(-1);

  L->UpdateText(0, "a", 0);
  EXPECT_EQ(u"a", L->Text16()); EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ(u"a", ta.token[0].word); }
  ta.token.clear();

  L->UpdateText(1, "c", 0);
  EXPECT_EQ(u"ac", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -8); EXPECT_EQ(u"a",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"ac", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(1, "b", 0);
  EXPECT_EQ(u"abc", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ(u"ac",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ(u"abc", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(0, "0", 0);
  EXPECT_EQ(u"0abc", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -9); EXPECT_EQ(u"abc",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ(u"0abc", ta.token[1].word); }
  ta.token.clear();

  L->Erase(0, 1);
  EXPECT_EQ(u"abc", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -3); EXPECT_EQ(u"0abc", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   9); EXPECT_EQ(u"abc",  ta.token[1].word); }
  ta.token.clear();

  L->Erase(1, 1);
  EXPECT_EQ(u"ac", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -1); EXPECT_EQ(u"abc", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   7); EXPECT_EQ(u"ac",  ta.token[1].word); }
  ta.token.clear();

  L->Erase(1, 1);
  EXPECT_EQ(u"a", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -2); EXPECT_EQ(u"ac", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   8); EXPECT_EQ(u"a",  ta.token[1].word); }
  ta.token.clear();

  L->Erase(0, 1);
  EXPECT_EQ(u"", L->Text16()); EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -4); EXPECT_EQ(u"a", ta.token[0].word); }
  ta.token.clear();

  L->UpdateText(0, "aabb", 0);
  EXPECT_EQ(u"aabb", L->Text16()); EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ(u"aabb", ta.token[0].word); }
  ta.token.clear();

  L->UpdateText(2, " ", 0);
  EXPECT_EQ(u"aa bb", L->Text16()); EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ(u"aabb", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ(u"aa",   ta.token[1].word); }
  if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  6); EXPECT_EQ(u"bb",   ta.token[2].word); }
  ta.token.clear();

  L->Erase(2, 1);
  EXPECT_EQ(u"aabb", L->Text16()); EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"aa",   ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type, -6); EXPECT_EQ(u"bb",   ta.token[1].word); }
  if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  7); EXPECT_EQ(u"aabb", ta.token[2].word); }
  ta.token.clear();

  L->Clear();
  ta.insert_mode = 0;
  L->UpdateText(0, "a", 0);
  EXPECT_EQ(u"a", L->Text16()); EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ(u"a", ta.token[0].word); }
  ta.token.clear();

  L->UpdateText(1, "cc", 0);
  EXPECT_EQ(u"acc", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"a",   ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"acc", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(1, "b", 0);
  EXPECT_EQ(u"abc", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -1); EXPECT_EQ(u"acc", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ(u"abc", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(3, "bb", 0);
  EXPECT_EQ(u"abcbb", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"abc",   ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"abcbb", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(0, "aab", 0);
  EXPECT_EQ(u"aabbb", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -3); EXPECT_EQ(u"abcbb", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ(u"aabbb", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(2, " ", 0);
  EXPECT_EQ(u"aa bb", L->Text16()); EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -1); EXPECT_EQ(u"aabbb", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ(u"aa",    ta.token[1].word); }
  if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  6); EXPECT_EQ(u"bb",    ta.token[2].word); }
  ta.token.clear();

  L->UpdateText(2, "c", 0);
  EXPECT_EQ(u"aacbb", L->Text16()); EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"aa",    ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type, -6); EXPECT_EQ(u"bb",    ta.token[1].word); }
  if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  1); EXPECT_EQ(u"aacbb", ta.token[2].word); }
  ta.token.clear();

  L->UpdateText(5, "ccdee", 0);
  EXPECT_EQ(u"aacbbccdee", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"aacbb",      ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"aacbbccdee", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(5, " ccdee", 0);
  EXPECT_EQ(u"aacbb ccdee", L->Text16()); EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ(u"aacbbccdee", ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  4); EXPECT_EQ(u"ccdee",      ta.token[1].word); }
  if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  5); EXPECT_EQ(u"aacbb",      ta.token[2].word); }
  ta.token.clear();

  L->UpdateText(5, " ccdee", 0);
  EXPECT_EQ(u"aacbb ccdee", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -4); EXPECT_EQ(u"ccdee",      ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  4); EXPECT_EQ(u"ccdee",      ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(5, " ", 0);
  EXPECT_EQ(u"aacbb ccdee", L->Text16()); EXPECT_EQ(0, ta.token.size());

  L->OverwriteTextAt(10, StringPiece(string("Z")), 0);
  EXPECT_EQ(u"aacbb ccdeZ", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ(u"ccdee",      ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"ccdeZ",      ta.token[1].word); }
  ta.token.clear();

  L->OverwriteTextAt(10, StringPiece(string(" ")), 0);
  EXPECT_EQ(u"aacbb ccde ", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ(u"ccdeZ",      ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ(u"ccde",       ta.token[1].word); }
  ta.token.clear();

  L->OverwriteTextAt(10, StringPiece(string(" ")), 0);
  EXPECT_EQ(u"aacbb ccde ", L->Text16()); EXPECT_EQ(0, ta.token.size());
  ta.token.clear();

  L->OverwriteTextAt(10, StringPiece(string("A")), 0);
  EXPECT_EQ(u"aacbb ccdeA", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ(u"ccde",       ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"ccdeA",      ta.token[1].word); }
  ta.token.clear();

  // test wide chars
  L->Clear();
  ta.insert_mode = 1;
  L->UpdateText(0, "\xff", 0);
  EXPECT_EQ(u"\xff\ufeff", L->Text16()); EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ(u"\xff\ufeff", ta.token[0].word); }
  ta.token.clear();

  L->UpdateText(2, "y", 0);
  EXPECT_EQ(u"\xff\ufeffy", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -8); EXPECT_EQ(u"\xff\ufeff",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ(u"\xff\ufeffy", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(2, "x", 0);
  EXPECT_EQ(u"\xff\ufeffxy", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ(u"\xff\ufeffy",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ(u"\xff\ufeffxy", ta.token[1].word); }
  ta.token.clear();

  L->UpdateText(0, "0", 0);
  EXPECT_EQ(u"0\xff\ufeffxy", L->Text16()); EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -9); EXPECT_EQ(u"\xff\ufeffxy",  ta.token[0].word); }
  if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ(u"0\xff\ufeffxy", ta.token[1].word); }
  ta.token.clear();
}

TEST(GUITest, TerminalTokenProcessor) {
  TerminalTest ta(NULL, screen->gd, app->fonts->Fake(), point(80, 25));
  EXPECT_EQ(80, ta.term_width);
  EXPECT_EQ(25, ta.line.Size());

  ta.Write(string(80, 'a'), 0, 0);
  EXPECT_EQ(1, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(4,   ta.token[0].type); EXPECT_EQ(string(80, 'a'), ta.token[0].text); }
  ta.token.clear();

  ta.Write("b", 0, 0);
  EXPECT_EQ(2, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(-40, ta.token[0].type); EXPECT_EQ(string(80, 'a'),       ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(  4, ta.token[1].type); EXPECT_EQ(string(80, 'a') + "b", ta.token[1].text); }
  ta.token.clear();                       

  ta.Write("c", 0, 0);                    
  EXPECT_EQ(2, ta.token.size());          
  if (ta.token.size()>0) { EXPECT_EQ( -5, ta.token[0].type); EXPECT_EQ(string(80, 'a') + "b",  ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(  2, ta.token[1].type); EXPECT_EQ(string(80, 'a') + "bc", ta.token[1].text); }
  ta.token.clear();

  ta.term_cursor.x = 1;
  ta.Write(" ", 0, 0);
  EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(-3, ta.token[0].type); EXPECT_EQ(string(80, 'a') + "bc", ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(30, ta.token[1].type); EXPECT_EQ(string(80, 'a') + "",   ta.token[1].text); }
  if (ta.token.size()>2) { EXPECT_EQ( 6, ta.token[2].type); EXPECT_EQ(                  "c",  ta.token[2].text); }
  ta.token.clear();

  ta.term_cursor.x = 1;
  ta.Write("z", 0, 0);
  EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(-6,  ta.token[0].type); EXPECT_EQ(                  "c",  ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(-30, ta.token[1].type); EXPECT_EQ(string(80, 'a') + "",   ta.token[1].text); }
  if (ta.token.size()>2) { EXPECT_EQ( 3,  ta.token[2].type); EXPECT_EQ(string(80, 'a') + "zc", ta.token[2].text); }
  ta.token.clear();

  ta.term_cursor.y--;
  ta.term_cursor.x = 80;
  ta.Write(" ", 0, 0);
  EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(-2,  ta.token[0].type); EXPECT_EQ(string(80, 'a') + "zc", ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(22,  ta.token[1].type); EXPECT_EQ(                  "zc", ta.token[1].text); }
  if (ta.token.size()>2) { EXPECT_EQ( 5,  ta.token[2].type); EXPECT_EQ(string(79, 'a') + "",   ta.token[2].text); }
  ta.token.clear();

  ta.term_cursor.x = 80;
  ta.Write("w", 0, 0);
  EXPECT_EQ(3, ta.token.size());
  if (ta.token.size()>0) { EXPECT_EQ(-5,  ta.token[0].type); EXPECT_EQ(string(79, 'a') + "",    ta.token[0].text); }
  if (ta.token.size()>1) { EXPECT_EQ(-22, ta.token[1].type); EXPECT_EQ(                  "zc",  ta.token[1].text); }
  if (ta.token.size()>2) { EXPECT_EQ( 2,  ta.token[2].type); EXPECT_EQ(string(79, 'a') + "wzc", ta.token[2].text); }
  ta.token.clear();
}

}; // namespace LFL
