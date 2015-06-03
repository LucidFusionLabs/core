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
#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/ipc.h"
#include "crawler/html.h"
#include "crawler/document.h"

namespace LFL {
struct LinesFrameBufferTest : public TextGUI::LinesFrameBuffer {
    struct LineOp {
        point p; string text; int xo, wlo, wll;
        LineOp(TextGUI::Line *L, int X, int O, int S) : p(L->p), text(L->Text()), xo(X), wlo(O), wll(S) {}
        string DebugString() const { return StrCat("LineOp text(", text, ") ", xo, " ", wlo, " ", wll); }
    };
    struct PaintOp {
        string text; int lines; point p; Box b;
        PaintOp(TextGUI::Line *L, const point &P, const Box &B) : text(L->Text()), lines(L->Lines()), p(P), b(B) {}
        string DebugString() const { return StrCat("PaintOp text(", text, ") ", p, " ", b); }
    };
    static vector<LineOp> front, back;
    static vector<PaintOp> paint;
    LinesFrameBufferTest() { paint_cb = &LinesFrameBufferTest::PaintTest; }
    virtual void Clear(TextGUI::Line *l)                              override { EXPECT_EQ(0, 1); }
    virtual void Update(TextGUI::Line *l, int flag=0)                 override { EXPECT_EQ(0, 1); }
    virtual void Update(TextGUI::Line *l, const point &p, int flag=0) override { EXPECT_EQ(0, 1); }
    virtual void PushFrontAndUpdateOffset(TextGUI::Line *l, int lo)   override { EXPECT_EQ(0, 1); }
    virtual void PushBackAndUpdateOffset (TextGUI::Line *l, int lo)   override { EXPECT_EQ(0, 1); }
    virtual int PushFrontAndUpdate(TextGUI::Line *l, int xo=0, int wlo=0, int wll=0, int flag=0) override { int ret = TextGUI::LinesFrameBuffer::PushFrontAndUpdate(l,xo,wlo,wll); front.emplace_back(l,xo,wlo,wll); return ret; }
    virtual int PushBackAndUpdate (TextGUI::Line *l, int xo=0, int wlo=0, int wll=0, int flag=0) override { int ret = TextGUI::LinesFrameBuffer::PushBackAndUpdate (l,xo,wlo,wll); back .emplace_back(l,xo,wlo,wll); return ret; }
    static point PaintTest(TextGUI::Line *l, point lp, const Box &b)
    { l->Draw(lp + b.Position()); paint.emplace_back(l, lp, b); return point(lp.x, lp.y-b.h); }
};
vector<LinesFrameBufferTest::LineOp> LinesFrameBufferTest::front;
vector<LinesFrameBufferTest::LineOp> LinesFrameBufferTest::back;
vector<LinesFrameBufferTest::PaintOp> LinesFrameBufferTest::paint;

struct TextAreaTest : public TextArea {
    struct UpdateTokenOp {
        Line *l; string word; int type;
        UpdateTokenOp(Line *L, const string &W, int T) : l(L), word(W), type(T) {}
    };
    vector<UpdateTokenOp> token;
    LinesFrameBufferTest line_fb_test;
    TextAreaTest(Window *W, Font *F, int S=200) : TextArea(W,F,S) {}
    virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
    virtual void UpdateToken(Line *l, int wo, int wl, int t, const LineTokenProcessor*) override {
        token.emplace_back(l, DrawableBoxRun(&l->data->glyphs[wo], wl).Text(), t);
    }
};
struct EditorTest : public Editor {
    LinesFrameBufferTest line_fb_test;
    EditorTest(Window *W, Font *F, File *I, bool Wrap=0) : Editor(W,F,I,Wrap) { line_fb_test.wrap=Wrap; }
    virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
};
struct TerminalTest : public Terminal {
    struct UpdateTokenOp {
        Line *bl, *el; int bo, eo, type; string text;
        UpdateTokenOp(Line *BL, int BO, Line *EL, int EO, const string &X, int T) : bl(BL), el(EL), bo(BO), eo(EO), type(T), text(X) {}
    };
    vector<UpdateTokenOp> token;
    LinesFrameBufferTest line_fb_test;
    TerminalTest(ByteSink *S, Window *W, Font *F) : Terminal(S,W,F) {}
    virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
    void UpdateLongToken(Line *BL, int BO, Line *EL, int EO, const string &text, int T) {
        token.emplace_back(BL, BO, EL, EO, text, T);
    }
};

TEST(GUITest, TextArea) { 
    {
        TextAreaTest ta(screen, Fonts::Fake(), 10);
        ta.line.InsertAt(-1)->AssignText("a");
        EXPECT_EQ("a", ta.line[-1].Text());

        ta.line.InsertAt(-1)->AssignText("b");
        ta.line.InsertAt(-1)->AssignText("c");
        ta.line.InsertAt(-1)->AssignText("d");
        EXPECT_EQ("a", ta.line[-4].Text()); EXPECT_EQ(-4, ta.line.IndexOf(&ta.line[-4]));
        EXPECT_EQ("b", ta.line[-3].Text()); EXPECT_EQ(-3, ta.line.IndexOf(&ta.line[-3]));
        EXPECT_EQ("c", ta.line[-2].Text()); EXPECT_EQ(-2, ta.line.IndexOf(&ta.line[-2]));
        EXPECT_EQ("d", ta.line[-1].Text()); EXPECT_EQ(-1, ta.line.IndexOf(&ta.line[-1]));
        ta.line.InsertAt(-4)->AssignText("z");
        EXPECT_EQ("z", ta.line[-4].Text());
        EXPECT_EQ("a", ta.line[-3].Text());
        EXPECT_EQ("b", ta.line[-2].Text());
        EXPECT_EQ("c", ta.line[-1].Text());
        ta.line.InsertAt(-2)->AssignText("y");
        EXPECT_EQ("z", ta.line[-4].Text());
        EXPECT_EQ("a", ta.line[-3].Text());
        EXPECT_EQ("y", ta.line[-2].Text());
        EXPECT_EQ("b", ta.line[-1].Text());
        ta.line.PopBack(2);
        EXPECT_EQ("z", ta.line[-2].Text());
        EXPECT_EQ("a", ta.line[-1].Text());
        ta.line.PushFront()->AssignText("w");
        ta.line.PushFront()->AssignText("u");
        EXPECT_EQ("u", ta.line[-4].Text());
        EXPECT_EQ("w", ta.line[-3].Text());
        EXPECT_EQ("z", ta.line[-2].Text());
        EXPECT_EQ("a", ta.line[-1].Text());
        ta.line.PushFront()->AssignText("1");
        ta.line.PushFront()->AssignText("2");
        ta.line.PushFront()->AssignText("3");
        ta.line.PushFront()->AssignText("4");
        ta.line.PushFront()->AssignText("5");
        ta.line.PushFront()->AssignText("6");
        ta.line.PushFront()->AssignText("7");
        EXPECT_EQ("7", ta.line[-10].Text()); EXPECT_EQ(-10, ta.line.IndexOf(&ta.line[-10]));
        EXPECT_EQ("6", ta.line[-9] .Text()); EXPECT_EQ(-9,  ta.line.IndexOf(&ta.line[-9]));
        EXPECT_EQ("5", ta.line[-8] .Text()); EXPECT_EQ(-8,  ta.line.IndexOf(&ta.line[-8]));
        EXPECT_EQ("4", ta.line[-7] .Text()); EXPECT_EQ(-7,  ta.line.IndexOf(&ta.line[-7]));
        EXPECT_EQ("3", ta.line[-6] .Text()); EXPECT_EQ(-6,  ta.line.IndexOf(&ta.line[-6]));
        EXPECT_EQ("2", ta.line[-5] .Text()); EXPECT_EQ(-5,  ta.line.IndexOf(&ta.line[-5]));
        EXPECT_EQ("1", ta.line[-4] .Text()); EXPECT_EQ(-4,  ta.line.IndexOf(&ta.line[-4]));
        EXPECT_EQ("u", ta.line[-3] .Text()); EXPECT_EQ(-3,  ta.line.IndexOf(&ta.line[-3]));
        EXPECT_EQ("w", ta.line[-2] .Text()); EXPECT_EQ(-2,  ta.line.IndexOf(&ta.line[-2]));
        EXPECT_EQ("z", ta.line[-1] .Text()); EXPECT_EQ(-1,  ta.line.IndexOf(&ta.line[-1]));
    }

    { // 0: 122, 1: 222, 2: 223, 3: 234, 4: 344, 5: 445
        TextGUI::Line *L;
        Font *font = Fonts::Fake();
        TextAreaTest ta(screen, font);
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

        LinesFrameBufferTest *test_fb = &ta.line_fb_test;
        int fh = font->Height();
        Box b(128, 3*fh);
        ta.Draw(b, false);
        int w = test_fb->w;

        // 0: 122 : 1=1:0, 2=2:2, 3=2:1
        EXPECT_EQ(b.w, test_fb->w);
        EXPECT_EQ(b.h, test_fb->h);
        EXPECT_EQ(0, ta.start_line); EXPECT_EQ(0, ta.start_line_adjust);
        EXPECT_EQ(1, ta.end_line);   EXPECT_EQ(1, ta.end_line_cutoff);
        EXPECT_EQ(2, test_fb->front.size());
        EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,0), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("1",           test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(3,             test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,fh),   test_fb->front[0].p);
        if (test_fb->front.size() > 1) EXPECT_EQ("2\n2\n2",     test_fb->front[1].text);
        if (test_fb->front.size() > 1) EXPECT_EQ(0,             test_fb->front[1].wlo);
        if (test_fb->front.size() > 1) EXPECT_EQ(2,             test_fb->front[1].wll);
        if (test_fb->front.size() > 1) EXPECT_EQ(point(0,4*fh), test_fb->front[1].p);
        test_fb->front.clear();
        EXPECT_EQ(2, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("1",              test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),      test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[0].b);
        if (test_fb->paint.size() > 1) EXPECT_EQ("2\n2\n2",        test_fb->paint[1].text);
        if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,3*fh),    test_fb->paint[1].p);
        if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,fh,w,2*fh), test_fb->paint[1].b);
        test_fb->paint.clear();

        // 1: 222 : 2=2:2, 3=2:1, 1=2:0
        ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(0, ta.start_line_adjust);
        EXPECT_EQ(1, ta.end_line);   EXPECT_EQ(0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->front.size());
        EXPECT_NEAR(-1/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,fh), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("2\n2\n2",   test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(2,           test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(1,           test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,fh), test_fb->front[0].p);
        test_fb->front.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("2\n2\n2",        test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),      test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[0].b);
        test_fb->paint.clear();

        // 2: 223 : 3=2:1, 1=2:0, 2=3:0
        ta.v_scrolled = 2.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(-1, ta.start_line_adjust);
        EXPECT_EQ(2, ta.end_line);   EXPECT_EQ( 0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->front.size());
        EXPECT_NEAR(-2/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,2*fh), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("3",           test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,fh*2), test_fb->front[0].p);
        test_fb->front.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("3",           test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh*2), test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
        test_fb->paint.clear();

        // 1: 222 : 2=2:2, 3=2:1, 1=2:0
        ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(0, ta.start_line_adjust);
        EXPECT_EQ(1, ta.end_line);   EXPECT_EQ(0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->back.size());
        EXPECT_NEAR(-1/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,fh), test_fb->p);
        if (test_fb->back.size() > 0) EXPECT_EQ("2\n2\n2",     test_fb->back[0].text);
        if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wlo);
        if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
        if (test_fb->back.size() > 0) EXPECT_EQ(point(0,fh*4), test_fb->back[0].p);
        test_fb->back.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("2\n2\n2",        test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh*2),    test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,fh*2,w,fh), test_fb->paint[0].b);
        test_fb->paint.clear();

        // 0: 122 : 1=1:0, 2=2:2, 3=2:1
        ta.v_scrolled = 0;
        ta.UpdateScrolled();
        EXPECT_EQ(0, ta.start_line); EXPECT_EQ(0, ta.start_line_adjust);
        EXPECT_EQ(1, ta.end_line);   EXPECT_EQ(1, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->back.size());
        EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,0), test_fb->p);
        if (test_fb->back.size() > 0) EXPECT_EQ("1",         test_fb->back[0].text);
        if (test_fb->back.size() > 0) EXPECT_EQ(0,           test_fb->back[0].wlo);
        if (test_fb->back.size() > 0) EXPECT_EQ(point(0,fh), test_fb->back[0].p);
        test_fb->back.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("1",           test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),   test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
        test_fb->paint.clear();

        // 2: 223 : 3=2:1, 1=2:0, 2=3:0
        ta.v_scrolled = 2.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(-1, ta.start_line_adjust);
        EXPECT_EQ(2, ta.end_line);   EXPECT_EQ( 0, ta.end_line_cutoff);
        EXPECT_EQ(2, test_fb->front.size());
        EXPECT_NEAR(-2/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,2*fh), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("2\n2\n2",     test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,fh),   test_fb->front[0].p);
        if (test_fb->front.size() > 1) EXPECT_EQ("3",           test_fb->front[1].text);
        if (test_fb->front.size() > 1) EXPECT_EQ(0,             test_fb->front[1].wlo);
        if (test_fb->front.size() > 1) EXPECT_EQ(1,             test_fb->front[1].wll);
        if (test_fb->front.size() > 1) EXPECT_EQ(point(0,fh*2), test_fb->front[1].p);
        test_fb->front.clear();
        EXPECT_EQ(2, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("2\n2\n2",     test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),   test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
        if (test_fb->paint.size() > 1) EXPECT_EQ("3",           test_fb->paint[1].text);
        if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh), test_fb->paint[1].p);
        if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[1].b);
        test_fb->paint.clear();

        // 4: 344 : 2=3:0, 3=4:1, 1=4:0
        ta.v_scrolled = 4.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(2, ta.start_line); EXPECT_EQ( 0, ta.start_line_adjust);
        EXPECT_EQ(3, ta.end_line);   EXPECT_EQ( 0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->front.size());
        EXPECT_NEAR(-1/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,fh), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("4\n4",      test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(0,           test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(2,           test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,fh), test_fb->front[0].p);
        test_fb->front.clear();
        EXPECT_EQ(2, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("4\n4",          test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,4*fh),   test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
        if (test_fb->paint.size() > 1) EXPECT_EQ("4\n4",          test_fb->paint[1].text);
        if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,fh),     test_fb->paint[1].p);
        if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
        test_fb->paint.clear();

        // 5: 445 : 3=4:1, 1=4:0, 2=5:0
        ta.v_scrolled = 5.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(3, ta.start_line); EXPECT_EQ( 0, ta.start_line_adjust);
        EXPECT_EQ(4, ta.end_line);   EXPECT_EQ( 0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->front.size());
        EXPECT_NEAR(-2/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,2*fh), test_fb->p);
        if (test_fb->front.size() > 0) EXPECT_EQ("5",           test_fb->front[0].text);
        if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
        if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
        if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
        test_fb->front.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("5",           test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
        test_fb->paint.clear();

        // 3: 234 : 1=2:0, 2=3:0, 3=4:1
        ta.v_scrolled = 3.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(-2, ta.start_line_adjust);
        EXPECT_EQ(3, ta.end_line);   EXPECT_EQ( 1, ta.end_line_cutoff);
        EXPECT_EQ(2, test_fb->back.size());
        EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,0), test_fb->p);
        if (test_fb->back.size() > 0) EXPECT_EQ("3",           test_fb->back[0].text);
        if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
        if (test_fb->back.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->back[0].p);
        if (test_fb->back.size() > 1) EXPECT_EQ("2\n2\n2",     test_fb->back[1].text);
        if (test_fb->back.size() > 1) EXPECT_EQ(0,             test_fb->back[1].wlo);
        if (test_fb->back.size() > 1) EXPECT_EQ(point(0,fh),   test_fb->back[1].p);
        test_fb->back.clear();
        EXPECT_EQ(2, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("3",           test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
        if (test_fb->paint.size() > 0) EXPECT_EQ("2\n2\n2",     test_fb->paint[1].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),   test_fb->paint[1].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[1].b);
        test_fb->paint.clear();

        // 1: 222 : 2=2:2, 3=2:1, 1=2:0
        ta.v_scrolled = 1.0/(ta.WrappedLines()-1);
        ta.UpdateScrolled();
        EXPECT_EQ(1, ta.start_line); EXPECT_EQ(0, ta.start_line_adjust);
        EXPECT_EQ(1, ta.end_line);   EXPECT_EQ(0, ta.end_line_cutoff);
        EXPECT_EQ(1, test_fb->back.size());
        EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
        EXPECT_EQ(point(0,fh), test_fb->p);
        if (test_fb->back.size() > 0) EXPECT_EQ("2\n2\n2",     test_fb->back[0].text);
        if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wlo);
        if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wll);
        if (test_fb->back.size() > 0) EXPECT_EQ(point(0,fh*4), test_fb->back[0].p);
        test_fb->back.clear();
        EXPECT_EQ(1, test_fb->paint.size());
        if (test_fb->paint.size() > 0) EXPECT_EQ("2\n2\n2",        test_fb->paint[0].text);
        if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh*3),    test_fb->paint[0].p);
        if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,fh,w,fh*2), test_fb->paint[0].b);
        test_fb->paint.clear();
    }
}

TEST(GUITest, Editor) {
    Font *font = Fonts::Fake();
    int fh = font->Height(), w = font->fixed_width;
    EXPECT_NE(0, fh); EXPECT_NE(0, w);
    EditorTest e(screen, font, new BufferFile("1\n2 2 2\n3\n4 4\n5\n"), true);
    LinesFrameBufferTest *test_fb = &e.line_fb_test;
    Box b(w, 3*fh);
    e.Draw(b, false);

    // 0: 122 : 3=1:0, 2=2:0, 1=2:1
    EXPECT_EQ(0, e.start_line_adjust);
    EXPECT_EQ(1, e.start_line_cutoff);
    EXPECT_EQ(2, e.end_line_adjust);
    EXPECT_EQ(1, e.end_line_cutoff);
    EXPECT_EQ(2, e.line.Size());
    EXPECT_EQ(4, e.fb_wrapped_lines);
    EXPECT_EQ(2, test_fb->back.size());
    if (test_fb->back.size() > 0) EXPECT_EQ("1",           test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(3,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->back[0].p);
    if (test_fb->back.size() > 1) EXPECT_EQ("2 2 2",       test_fb->back[1].text);
    if (test_fb->back.size() > 1) EXPECT_EQ(0,             test_fb->back[1].wlo);
    if (test_fb->back.size() > 1) EXPECT_EQ(2,             test_fb->back[1].wll);
    if (test_fb->back.size() > 1) EXPECT_EQ(point(0,2*fh), test_fb->back[1].p);
    test_fb->back.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("1",             test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),   test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),   test_fb->paint[0].b);
    if (test_fb->paint.size() > 1) EXPECT_EQ("2 2 2",         test_fb->paint[1].text);
    if (test_fb->paint.size() > 1) EXPECT_EQ(3,               test_fb->paint[1].lines);
    if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),   test_fb->paint[1].p);
    if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
    test_fb->paint.clear();

    // 1: 222 : 2=2:0, 1=2:1, 3=2:2
    e.v_scrolled = 1.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(0, e.start_line_adjust);
    EXPECT_EQ(3, e.start_line_cutoff);
    EXPECT_EQ(0, e.end_line_cutoff);
    EXPECT_EQ(3, e.end_line_adjust);
    EXPECT_EQ(1, e.line.Size());
    EXPECT_EQ(3, e.fb_wrapped_lines);
    EXPECT_EQ(1, test_fb->back.size());
    EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb->back.size() > 0) EXPECT_EQ("2 2 2",       test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,5*fh), test_fb->back[0].p);
    test_fb->back.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("2 2 2",          test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),    test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[0].b);
    test_fb->paint.clear();

    // 2: 223 : 1=2:1, 3=2:2, 2=3:0
    e.v_scrolled = 2.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(-1, e.start_line_adjust);
    EXPECT_EQ( 2, e.start_line_cutoff);
    EXPECT_EQ( 0, e.end_line_cutoff);
    EXPECT_EQ( 1, e.end_line_adjust);
    EXPECT_EQ( 2, e.line.Size());
    EXPECT_EQ( 4, e.fb_wrapped_lines);
    EXPECT_EQ( 1, test_fb->back.size());
    EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,fh), test_fb->p);
    if (test_fb->back.size() > 0) EXPECT_EQ("3",           test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->back[0].p);
    test_fb->back.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("3",           test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    test_fb->paint.clear();

    // 1: 222 : 2=2:0, 1=2:1, 3=2:2
    e.v_scrolled = 1.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(0, e.start_line_adjust);
    EXPECT_EQ(3, e.start_line_cutoff);
    EXPECT_EQ(0, e.end_line_cutoff);
    EXPECT_EQ(3, e.end_line_adjust);
    EXPECT_EQ(1, e.line.Size());
    EXPECT_EQ(3, e.fb_wrapped_lines);
    EXPECT_EQ(1, test_fb->front.size());
    EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb->front.size() > 0) EXPECT_EQ("2 2 2",       test_fb->front[0].text);
    if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wlo);
    if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
    if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
    test_fb->front.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("2 2 2",       test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    test_fb->paint.clear();

    // 0: 122 : 3=1:0, 2=2:0, 1=2:1
    e.v_scrolled = 0;
    e.UpdateScrolled();
    EXPECT_EQ(0, e.start_line_adjust);
    EXPECT_EQ(1, e.start_line_cutoff);
    EXPECT_EQ(1, e.end_line_cutoff);
    EXPECT_EQ(2, e.end_line_adjust);
    EXPECT_EQ(2, e.line.Size());
    EXPECT_EQ(4, e.fb_wrapped_lines);
    EXPECT_EQ(1, test_fb->front.size());
    if (test_fb->front.size() > 0) EXPECT_EQ("1",           test_fb->front[0].text);
    if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
    if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wll);
    if (test_fb->front.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->front[0].p);
    test_fb->front.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("1",           test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh), test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    test_fb->paint.clear();

    // 2: 223 : 1=2:1, 3=2:2, 2=3:0
    e.v_scrolled = 2.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(-1, e.start_line_adjust);
    EXPECT_EQ( 2, e.start_line_cutoff);
    EXPECT_EQ( 0, e.end_line_cutoff);
    EXPECT_EQ( 1, e.end_line_adjust);
    EXPECT_EQ( 2, e.line.Size());
    EXPECT_EQ( 4, e.fb_wrapped_lines);
    EXPECT_EQ( 2, test_fb->back.size());
    EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,fh), test_fb->p);
    if (test_fb->back.size() > 0) EXPECT_EQ("2 2 2",       test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,5*fh), test_fb->back[0].p);
    if (test_fb->back.size() > 1) EXPECT_EQ("3",           test_fb->back[1].text);
    if (test_fb->back.size() > 1) EXPECT_EQ(0,             test_fb->back[1].wlo);
    if (test_fb->back.size() > 1) EXPECT_EQ(1,             test_fb->back[1].wll);
    if (test_fb->back.size() > 1) EXPECT_EQ(point(0,2*fh), test_fb->back[1].p);
    test_fb->back.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("2 2 2",          test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,3*fh),    test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[0].b);
    if (test_fb->paint.size() > 1) EXPECT_EQ("3",              test_fb->paint[1].text);
    if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),    test_fb->paint[1].p);
    if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[1].b);
    test_fb->paint.clear();

    // 4: 344 : 2=3:0, 1=4:0, 3=4:1
    e.v_scrolled = 4.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ( 0, e.start_line_adjust);
    EXPECT_EQ( 1, e.start_line_cutoff);
    EXPECT_EQ( 0, e.end_line_cutoff);
    EXPECT_EQ( 2, e.end_line_adjust);
    EXPECT_EQ( 2, e.line.Size());
    EXPECT_EQ( 3, e.fb_wrapped_lines);
    EXPECT_EQ( 1, test_fb->back.size());
    EXPECT_NEAR(1/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb->back.size() > 0) EXPECT_EQ("4 4",         test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(2,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,4*fh), test_fb->back[0].p);
    test_fb->back.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("4 4",           test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,fh),     test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
    if (test_fb->paint.size() > 1) EXPECT_EQ("4 4",           test_fb->paint[1].text);
    if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,4*fh),   test_fb->paint[1].p);
    if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
    test_fb->paint.clear();

    // 5: 445 : 1=4:0, 3=4:1, 2=5:0
    e.v_scrolled = 5.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ( 0, e.start_line_adjust);
    EXPECT_EQ( 2, e.start_line_cutoff);
    EXPECT_EQ( 0, e.end_line_cutoff);
    EXPECT_EQ( 1, e.end_line_adjust);
    EXPECT_EQ( 2, e.line.Size());
    EXPECT_EQ( 3, e.fb_wrapped_lines);
    EXPECT_EQ( 1, test_fb->back.size());
    EXPECT_NEAR(2/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,1*fh), test_fb->p);
    if (test_fb->back.size() > 0) EXPECT_EQ("5",           test_fb->back[0].text);
    if (test_fb->back.size() > 0) EXPECT_EQ(0,             test_fb->back[0].wlo);
    if (test_fb->back.size() > 0) EXPECT_EQ(1,             test_fb->back[0].wll);
    if (test_fb->back.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->back[0].p);
    test_fb->back.clear();
    EXPECT_EQ(1, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("5",           test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh), test_fb->paint[0].b);
    test_fb->paint.clear();

    // 3: 234 : 3=2:2, 2=3:0, 1=4:0
    e.v_scrolled = 3.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(-2, e.start_line_adjust);
    EXPECT_EQ( 1, e.start_line_cutoff);
    EXPECT_EQ( 1, e.end_line_cutoff);
    EXPECT_EQ( 1, e.end_line_adjust);
    EXPECT_EQ( 3, e.line.Size());
    EXPECT_EQ( 6, e.fb_wrapped_lines);
    EXPECT_EQ( 2, test_fb->front.size());
    EXPECT_NEAR(0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,3*fh), test_fb->p);
    if (test_fb->front.size() > 0) EXPECT_EQ("3",           test_fb->front[0].text);
    if (test_fb->front.size() > 0) EXPECT_EQ(0,             test_fb->front[0].wlo);
    if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wll);
    if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
    if (test_fb->front.size() > 1) EXPECT_EQ("2 2 2",       test_fb->front[1].text);
    if (test_fb->front.size() > 1) EXPECT_EQ(0,             test_fb->front[1].wlo);
    if (test_fb->front.size() > 1) EXPECT_EQ(1,             test_fb->front[1].wll);
    if (test_fb->front.size() > 1) EXPECT_EQ(point(0,5*fh), test_fb->front[1].p);
    test_fb->front.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("3",              test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,2*fh),    test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,fh),    test_fb->paint[0].b);
    if (test_fb->paint.size() > 1) EXPECT_EQ("2 2 2",          test_fb->paint[1].text);
    if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,3*fh),    test_fb->paint[1].p);
    if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,2*fh,w,fh), test_fb->paint[1].b);
    test_fb->paint.clear();

    // 1: 222 : 2=2:0, 1=2:1, 3=2:2
    e.v_scrolled = 1.0/(e.WrappedLines()-1);
    e.UpdateScrolled();
    EXPECT_EQ(0, e.start_line_adjust);
    EXPECT_EQ(3, e.start_line_cutoff);
    EXPECT_EQ(0, e.end_line_cutoff);
    EXPECT_EQ(3, e.end_line_adjust);
    EXPECT_EQ(1, e.line.Size());
    EXPECT_EQ(3, e.fb_wrapped_lines);
    EXPECT_EQ(1, test_fb->front.size());
    EXPECT_NEAR(-2/3.0, test_fb->scroll.y, 1e-6);
    EXPECT_EQ(point(0,2*fh), test_fb->p);
    if (test_fb->front.size() > 0) EXPECT_EQ("2 2 2",       test_fb->front[0].text);
    if (test_fb->front.size() > 0) EXPECT_EQ(1,             test_fb->front[0].wlo);
    if (test_fb->front.size() > 0) EXPECT_EQ(2,             test_fb->front[0].wll);
    if (test_fb->front.size() > 0) EXPECT_EQ(point(0,2*fh), test_fb->front[0].p);
    test_fb->back.clear();
    EXPECT_EQ(2, test_fb->paint.size());
    if (test_fb->paint.size() > 0) EXPECT_EQ("2 2 2",         test_fb->paint[0].text);
    if (test_fb->paint.size() > 0) EXPECT_EQ(point(0,5*fh),   test_fb->paint[0].p);
    if (test_fb->paint.size() > 0) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[0].b);
    if (test_fb->paint.size() > 1) EXPECT_EQ("2 2 2",         test_fb->paint[1].text);
    if (test_fb->paint.size() > 1) EXPECT_EQ(point(0,2*fh),   test_fb->paint[1].p);
    if (test_fb->paint.size() > 1) EXPECT_EQ(Box(0,0,w,2*fh), test_fb->paint[1].b);
    test_fb->paint.clear();
}

TEST(GUITest, Terminal) {
    int cursor_attr = 0;
    Terminal::Attr::SetFGColorIndex(&cursor_attr, 7);
    EXPECT_EQ(7,  Terminal::Attr::GetFGColorIndex(cursor_attr));
    Terminal::Attr::SetFGColorIndex(&cursor_attr, 2);
    EXPECT_EQ(2,  Terminal::Attr::GetFGColorIndex(cursor_attr));
    Terminal::Attr::SetBGColorIndex(&cursor_attr, 3);
    EXPECT_EQ(3,  Terminal::Attr::GetBGColorIndex(cursor_attr));
    EXPECT_EQ(2,  Terminal::Attr::GetFGColorIndex(cursor_attr));
    cursor_attr |= Terminal::Attr::Bold;
    EXPECT_EQ(10, Terminal::Attr::GetFGColorIndex(cursor_attr));
    EXPECT_EQ(3,  Terminal::Attr::GetBGColorIndex(cursor_attr));
    Terminal::Attr::SetFGColorIndex(&cursor_attr, 4);
    Terminal::Attr::SetBGColorIndex(&cursor_attr, 0);
    EXPECT_EQ(12, Terminal::Attr::GetFGColorIndex(cursor_attr));
    EXPECT_EQ(0,  Terminal::Attr::GetBGColorIndex(cursor_attr));
    cursor_attr &= ~Terminal::Attr::Bold;
    EXPECT_EQ(4,  Terminal::Attr::GetFGColorIndex(cursor_attr));
}

TEST(GUITest, LineTokenProcessor) {
    TextAreaTest ta(screen, Fonts::Fake(), 10);
    ta.token_processing = 1;
    TextGUI::Line *L = ta.line.InsertAt(-1);

    L->UpdateText(0, "a", 0);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(1, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ("a", ta.token[0].word); }
    ta.token.clear();

    L->UpdateText(1, "c", 0);
    EXPECT_EQ("ac", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -8); EXPECT_EQ("a",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("ac", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(1, "b", 0);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ("ac",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ("abc", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(0, "0", 0);
    EXPECT_EQ("0abc", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -9); EXPECT_EQ("abc",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ("0abc", ta.token[1].word); }
    ta.token.clear();

    L->Erase(0, 1);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -3); EXPECT_EQ("0abc", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   9); EXPECT_EQ("abc",  ta.token[1].word); }
    ta.token.clear();

    L->Erase(1, 1);
    EXPECT_EQ("ac", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -1); EXPECT_EQ("abc", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   7); EXPECT_EQ("ac",  ta.token[1].word); }
    ta.token.clear();

    L->Erase(1, 1);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -2); EXPECT_EQ("ac", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,   8); EXPECT_EQ("a",  ta.token[1].word); }
    ta.token.clear();

    L->Erase(0, 1);
    EXPECT_EQ("", L->Text()); EXPECT_EQ(1, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type,  -4); EXPECT_EQ("a", ta.token[0].word); }
    ta.token.clear();
    
    L->UpdateText(0, "aabb", 0);
    EXPECT_EQ("aabb", L->Text()); EXPECT_EQ(1, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ("aabb", ta.token[0].word); }
    ta.token.clear();

    L->UpdateText(2, " ", 0);
    EXPECT_EQ("aa bb", L->Text()); EXPECT_EQ(3, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ("aabb", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ("aa",   ta.token[1].word); }
    if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  6); EXPECT_EQ("bb",   ta.token[2].word); }
    ta.token.clear();

    L->Erase(2, 1);
    EXPECT_EQ("aabb", L->Text()); EXPECT_EQ(3, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("aa",   ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type, -6); EXPECT_EQ("bb",   ta.token[1].word); }
    if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  7); EXPECT_EQ("aabb", ta.token[2].word); }
    ta.token.clear();

    L->Clear();
    ta.insert_mode = 0;
    L->UpdateText(0, "a", 0);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(1, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ("a", ta.token[0].word); }
    ta.token.clear();

    L->UpdateText(1, "cc", 0);
    EXPECT_EQ("acc", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("a",   ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("acc", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(1, "b", 0);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -1); EXPECT_EQ("acc", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ("abc", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(3, "bb", 0);
    EXPECT_EQ("abcbb", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("abc",   ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("abcbb", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(0, "aab", 0);
    EXPECT_EQ("aabbb", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -3); EXPECT_EQ("abcbb", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ("aabbb", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(2, " ", 0);
    EXPECT_EQ("aa bb", L->Text()); EXPECT_EQ(3, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -1); EXPECT_EQ("aabbb", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ("aa",    ta.token[1].word); }
    if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  6); EXPECT_EQ("bb",    ta.token[2].word); }
    ta.token.clear();

    L->UpdateText(2, "c", 0);
    EXPECT_EQ("aacbb", L->Text()); EXPECT_EQ(3, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("aa",    ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type, -6); EXPECT_EQ("bb",    ta.token[1].word); }
    if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  1); EXPECT_EQ("aacbb", ta.token[2].word); }
    ta.token.clear();

    L->UpdateText(5, "ccdee", 0);
    EXPECT_EQ("aacbbccdee", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("aacbb",      ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("aacbbccdee", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(5, " ccdee", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(3, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ("aacbbccdee", ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  4); EXPECT_EQ("ccdee",      ta.token[1].word); }
    if (ta.token.size()>2) { EXPECT_EQ(ta.token[2].type,  5); EXPECT_EQ("aacbb",      ta.token[2].word); }
    ta.token.clear();

    L->UpdateText(5, " ccdee", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -4); EXPECT_EQ("ccdee",      ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  4); EXPECT_EQ("ccdee",      ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(5, " ", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(0, ta.token.size());

    L->OverwriteTextAt(10, StringPiece(string("Z")), 0);
    EXPECT_EQ("aacbb ccdeZ", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ("ccdee",      ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("ccdeZ",      ta.token[1].word); }
    ta.token.clear();

    L->OverwriteTextAt(10, StringPiece(string(" ")), 0);
    EXPECT_EQ("aacbb ccde ", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -2); EXPECT_EQ("ccdeZ",      ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  5); EXPECT_EQ("ccde",       ta.token[1].word); }
    ta.token.clear();

    L->OverwriteTextAt(10, StringPiece(string(" ")), 0);
    EXPECT_EQ("aacbb ccde ", L->Text()); EXPECT_EQ(0, ta.token.size());
    ta.token.clear();

    L->OverwriteTextAt(10, StringPiece(string("A")), 0);
    EXPECT_EQ("aacbb ccdeA", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -5); EXPECT_EQ("ccde",       ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("ccdeA",      ta.token[1].word); }
    ta.token.clear();

    // test wide chars
    L->Clear();
    ta.insert_mode = 1;
    L->UpdateText(0, "\xff", 0);
    EXPECT_EQ("\xff\xa0", L->Text()); EXPECT_EQ(1, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, 4); EXPECT_EQ("\xff\xa0", ta.token[0].word); }
    ta.token.clear();

    L->UpdateText(2, "y", 0);
    EXPECT_EQ("\xff\xa0y", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -8); EXPECT_EQ("\xff\xa0",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  2); EXPECT_EQ("\xff\xa0y", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(2, "x", 0);
    EXPECT_EQ("\xff\xa0xy", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -7); EXPECT_EQ("\xff\xa0y",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  1); EXPECT_EQ("\xff\xa0xy", ta.token[1].word); }
    ta.token.clear();

    L->UpdateText(0, "0", 0);
    EXPECT_EQ("0\xff\xa0xy", L->Text()); EXPECT_EQ(2, ta.token.size());
    if (ta.token.size()>0) { EXPECT_EQ(ta.token[0].type, -9); EXPECT_EQ("\xff\xa0xy",  ta.token[0].word); }
    if (ta.token.size()>1) { EXPECT_EQ(ta.token[1].type,  3); EXPECT_EQ("0\xff\xa0xy", ta.token[1].word); }
    ta.token.clear();
}

TEST(GUITest, TerminalTokenProcessor) {
    TerminalTest ta(NULL, screen, Fonts::Fake());
    ta.SetDimension(80, 25);
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
