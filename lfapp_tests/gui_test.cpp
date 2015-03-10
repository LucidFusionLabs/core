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
#include "lfapp/gui.h"
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
    LinesFrameBufferTest line_fb_test;
    struct UpdateSyntaxOp {
        Line *l; string word; int type;
        UpdateSyntaxOp(Line *L, const BoxRun &W, int T) : l(L), word(W.Text()), type(T) {}
    };
    vector<UpdateSyntaxOp> syntax;
    TextAreaTest(Window *W, Font *F, int S=200) : TextArea(W,F,S) {}
    virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
    virtual void UpdateSyntax(Line *l, const BoxRun &w, int t) override { syntax.emplace_back(l, w, t); }
};
struct EditorTest : public Editor {
    LinesFrameBufferTest line_fb_test;
    EditorTest(Window *W, Font *F, File *I, bool Wrap=0) : Editor(W,F,I,Wrap) { line_fb_test.wrap=Wrap; }
    virtual LinesFrameBuffer *GetFrameBuffer() override { return &line_fb_test; }
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
        int fh = font->height;
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
    int fh = font->height, w = font->fixed_width;
    EXPECT_NE(0, fh); EXPECT_NE(0, w);
    EditorTest e(screen, font, new BufferFile("1\n2 2 2\n3\n4 4\n5\n"), true);
    LinesFrameBufferTest *test_fb = &e.line_fb_test;
    Box b(w, 3*fh);
    e.Draw(b);

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

TEST(GUITest, LineSyntaxProcessor) {
    TextAreaTest ta(screen, Fonts::Fake(), 10);
    ta.syntax_processing = 1;
    TextGUI::Line *L = ta.line.InsertAt(-1);

    L->UpdateText(0, "a", 0);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(1, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, 4); EXPECT_EQ("a", ta.syntax[0].word); }
    ta.syntax.clear();

    L->UpdateText(1, "c", 0);
    EXPECT_EQ("ac", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -8); EXPECT_EQ("a",  ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  2); EXPECT_EQ("ac", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(1, "b", 0);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -7); EXPECT_EQ("ac",  ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  1); EXPECT_EQ("abc", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(0, "0", 0);
    EXPECT_EQ("0abc", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -9); EXPECT_EQ("abc",  ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  3); EXPECT_EQ("0abc", ta.syntax[1].word); }
    ta.syntax.clear();

    L->Erase(0, 1);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type,  -3); EXPECT_EQ("0abc", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,   9); EXPECT_EQ("abc",  ta.syntax[1].word); }
    ta.syntax.clear();

    L->Erase(1, 1);
    EXPECT_EQ("ac", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type,  -1); EXPECT_EQ("abc", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,   7); EXPECT_EQ("ac",  ta.syntax[1].word); }
    ta.syntax.clear();

    L->Erase(1, 1);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type,  -2); EXPECT_EQ("ac", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,   8); EXPECT_EQ("a",  ta.syntax[1].word); }
    ta.syntax.clear();

    L->Erase(0, 1);
    EXPECT_EQ("", L->Text()); EXPECT_EQ(1, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type,  -4); EXPECT_EQ("a", ta.syntax[0].word); }
    ta.syntax.clear();
    
    L->UpdateText(0, "aabb", 0);
    EXPECT_EQ("aabb", L->Text()); EXPECT_EQ(1, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, 4); EXPECT_EQ("aabb", ta.syntax[0].word); }
    ta.syntax.clear();

    L->UpdateText(2, " ", 0);
    EXPECT_EQ("aa bb", L->Text()); EXPECT_EQ(3, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -7); EXPECT_EQ("aabb", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  5); EXPECT_EQ("aa",   ta.syntax[1].word); }
    if (ta.syntax.size()>2) { EXPECT_EQ(ta.syntax[2].type,  6); EXPECT_EQ("bb",   ta.syntax[2].word); }
    ta.syntax.clear();

    L->Erase(2, 1);
    EXPECT_EQ("aabb", L->Text()); EXPECT_EQ(3, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -5); EXPECT_EQ("aa",   ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type, -6); EXPECT_EQ("bb",   ta.syntax[1].word); }
    if (ta.syntax.size()>2) { EXPECT_EQ(ta.syntax[2].type,  7); EXPECT_EQ("aabb", ta.syntax[2].word); }
    ta.syntax.clear();

    L->Clear();
    ta.insert_mode = 0;
    L->UpdateText(0, "a", 0);
    EXPECT_EQ("a", L->Text()); EXPECT_EQ(1, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, 4); EXPECT_EQ("a", ta.syntax[0].word); }
    ta.syntax.clear();

    L->UpdateText(1, "cc", 0);
    EXPECT_EQ("acc", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -5); EXPECT_EQ("a",   ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  2); EXPECT_EQ("acc", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(1, "b", 0);
    EXPECT_EQ("abc", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -1); EXPECT_EQ("acc", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  1); EXPECT_EQ("abc", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(3, "bb", 0);
    EXPECT_EQ("abcbb", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -5); EXPECT_EQ("abc",   ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  2); EXPECT_EQ("abcbb", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(0, "aab", 0);
    EXPECT_EQ("aabbb", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -3); EXPECT_EQ("abcbb", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  3); EXPECT_EQ("aabbb", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(2, " ", 0);
    EXPECT_EQ("aa bb", L->Text()); EXPECT_EQ(3, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -1); EXPECT_EQ("aabbb", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  5); EXPECT_EQ("aa",    ta.syntax[1].word); }
    if (ta.syntax.size()>2) { EXPECT_EQ(ta.syntax[2].type,  6); EXPECT_EQ("bb",    ta.syntax[2].word); }
    ta.syntax.clear();

    L->UpdateText(2, "c", 0);
    EXPECT_EQ("aacbb", L->Text()); EXPECT_EQ(3, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -5); EXPECT_EQ("aa",    ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type, -6); EXPECT_EQ("bb",    ta.syntax[1].word); }
    if (ta.syntax.size()>2) { EXPECT_EQ(ta.syntax[2].type,  1); EXPECT_EQ("aacbb", ta.syntax[2].word); }
    ta.syntax.clear();

    L->UpdateText(5, "ccdee", 0);
    EXPECT_EQ("aacbbccdee", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -5); EXPECT_EQ("aacbb",      ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  2); EXPECT_EQ("aacbbccdee", ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(5, " ccdee", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(3, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -2); EXPECT_EQ("aacbbccdee", ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  4); EXPECT_EQ("ccdee",      ta.syntax[1].word); }
    if (ta.syntax.size()>2) { EXPECT_EQ(ta.syntax[2].type,  5); EXPECT_EQ("aacbb",      ta.syntax[2].word); }
    ta.syntax.clear();

    L->UpdateText(5, " ccdee", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(2, ta.syntax.size());
    if (ta.syntax.size()>0) { EXPECT_EQ(ta.syntax[0].type, -4); EXPECT_EQ("ccdee",      ta.syntax[0].word); }
    if (ta.syntax.size()>1) { EXPECT_EQ(ta.syntax[1].type,  4); EXPECT_EQ("ccdee",      ta.syntax[1].word); }
    ta.syntax.clear();

    L->UpdateText(5, " ", 0);
    EXPECT_EQ("aacbb ccdee", L->Text()); EXPECT_EQ(0, ta.syntax.size());
}

TEST(BrowserTest, DOMNode) {
    LFL::DOM::FixedObjectAlloc<65536> alloc;
    LFL::DOM::HTMLDocument *doc = AllocatorNew(&alloc, (LFL::DOM::HTMLDocument), (0, &alloc, 0));
    LFL::DOM::Text A(doc), B(doc), C(doc), D(doc);
    EXPECT_EQ(0, doc->firstChild()); EXPECT_EQ(0, doc->lastChild());

    EXPECT_EQ(0, A.parentNode);
    doc->appendChild(&A);
    EXPECT_EQ(doc, A.parentNode);
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(0,  A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&A, doc->lastChild());

    EXPECT_EQ(0, B.parentNode);
    doc->appendChild(&B);
    EXPECT_EQ(doc, B.parentNode);
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(0,  B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&B, doc->lastChild());

    EXPECT_EQ(0, C.parentNode);
    doc->appendChild(&C);
    EXPECT_EQ(doc, C.parentNode);
    EXPECT_EQ(&B,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(&C, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    EXPECT_EQ(0, D.parentNode);
    doc->insertBefore(&D, &C);
    EXPECT_EQ(doc, D.parentNode);
    EXPECT_EQ(&B,  D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(&D, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    doc->removeChild(&B);
    EXPECT_EQ(0, B.parentNode);
    EXPECT_EQ(&A, D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D, C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(0,  B.previousSibling()); EXPECT_EQ(0,  B.nextSibling());
    EXPECT_EQ(0,  A.previousSibling()); EXPECT_EQ(&D, A.nextSibling());
    EXPECT_EQ(&A, doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    doc->replaceChild(&B, &A);
    EXPECT_EQ(doc, B.parentNode);
    EXPECT_EQ(0,   A.parentNode);
    EXPECT_EQ(&B,  D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(0,   B.previousSibling()); EXPECT_EQ(&D, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(0,  A.nextSibling());
    EXPECT_EQ(&B,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());
}

TEST(BrowserTest, DOM) {
    Browser sb(screen, screen->Box());
    sb.doc.parser->OpenHTML("<html>\n"
                            "<head><style> h1 { background-color: #111111; } </style></head>\n"
                            "<body style=\"background-color: #00ff00\">\n"
                            "<H1 class=\"foo  bar\">Very header</h1>\n"
                            "<P id=cat>In  the  begining  was  fun.</p>\n"
                            "</body>\n");
    Box viewport = screen->Box();
    sb.doc.v_scrollbar.menuicon2 = Fonts::Fake();
    sb.doc.h_scrollbar.menuicon2 = Fonts::Fake();
    sb.Draw(&viewport);
    CHECK(sb.doc.node);
    EXPECT_EQ("#document", String::ToUTF8(sb.doc.node->nodeName()));
    EXPECT_EQ(1, sb.doc.node->childNodes.length());
    EXPECT_EQ(0, sb.doc.node->AsHTMLImageElement());
    EXPECT_EQ(sb.doc.node, sb.doc.node->AsHTMLDocument());

    LFL::DOM::Node *html_node = sb.doc.node->firstChild();
    EXPECT_EQ("html", String::ToUTF8(html_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, html_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_HTML_ELEMENT, html_node->htmlElementType);
    LFL::DOM::HTMLElement *html_element = (LFL::DOM::HTMLElement*)html_node;
    EXPECT_EQ(html_node->AsElement(), html_element);
    EXPECT_EQ("html", String::ToUTF8(html_element->tagName));
    EXPECT_EQ(2, html_element->childNodes.length());
#ifdef LFL_LIBCSS
    void *vnode = 0; bool matched = 0; int count = 0;
    css_qname html_qn = { 0, 0 }, html_query = { 0, LibCSS_String::Intern("html") };
    css_qname foo_query = { 0, LibCSS_String::Intern("foo") };
    StyleContext::node_name(0, html_node, &html_qn);
    EXPECT_EQ("html", LibCSS_String::ToUTF8String(html_qn.name));
    lwc_string **html_classes = 0; uint32_t n_html_classes = 0;
    StyleContext::node_classes(0, html_node, &html_classes, &n_html_classes);
    EXPECT_EQ(0, n_html_classes);
    EXPECT_EQ(0, html_classes);
    lwc_string *html_id = 0;
    StyleContext::node_id(0, html_node, &html_id);
    EXPECT_EQ(0, html_id);
    StyleContext::node_is_root(0, html_node, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::named_parent_node(0, html_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::node_has_name(0, html_node, &html_query, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_name(0, html_node, &foo_query, &matched);
    EXPECT_EQ(0, matched);
    CHECK(html_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, html_node->render->style.Display().v);
#endif

    LFL::DOM::Node *head_node = html_element->firstChild();
    EXPECT_EQ("head", String::ToUTF8(head_node->nodeName()));

    LFL::DOM::Node *body_node = html_element->childNodes.item(1);
    EXPECT_EQ("body", String::ToUTF8(body_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, body_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_BODY_ELEMENT, body_node->htmlElementType);
    LFL::DOM::HTMLBodyElement *body_element = (LFL::DOM::HTMLBodyElement*)body_node;
    EXPECT_EQ(body_node->AsElement(), body_element);
    EXPECT_EQ(body_node->AsHTMLBodyElement(), body_element);
    EXPECT_EQ("body", String::ToUTF8(body_element->tagName));
    EXPECT_EQ("background-color: #00ff00", String::ToUTF8(body_element->getAttribute("style")));
    EXPECT_EQ(2, body_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname body_qn = { 0, 0 }, body_query = { 0, LibCSS_String::Intern("body") };
    css_qname style_query = { 0, LibCSS_String::Intern("style") };
    StyleContext::node_name(0, body_node, &body_qn);
    EXPECT_EQ("body", LibCSS_String::ToUTF8String(body_qn.name));
    StyleContext::node_is_root(0, body_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::named_ancestor_node(0, body_node, &html_query, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::named_parent_node(0, body_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::named_parent_node(0, body_node, &html_query, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::parent_node(0, body_node, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::sibling_node(0, body_node, &vnode);
    EXPECT_EQ(head_node, vnode);
    StyleContext::node_has_attribute(0, body_node, &foo_query, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute(0, body_node, &style_query, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_equal(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_equal(0, body_node, &style_query, LibCSS_String::Intern("background-color: #00ff00"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_dashmatch(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_dashmatch(0, body_node, &style_query, LibCSS_String::Intern("background-"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_includes(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_includes(0, body_node, &style_query, LibCSS_String::Intern("background-color:"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_prefix(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_prefix(0, body_node, &style_query, LibCSS_String::Intern("background-color: #0"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_suffix(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_suffix(0, body_node, &style_query, LibCSS_String::Intern("0ff00"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_attribute_substring(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_attribute_substring(0, body_node, &style_query, LibCSS_String::Intern("ground-color"), &matched);
    EXPECT_EQ(1, matched);
    CHECK(body_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, body_node->render->style.Display().v);
    EXPECT_EQ("00FF00", String::ToUTF8(body_node->render->style.BackgroundColor().cssText()));
#endif

    LFL::DOM::Node *h_node = body_element->firstChild();
    EXPECT_EQ("h1", String::ToUTF8(h_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, h_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_HEADING_ELEMENT, h_node->htmlElementType);
    LFL::DOM::HTMLHeadingElement *h_element = (LFL::DOM::HTMLHeadingElement*)h_node;
    EXPECT_EQ(h_node->AsElement(), h_element);
    EXPECT_EQ(h_node->AsHTMLHeadingElement(), h_element);
    EXPECT_EQ("h1", String::ToUTF8(h_element->tagName));
    EXPECT_EQ("foo  bar", String::ToUTF8(h_element->getAttribute("class")));
    EXPECT_EQ(1, h_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname h_qn = { 0, 0 }, h_query = { 0, LibCSS_String::Intern("h1") }, p_query = { 0, LibCSS_String::Intern("p") };
    css_qname bar_query = { 0, LibCSS_String::Intern("bar") }, cat_query = { 0, LibCSS_String::Intern("cat") };
    StyleContext::node_name(0, h_node, &h_qn);
    EXPECT_EQ("h1", LibCSS_String::ToUTF8String(h_qn.name));
    lwc_string **h_classes = 0; uint32_t n_h_classes = 0;
    StyleContext::node_classes(0, h_node, &h_classes, &n_h_classes);
    EXPECT_EQ(2, n_h_classes);
    EXPECT_EQ("foo", LibCSS_String::ToUTF8String(h_classes[0]));
    EXPECT_EQ("bar", LibCSS_String::ToUTF8String(h_classes[1]));
    StyleContext::node_is_root(0, h_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::sibling_node(0, h_node, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::named_sibling_node(0, h_node, &p_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::node_has_class(0, h_node, foo_query.name, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_has_class(0, h_node, cat_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_class(0, h_node, bar_query.name, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::node_count_siblings(0, h_node, false, false, &count);
    EXPECT_EQ(0, count);
    StyleContext::node_count_siblings(0, h_node, false, true, &count);
    EXPECT_EQ(1, count);
    StyleContext::node_is_empty(0, h_node, &matched);
    EXPECT_EQ(0, matched);
    CHECK(h_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, h_node->render->style.Display().v);
    EXPECT_EQ("000000", String::ToUTF8(h_node->render->style.Color().cssText()));
    EXPECT_EQ("111111", String::ToUTF8(h_node->render->style.BackgroundColor().cssText()));
#endif

    LFL::DOM::Node *h1_text = h_element->firstChild();
    EXPECT_EQ(LFL::DOM::TEXT_NODE, h1_text->nodeType);
    EXPECT_EQ("#text", String::ToUTF8(h1_text->nodeName()));
    EXPECT_EQ("Very header", String::ToUTF8(h1_text->nodeValue()));
    CHECK_NE(h1_text->AsText(), 0);
#ifdef LFL_LIBCSS
    StyleContext::node_is_empty(0, h1_text, &matched);
    EXPECT_EQ(1, matched);
#endif

    LFL::DOM::Node *p_node = body_element->childNodes.item(1);
    EXPECT_EQ("p", String::ToUTF8(p_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, p_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_PARAGRAPH_ELEMENT, p_node->htmlElementType);
    LFL::DOM::HTMLParagraphElement *p_element = (LFL::DOM::HTMLParagraphElement*)p_node;
    EXPECT_EQ(p_node->AsElement(), p_element);
    EXPECT_EQ(p_node->AsHTMLParagraphElement(), p_element);
    EXPECT_EQ("p", String::ToUTF8(p_element->tagName));
    EXPECT_EQ("cat", String::ToUTF8(p_element->getAttribute("id")));
    EXPECT_EQ(1, p_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname p_qn = { 0, 0 };
    StyleContext::node_name(0, p_node, &p_qn);
    EXPECT_EQ("p", LibCSS_String::ToUTF8String(p_qn.name));
    StyleContext::node_is_root(0, p_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::sibling_node(0, p_node, &vnode);
    EXPECT_EQ(h_node, vnode);
    StyleContext::named_sibling_node(0, p_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::named_sibling_node(0, p_node, &h_query, &vnode);
    EXPECT_EQ(h_node, vnode);
    StyleContext::node_has_id(0, p_node, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::node_has_id(0, p_node, cat_query.name, &matched);
    EXPECT_EQ(1, matched);
#endif

    LFL::DOM::Node *p_text = p_element->firstChild();
    EXPECT_EQ(LFL::DOM::TEXT_NODE, p_text->nodeType);
    EXPECT_EQ("#text", String::ToUTF8(p_text->nodeName()));
    EXPECT_EQ("In the begining was fun.", String::ToUTF8(p_text->nodeValue()));

    LFL::DOM::NodeList node_list = sb.doc.node->getElementsByTagName("html");
    EXPECT_EQ(1, node_list.length());
    EXPECT_EQ(html_node, node_list.item(0));
    node_list.clear();
    node_list = body_element->getElementsByTagName("h1");
    EXPECT_EQ(1, node_list.length());
    EXPECT_EQ(h_node, node_list.item(0));
}

#ifdef LFL_LIBCSS
#include "lfapp/css.h"
TEST(BrowserTest, CSS) {
    StyleSheet sheet(0, 0);
    sheet.Parse("h1 { color: red }\n"
                "em { color: blue }\n"
                "h1 em { color: green }");
    sheet.Done();
}
#endif
}; // namespace LFL
