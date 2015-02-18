/*
 * $Id: gui.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "../crawler/html.h"

#ifndef _WIN32
#include <sys/ioctl.h>
#include <termios.h>
#endif

#ifdef LFL_QT
#include <QWindow>
#include <QWebView>
#include <QWebFrame>
#include <QWebElement>
#include <QMessageBox>
#include <QMouseEvent>
#include <QApplication>
#endif

#ifdef LFL_BERKELIUM
#include "berkelium/Berkelium.hpp"
#include "berkelium/Window.hpp"
#include "berkelium/WindowDelegate.hpp"
#include "berkelium/Context.hpp"
#endif

namespace LFL {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
DEFINE_bool(multitouch, true, "Touchscreen controls");
#else
DEFINE_bool(multitouch, false, "Touchscreen controls");
#endif
DEFINE_bool(draw_grid, false, "Draw lines intersecting mouse x,y");

Window::WindowMap Window::active;
Window *Window::Get(void *id) { return FindOrNull(Window::active, id); }

Window::Window() : caption("lfapp"), fps(128) {
    id = gl = surface = glew_context = user1 = user2 = user3 = 0;
    minimized = cursor_grabbed = 0;
    opengles_version = 1;
    opengles_cubemap = 0;
    width = 640; height = 480;
    multitouch_keyboard_x = .93; 
    cam = new Entity(v3(5.54, 1.70, 4.39), v3(-.51, -.03, -.49), v3(-.03, 1, -.03));
    ClearEvents();
    ClearGesture();
}

Window::~Window() {
    if (console) {
        console->WriteHistory(LFAppDownloadDir(), "console");
        delete console;
    }
    delete cam;
}

void Window::ClearMouseGUIEvents() {
    for (auto i = mouse_gui.begin(); i != mouse_gui.end(); ++i) (*i)->ClearEvents();
}
void Window::ClearKeyboardGUIEvents() {
    for (auto i = keyboard_gui.begin(); i != keyboard_gui.end(); ++i) (*i)->ClearEvents();
}
void Window::ClearInputBindEvents() {
    for (auto i = input_bind.begin(); i != input_bind.end(); ++i) (*i)->ClearEvents();
}

void Window::ClearEvents() { 
    memzero(events);
    ClearMouseGUIEvents();
    ClearKeyboardGUIEvents();
    ClearInputBindEvents();
}

void Window::ClearGesture() {
    gesture_swipe_up = gesture_swipe_down = 0;
    gesture_tap[0] = gesture_tap[1] = gesture_dpad_stop[0] = gesture_dpad_stop[1] = 0;
    gesture_dpad_dx[0] = gesture_dpad_dx[1] = gesture_dpad_dy[0] = gesture_dpad_dy[1] = 0;
}

void Window::InitConsole() {
    console = new Console(screen, Fonts::Get(FLAGS_default_font, 9, Color::white));
    console->ReadHistory(LFAppDownloadDir(), "console");
    console->Write(StrCat(screen->caption, " started"));
    console->Write("Try console commands 'cmds' and 'flags'");
}

void Window::DrawDialogs() {
    if (screen->console) screen->console->Draw();
    if (FLAGS_draw_grid) {
        Color c(.7, .7, .7);
        glIntersect(screen->mouse.x, screen->mouse.y, &c);
        Fonts::Default()->Draw(StrCat("draw_grid ", screen->mouse.x, " , ", screen->mouse.y), point(0,0));
    }
    for (auto i = screen->dialogs.rbegin(); i != screen->dialogs.rend(); ++i) (*i)->Draw();
}

void KeyboardGUI::AddHistory(const string &cmd) {
    lastcmd.ring.PushBack(1);
    lastcmd[(lastcmd_ind = -1)] = cmd;
}

int KeyboardGUI::WriteHistory(const string &dir, const string &name, const string &hdr) {
    if (!lastcmd.Size()) return 0;
    LocalFile history(dir + MatrixFile::Filename(name, "history", "string", 0), "w");
    MatrixFile::WriteHeader(&history, basename(history.fn.c_str(),0,0), hdr, lastcmd.ring.count, 1);
    for (int i=0; i<lastcmd.ring.count; i++) StringFile::WriteRow(&history, lastcmd[-1-i]);
    return 0;
}

int KeyboardGUI::ReadHistory(const string &dir, const string &name) {
    StringFile history;
    VersionedFileName vfn(dir.c_str(), name.c_str(), "history");
    if (history.ReadVersioned(vfn) < 0) { ERROR("missing ", name); return -1; }
    for (int i=0, l=history.Lines(); i<l; i++) AddHistory((*history.F)[l-1-i]);
    return 0;
}

void TextGUI::Enter() {
    string cmd = Text();
    AssignInput("");
    if (!cmd.empty()) { AddHistory(cmd); Run(cmd); }
    TouchDevice::CloseKeyboard();
    if (deactivate_on_enter) active = false;
}

void TextGUI::Draw(const Box &b) {
    if (cmd_fb.SizeChanged(b.w, b.h, font)) {
        cmd_fb.p = point(0, font->height);
        cmd_line.Draw(point(0, cmd_line.Lines() * font->height), true, cmd_fb.w);
        cmd_fb.SizeChangedDone();
    }
    // screen->gd->PushColor();
    // screen->gd->SetColor(cmd_color);
    cmd_fb.Draw(b.Position(), point());
    DrawCursor(b.Position() + cursor.p);
    // screen->gd->PopColor();
}

void TextGUI::DrawCursor(point p) {
    if (cursor.type == Cursor::Block) {
        screen->gd->EnableBlend();
        screen->gd->BlendMode(GraphicsDevice::OneMinusDstColor, GraphicsDevice::OneMinusSrcAlpha);
        screen->gd->FillColor(cmd_color);
        Box(p.x, p.y - font->height, font->max_width, font->height).Draw();
        screen->gd->BlendMode(GraphicsDevice::SrcAlpha, GraphicsDevice::One);
        screen->gd->DisableBlend();
    } else {
        bool blinking = false;
        Time now = Now(), elapsed; 
        if (active && (elapsed = now - cursor.blink_begin) > cursor.blink_time) {
            if (elapsed > cursor.blink_time * 2) cursor.blink_begin = now;
            else blinking = true;
        }
        if (blinking) font->Draw("_", p);
    }
}

TextGUI::LinesFrameBuffer *TextGUI::LinesFrameBuffer::Attach(TextGUI::LinesFrameBuffer **last_fb) {
    if (*last_fb != this) fb.Attach();
    return *last_fb = this;
}

bool TextGUI::LinesFrameBuffer::SizeChanged(int W, int H, Font *font) {
    lines = H / font->height;
    return RingFrameBuffer::SizeChanged(W, H, font);
}

void TextGUI::LinesFrameBuffer::Update(TextGUI::Line *l, int flag) {
    if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
    RingFrameBuffer::Update(l, Box(0, l->Lines() * font_height), paint_cb, true);
}

int TextGUI::LinesFrameBuffer::PushFrontAndUpdate(TextGUI::Line *l, int xo, int wlo, int wll, int flag) {
    if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
    int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
    if (!lh) return 0; Box b(xo, wl * font_height - lh, 0, lh);
    return RingFrameBuffer::PushFrontAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

int TextGUI::LinesFrameBuffer::PushBackAndUpdate(TextGUI::Line *l, int xo, int wlo, int wll, int flag) {
    if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
    int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
    if (!lh) return 0; Box b(xo, wlo * font_height, 0, lh);
    return RingFrameBuffer::PushBackAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

void TextGUI::LinesFrameBuffer::PushFrontAndUpdateOffset(TextGUI::Line *l, int lo) {
    Update(l, RingFrameBuffer::BackPlus(point(0, (1 + lo) * font_height)));
    RingFrameBuffer::AdvancePixels(-l->Lines() * font_height);
}

void TextGUI::LinesFrameBuffer::PushBackAndUpdateOffset(TextGUI::Line *l, int lo) {
    Update(l, RingFrameBuffer::BackPlus(point(0, lo * font_height)));
    RingFrameBuffer::AdvancePixels(l->Lines() * font_height);
}

point TextGUI::LinesFrameBuffer::Paint(TextGUI::Line *l, point lp, const Box &b) {
    Scissor scissor(0, lp.y - b.h, w, b.h);
    screen->gd->Clear();
    l->Draw(lp + b.Position(), 0, 0);
    return point(lp.x, lp.y-b.h);
}

/* TextArea */

void TextArea::Write(const string &s, bool update_fb, bool release_fb) {
    if (!MainThread()) return RunInMainThread(new Callback(bind(&TextArea::Write, this, s, update_fb, release_fb)));
    write_last = Now();
    int update_flag = LineFBPushBack();
    LinesFrameBuffer *fb = GetFrameBuffer();
    if (update_fb && fb->lines) fb->fb.Attach();
    ScopedDrawMode drawmode(update_fb ? DrawMode::_2D : DrawMode::NullOp);
    StringLineIter add_lines(s, StringLineIter::Flag::BlankLines);
    for (const char *add_line = add_lines.Next(); add_line; add_line = add_lines.Next()) {
        bool append = !write_newline && add_lines.first && *add_line && line.ring.count;
        Line *l = append ? &line[-1] : line.InsertAt(-1);
        if (!append) {
            l->Clear();
            if (start_line) { start_line++; end_line++; }
        }
        if (write_timestamp) l->AppendText(StrCat(logtime(Now()), " "), cursor.attr);
        l->AppendText(add_line, cursor.attr);
        l->Layout(fb->w);
        if (scrolled_lines) v_scrolled = (float)++scrolled_lines / (WrappedLines()-1);
        if (!update_fb || start_line) continue;
        LineUpdate(&line[-start_line-1], fb, (!append ? update_flag : 0));
    }
    if (update_fb && release_fb && fb->lines) fb->fb.Release();
}

void TextArea::Resized(int w, int h) {
    if (selection.enabled) {
        mouse_gui.Clear();
        mouse_gui.Activate();
        mouse_gui.AddDragBox(Box(w,h), MouseController::CoordCB(bind(&TextArea::ClickCB, this, _1, _2, _3, _4)));
    }
    UpdateLines(last_v_scrolled, 0, 0, 0);
    Redraw(false);
}

void TextArea::Redraw(bool attach) {
    ScopedDrawMode drawmode(DrawMode::_2D);
    LinesFrameBuffer *fb = GetFrameBuffer();
    int fb_flag = LinesFrameBuffer::Flag::NoVWrap | LinesFrameBuffer::Flag::Flush;
    int lines = start_line_adjust + skip_last_lines;
    int (LinesFrameBuffer::*update_cb)(Line*, int, int, int, int) =
        reverse_line_fb ? &LinesFrameBuffer::PushBackAndUpdate
                        : &LinesFrameBuffer::PushFrontAndUpdate;
    fb->p = reverse_line_fb ? point(0, fb->Height() - start_line_adjust * font->height)
                            : point(0, start_line_adjust * font->height);
    if (attach) { fb->fb.Attach(); screen->gd->Clear(); }
    for (int i=start_line; i<line.ring.count && lines < fb->lines; i++)
        lines += (fb->*update_cb)(&line[-i-1], -line_left, 0, fb->lines - lines, fb_flag);
    fb->p = point(0, fb->Height());
    if (attach) { fb->scroll = v2(); fb->fb.Release(); }
}

int TextArea::UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) {
    LinesFrameBuffer *fb = GetFrameBuffer();
    pair<int, int> old_first_line(start_line, -start_line_adjust), new_first_line, new_last_line;
    FlattenedArrayValues<TextGUI::Lines>
        flattened_lines(&line, line.Size(), bind(&TextArea::LayoutBackLine, this, _1, _2));
    flattened_lines.AdvanceIter(&new_first_line, (scrolled_lines = v_scrolled * (WrappedLines()-1)));
    flattened_lines.AdvanceIter(&(new_last_line = new_first_line), fb->lines-1);
    LayoutBackLine(&line, new_last_line.first);
    bool up = new_first_line < old_first_line;
    int dist = flattened_lines.Distance(new_first_line, old_first_line, fb->lines-1);
    if (first_offset) *first_offset = up ?  start_line_cutoff+1 :  end_line_adjust+1;
    if (first_ind)    *first_ind    = up ? -start_line-1        : -end_line-1;
    if (first_len)    *first_len    = up ? -start_line_adjust   :  end_line_cutoff;
    start_line        =  new_first_line.first;
    start_line_adjust = -new_first_line.second;
    end_line          =  new_last_line.first;
    end_line_adjust   =  new_last_line.second;
    end_line_cutoff   = line[  -end_line-1].Lines() -  new_last_line.second - 1;
    start_line_cutoff = line[-start_line-1].Lines() - new_first_line.second - 1;
    return dist * (up ? -1 : 1);
}

void TextArea::UpdateScrolled() {
    int max_w = 4000;
    bool h_changed = Wrap() ? 0 : Typed::EqualChanged(&last_h_scrolled, h_scrolled);
    bool v_updated=0, v_changed = Typed::EqualChanged(&last_v_scrolled, v_scrolled);
    if (v_changed) {
        int first_ind = 0, first_offset = 0, first_len = 0;
        int dist = UpdateLines(v_scrolled, &first_ind, &first_offset, &first_len);
        if ((v_updated = dist)) {
            if (h_changed) UpdateHScrolled(max_w * h_scrolled, false);
            if (1)         UpdateVScrolled(abs(dist), dist<0, first_ind, first_offset, first_len);
        }
    }
    if (h_changed && !v_updated) UpdateHScrolled(max_w * h_scrolled, true);
}

void TextArea::UpdateHScrolled(int x, bool update_fb) {
    line_left = x;
    if (!update_fb) return;
    Redraw(true);
}

void TextArea::UpdateVScrolled(int dist, bool up, int ind, int first_offset, int first_len) {
    LinesFrameBuffer *fb = GetFrameBuffer();
    if (dist >= fb->lines) Redraw(true);
    else {
        bool front = up == reverse_line_fb, decr = front != reverse_line_fb;
        int wl = 0, (LinesFrameBuffer::*update_cb)(Line*, int, int, int, int) =
            front ? &LinesFrameBuffer::PushFrontAndUpdate : &LinesFrameBuffer::PushBackAndUpdate;
        ScopedDrawMode drawmode(DrawMode::_2D);
        fb->fb.Attach();
        if (first_len)  wl += (fb->*update_cb)(&line[ind], -line_left, first_offset, min(dist, first_len), 0); 
        while (wl<dist) wl += (fb->*update_cb)(&line[decr ? --ind : ++ind], -line_left, 0, dist-wl, 0);
        fb->fb.Release();
    }
}

void TextArea::Draw(const Box &b, bool draw_cursor) {
    LinesFrameBuffer *fb = GetFrameBuffer();
    if (fb->SizeChanged(b.w, b.h, font)) { Resized(b.w, b.h); fb->SizeChangedDone(); }
    if (clip) screen->gd->PushScissor(Box::DelBorder(b, *clip));
    fb->Draw(b.Position(), point(0, CommandLines() * font->height));
    if (clip) screen->gd->PopScissor();
    if (draw_cursor) TextGUI::Draw(Box(b.x, b.y, b.w, font->height));
    if (selection.changing) DrawSelection();
#if 0
        vector<Link*> hover_link;
        Link *link = 0;
        if (cur_attr & Attr::Link) {
            CHECK_LT(link_ind, l->links.size());
            link = l->links[link_ind].v;
            if (link->widget.hover) underline = true;
            link->widget.Activate();
        }
        if (link) {
            link->widget.win = l->win[0];
            if (link->widget.hover && line.hover_link_cb) hover_link.push_back(link);
            link_ind++;
        }
        mouse_gui.activate();
        for (int i=0; i<hover_link.size(); i++) hover_link_cb(hover_link[i]);
#endif
}

void TextArea::DrawSelection() {
    screen->gd->EnableBlend();
    screen->gd->FillColor(selection_color);
    Box win(line_fb.Width(), line_fb.Height()), gb = selection.beg.glyph, ge = selection.end.glyph;
    int fh = font->height, sgex = ge.x + ge.w;
    if (ge.w == 0) ge.x += win.w;
#if 1
    if (gb.y == ge.y) {
        Box(gb.x, gb.y, sgex - gb.x, fh)                       .Draw();
    } else {
        Box::AddBorder(Box(gb.x, gb.y, win.w - gb.x, fh), 2, 1).Draw();
        Box(0, ge.y + fh, win.w, gb.y - ge.y - fh)             .Draw();
        Box::AddBorder(Box(0, ge.y, sgex, font->height), 2, 1) .Draw();
    }
#else
    Box3 box3(&box3, &win, gb.Position(), point(sgex, ge.y), fh,
              gb.y - ge.y - fh, font->height, selection.end.line_ind - selection.beg.line_ind).glWindow(&c);
#endif
}

void TextArea::ClickCB(int button, int x, int y, int down) {
    if (!(selection.changing = down)) {
        INFO(button, " wanna copy text ", selection.beg.DebugString(), " ", selection.end.DebugString());
        CopyText(selection.beg, selection.end);
        selection.changing_previously=0;
        return;
    }
    if (selection.changing_previously) GetGlyphFromCoords((selection.end.click = point(x,y)), &selection.end);
    else                             { GetGlyphFromCoords((selection.beg.click = point(x,y)), &selection.beg); selection.end = selection.beg; }
    selection.changing_previously = selection.changing;
}

bool TextArea::GetGlyphFromCoords(const point &p, Selection::Point *out) {
    LinesFrameBuffer *fb = GetFrameBuffer();
    int targ = reverse_line_fb ? ((fb->Height() - p.y) / font->height - start_line_adjust)
                               : (p.y                  / font->height + start_line_adjust);
    for (int i=start_line, lines=0, ll; i<line.ring.count && lines<line_fb.lines; i++, lines += ll) {
        Line *L = &line[-i-1];
        if (lines + (ll = L->Lines()) < targ) continue;
        L->data->glyphs.GetGlyphFromCoords(p, &out->char_ind, &out->glyph, targ - lines);
        out->line_ind = i;
        out->glyph += L->p;
        return true;
    }
    //printf("DrawOrCopy %s %s %d %d\n", selection.beg_click.DebugString().c_str(), selection.end_point.DebugString().c_str(), found_beg, found_end);
    return false;
}

void TextArea::CopyText(const Selection::Point &beg, const Selection::Point &end) {
    string copy_text;
    for (int i = beg.line_ind; i <= end.line_ind; i++) {
        Line *l = &line[-i-1];
        int len = l->Size();
        if (i == beg.line_ind) {
            if (!l->Size() || beg.char_ind < 0) continue;
            len = (beg.line_ind == end.line_ind && end.char_ind >= 0) ? end.char_ind+1 : l->Size();
            copy_text += l->Text().substr(beg.char_ind, max(0, len - beg.char_ind));
        } else if (i == end.line_ind) {
            len =                                 (end.char_ind >= 0) ? end.char_ind+1 : l->Size();
            copy_text += l->Text().substr(0, len);
        } else {
            copy_text += l->Text();
        }
        if (len == l->Size()) copy_text += "\n";
    }
    if (!copy_text.empty()) Clipboard::Set(copy_text);
}

/* Editor */

void Editor::UpdateWrappedLines(int cur_font_size, int width) {
    wrapped_lines = 0;
    file_line.Clear();
    file->Reset();
    int offset = 0, wrap = Wrap(), ll;
    for (const char *l = file->NextLineRaw(&offset); l; l = file->NextLineRaw(&offset)) {
        wrapped_lines += (ll = wrap ? TextArea::font->Lines(l, width) : 1);
        file_line.val.Insert(LineOffset(offset, file->nr.record_len, ll));
    }
    file_line.LoadFromSortedVal();
}

int Editor::UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) {
    LinesFrameBuffer *fb = GetFrameBuffer();
    bool width_changed = last_fb_width != fb->w, wrap = Wrap();
    if (width_changed) {
        last_fb_width = fb->w;
        if (wrap || !wrapped_lines) UpdateWrappedLines(TextArea::font->size, fb->w);
    }

    bool resized = (width_changed && wrap) || last_fb_lines != fb->lines;
    if (resized) { line.Clear(); fb_wrapped_lines = 0; }

    int new_first_line = v_scrolled * (wrapped_lines - 1), new_last_line = new_first_line + fb->lines;
    int dist = resized ? fb->lines : abs(new_first_line - last_first_line), read_len = 0, bo = 0, l, e;
    if (!dist || !file_line.size()) return 0;

    bool up = !resized && new_first_line < last_first_line;
    if (first_offset) *first_offset = up ?  start_line_cutoff : end_line_adjust;
    if (first_len)    *first_len    = up ? -start_line_adjust : end_line_cutoff;

    pair<int, int> read_lines;
    if (dist < fb->lines) {
        if (up) read_lines = pair<int, int>(new_first_line, dist);
        else    read_lines = pair<int, int>(new_first_line + fb->lines - dist, dist);
    } else      read_lines = pair<int, int>(new_first_line, fb->lines);

    bool head_read = new_first_line == read_lines.first;
    bool tail_read = new_last_line  == read_lines.first + read_lines.second;
    bool short_read = !(head_read && tail_read), shorten_read = short_read && head_read && start_line_adjust;
    int past_end_lines = max(0, min(dist, read_lines.first + read_lines.second - wrapped_lines)), added = 0;
    read_lines.second = max(0, read_lines.second - past_end_lines);

    if      ( up && dist <= -start_line_adjust) { start_line_adjust += dist; read_lines.second=past_end_lines=0; }
    else if (!up && dist <=  end_line_cutoff)   { end_line_cutoff   -= dist; read_lines.second=past_end_lines=0; }

    LineMap::ConstIterator lib, lie;
    if (read_lines.second) {
        CHECK((lib = file_line.LesserBound(read_lines.first)).val);
        if (wrap) {
            if (head_read) start_line_adjust = min(0, lib.key - new_first_line);
            if (short_read && tail_read && end_line_cutoff) ++lib;
        }
        int last_read_line = read_lines.first + read_lines.second - 1;
        for (lie = lib; lie.val && lie.key <= last_read_line; ++lie) {
            auto v = lie.val;
            if (shorten_read && !(lie.key + v->wrapped_lines <= last_read_line)) break;
            if (v->size >= 0) read_len += (v->size + 1);
        }
        if (wrap && tail_read) {
            LineMap::ConstIterator i = lie;
            end_line_cutoff = max(0, (--i).key + i.val->wrapped_lines - new_last_line);
        }
    }

    string buf((read_len = X_or_1(read_len)-1), 0);
    if (read_len) {
        file->Seek(lib.val->offset, File::Whence::SET);
        CHECK_EQ(buf.size(), file->Read((char*)buf.data(), buf.size()));
    }
    
    Line *L = 0;
    if (up) for (LineMap::ConstIterator li = lie; li != lib; bo += l + (L != 0), added++) {
        l = (e = -min(0, (--li).val->size)) ? 0 : li.val->size;
        (L = line.PushBack())->AssignText(e ? edits[e-1] : StringPiece(buf.data() + read_len - bo - l, l));
        fb_wrapped_lines += L->Layout(fb->w);
    }
    else for (LineMap::ConstIterator li = lib; li != lie; ++li, bo += l+1, added++) {
        l = (e = -min(0, li.val->size)) ? 0 : li.val->size;
        (L = line.PushFront())->AssignText(e ? edits[e-1] : StringPiece(buf.data() + bo, l));
        fb_wrapped_lines += L->Layout(fb->w);
    }
    if (!up) for (int i=0; i<past_end_lines; i++, added++) { 
        (L = line.PushFront())->Clear();
        fb_wrapped_lines += L->Layout(fb->w);
    }

    CHECK_LT(line.ring.count, line.ring.size);
    if (!resized) {
        for (bool first=1;;first=0) {
            int ll = (L = up ? line.Front() : line.Back())->Lines();
            if (fb_wrapped_lines + (up ? start_line_adjust : -end_line_cutoff) - ll < fb->lines) break;
            fb_wrapped_lines -= ll;
            if (up) line.PopFront(1);
            else    line.PopBack (1);
        }
        if (up) end_line_cutoff   =  (fb_wrapped_lines + start_line_adjust - fb->lines);
        else    start_line_adjust = -(fb_wrapped_lines - end_line_cutoff   - fb->lines);
    }

    end_line_adjust   = line.Front()->Lines() - end_line_cutoff;
    start_line_cutoff = line.Back ()->Lines() + start_line_adjust;
    if (first_ind) *first_ind = up ? -added-1 : -line.Size()+added;

    last_fb_lines = fb->lines;
    last_first_line = new_first_line;
    return dist * (up ? -1 : 1);
}

/* Terminal */

void Terminal::Resized(int w, int h) {
    bool init = !term_width && !term_height;
    int old_term_height = term_height;
    term_width  = w / font->FixedWidth();
    term_height = h / font->height;
    if (init) TextArea::Write(string(term_height, '\n'), 0);
    else {
        int height_dy = term_height - old_term_height;
        if      (height_dy > 0) TextArea::Write(string(height_dy, '\n'), 0);
        else if (height_dy < 0 && term_cursor.y < old_term_height) line.PopBack(-height_dy);
    }
    term_cursor.x = min(term_cursor.x, term_width);
    term_cursor.y = min(term_cursor.y, term_height);
#ifndef _WIN32
    struct winsize ws;
    memzero(ws);
    ws.ws_row = term_height;
    ws.ws_col = term_width;
    ioctl(fd, TIOCSWINSZ, &ws);
#endif
    UpdateCursor();
    TextArea::Resized(w, h);
    ResizedLeftoverRegion(w, h);
}

void Terminal::ResizedLeftoverRegion(int w, int h, bool update_fb) {
    if (!cmd_fb.SizeChanged(w, h, font)) return;
    if (update_fb) {
        for (int i=0; i<start_line;      i++) cmd_fb.Update(&line[-i-1],             GetCursorY(term_height-i));
        for (int i=0; i<skip_last_lines; i++) cmd_fb.Update(&line[-line_fb.lines+i], GetCursorY(i+1));
    }
    cmd_fb.SizeChangedDone();
    last_fb = 0;
}

void Terminal::SetScrollRegion(int b, int e, bool release_fb) {
    if (b<0 || e<0 || e>term_height || b>e) { ERROR(b, "-", e, " outside 1-", term_height); return; }
    int prev_region_beg = scroll_region_beg, prev_region_end = scroll_region_end;
    scroll_region_beg = b;
    scroll_region_end = e;
    bool no_region = !scroll_region_beg || !scroll_region_end ||
        (scroll_region_beg == 1 && scroll_region_end == term_height);
    skip_last_lines = no_region ? 0 : scroll_region_beg - 1;
    start_line_adjust = start_line = no_region ? 0 : term_height - scroll_region_end;
    clip_border.top    = font->height * skip_last_lines;
    clip_border.bottom = font->height * start_line_adjust;
    clip = no_region ? 0 : &clip_border;
    ResizedLeftoverRegion(line_fb.w, line_fb.h, false);

    if (release_fb) { last_fb=0; screen->gd->DrawMode(DrawMode::_2D, 0); }
    int   prev_beg_or_1=X_or_1(prev_region_beg),     prev_end_or_ht=X_or_Y(  prev_region_end, term_height);
    int scroll_beg_or_1=X_or_1(scroll_region_beg), scroll_end_or_ht=X_or_Y(scroll_region_end, term_height);

    if (scroll_beg_or_1 != prev_beg_or_1 || prev_end_or_ht != scroll_end_or_ht) GetPrimaryFrameBuffer();
    for (int i =  scroll_beg_or_1; i <    prev_beg_or_1; i++) line_fb.Update(GetTermLine(i),   line_fb.BackPlus(point(0, (term_height-i+1)*font->height)), LinesFrameBuffer::Flag::NoLayout);
    for (int i =   prev_end_or_ht; i < scroll_end_or_ht; i++) line_fb.Update(GetTermLine(i+1), line_fb.BackPlus(point(0, (term_height-i)  *font->height)), LinesFrameBuffer::Flag::NoLayout);

    if (prev_beg_or_1 < scroll_beg_or_1 || scroll_end_or_ht < prev_end_or_ht) GetSecondaryFrameBuffer();
    for (int i =    prev_beg_or_1; i < scroll_beg_or_1; i++) cmd_fb.Update(GetTermLine(i),   point(0, GetCursorY(i)),   LinesFrameBuffer::Flag::NoLayout);
    for (int i = scroll_end_or_ht; i <  prev_end_or_ht; i++) cmd_fb.Update(GetTermLine(i+1), point(0, GetCursorY(i+1)), LinesFrameBuffer::Flag::NoLayout);
    if (release_fb) cmd_fb.fb.Release();
}

void Terminal::Draw(const Box &b, bool draw_cursor) {
    TextArea::Draw(b, false);
    if (clip) {
        { Scissor s(Box::TopBorder(b, *clip)); cmd_fb.Draw(b.Position(), point(), false); }
        { Scissor s(Box::BotBorder(b, *clip)); cmd_fb.Draw(b.Position(), point(), false); }
    }
    if (draw_cursor) TextGUI::DrawCursor(b.Position() + cursor.p);
    if (selection.enabled) DrawSelection();
}

void Terminal::Write(const string &s, bool update_fb, bool release_fb) {
    if (!MainThread()) return RunInMainThread(new Callback(bind(&Terminal::Write, this, s, update_fb, release_fb)));
    screen->gd->DrawMode(DrawMode::_2D, 0);
    if (bg_color) screen->gd->ClearColor(*bg_color);
    last_fb = 0;
    for (int i = 0; i < s.size(); i++) {
        unsigned char c = s[i];
        if (c == 0x18 || c == 0x1a) { /* CAN or SUB */ parse_state = State::TEXT; continue; }
        if (parse_state == State::ESC) {
            parse_state = State::TEXT; // default next state
            // INFOf("ESC %c", c);
            if (c >= '(' && c <= '/') {
                parse_state = State::CHARSET;
                parse_charset = c;
            } else switch (c) {
                case '[':
                    parse_state = State::CSI; // control sequence introducer
                    parse_csi.clear(); break;
                case ']':
                    parse_state = State::OSC; // operating system command
                    parse_osc.clear();
                    parse_osc_escape = false; break;
                case 'D': Newline();                       break;
                case 'M': NewTopline();                    break;
                case '7': saved_term_cursor = term_cursor; break;
                case '8': term_cursor = saved_term_cursor; break;
                case '=': case '>':                        break; // application or normal keypad
                default: ERRORf("unhandled escape %c (%02x)", c, c);
            }
        } else if (parse_state == State::CHARSET) {
            // INFOf("charset %c%c", parse_charset, c);
            parse_state = State::TEXT;

        } else if (parse_state == State::OSC) {
            if (!parse_osc_escape) {
                if (c == 0x1b) { parse_osc_escape = 1; continue; }
                if (c != '\b') { parse_osc       += c; continue; }
            }
            else if (c != 0x5c) { ERRORf("within-OSC-escape %c (%02x)", c, c); parse_state = State::TEXT; continue; }
            parse_state = State::TEXT;

            if (0) {}
            else INFO("unhandled OSC ", parse_osc);

        } else if (parse_state == State::CSI) {
            // http://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
            if (c < 0x40 || c > 0x7e) { parse_csi += c; continue; }
            // INFOf("CSI %s%c (cur=%d,%d)", parse_csi.c_str(), c, term_cursor.x, term_cursor.y);
            parse_state = State::TEXT;

            int parsed_csi=0, parse_csi_argc=0, parse_csi_argv[16];
            unsigned char parse_csi_argv00 = parse_csi.empty() ? 0 : (isdig(parse_csi[0]) ? 0 : parse_csi[0]);
            for (/**/; Typed::Within<int>(parse_csi[parsed_csi], 0x20, 0x2f); parsed_csi++) {}
            StringPiece intermed(parse_csi.data(), parsed_csi);

            memzeros(parse_csi_argv);
            bool parse_csi_arg_done = 0;
            // http://www.inwap.com/pdp10/ansicode.txt 
            for (/**/; Typed::Within<int>(parse_csi[parsed_csi], 0x30, 0x3f); parsed_csi++) {
                if (parse_csi[parsed_csi] <= '9') { // 0x30 == '0'
                    AccumulateAsciiDigit(&parse_csi_argv[parse_csi_argc], parse_csi[parsed_csi]);
                    parse_csi_arg_done = 0;
                } else if (parse_csi[parsed_csi] <= ';') {
                    parse_csi_arg_done = 1;
                    parse_csi_argc++;
                } else continue;
            }
            if (!parse_csi_arg_done) parse_csi_argc++;

            switch (c) {
                case '@': {
                    LineUpdate l(GetCursorLine(), fb_cb);
                    l->InsertTextAt(term_cursor.x-1, string(X_or_1(parse_csi_argv[0]), ' '), cursor.attr);
                    l->Erase(term_width);
                } break;
                case 'A': term_cursor.y = max(term_cursor.y - X_or_1(parse_csi_argv[0]), 1);           break;
                case 'B': term_cursor.y = min(term_cursor.y + X_or_1(parse_csi_argv[0]), term_height); break;
                case 'C': term_cursor.x = min(term_cursor.x + X_or_1(parse_csi_argv[0]), term_width);  break;
                case 'D': term_cursor.x = max(term_cursor.x - X_or_1(parse_csi_argv[0]), 1);           break;
                case 'H': term_cursor = point(Clamp(parse_csi_argv[1], 1, term_width),
                                              Clamp(parse_csi_argv[0], 1, term_height)); break;
                case 'J': {
                    LineUpdate l(GetCursorLine(), fb_cb);
                    int clear_beg_y = 1, clear_end_y = term_height;
                    if      (parse_csi_argv[0] == 0) { l->Erase(term_cursor.x-1);  clear_beg_y = term_cursor.y; }
                    else if (parse_csi_argv[0] == 1) { l->Erase(0, term_cursor.x); clear_end_y = term_cursor.y; }
                    else if (parse_csi_argv[0] == 2) { l->Clear(); term_cursor.x = term_cursor.y = 1; }
                    for (int i = clear_beg_y; i <= clear_end_y; i++) LineUpdate(GetTermLine(i), fb_cb)->Clear();
                } break;
                case 'K': {
                    LineUpdate l(GetCursorLine(), fb_cb);
                    if      (parse_csi_argv[0] == 0) l->Erase(term_cursor.x-1);
                    else if (parse_csi_argv[0] == 1) l->Erase(0, term_cursor.x);
                    else if (parse_csi_argv[0] == 2) l->Clear();
                } break;
                case 'L': case 'M': {
                    int sl = (c == 'L' ? 1 : -1) * X_or_1(parse_csi_argv[0]);
                    if (clip && term_cursor.y < scroll_region_beg) {
                        ERROR(term_cursor.DebugString(), " outside scroll region ", scroll_region_beg, "-", scroll_region_end);
                    } else {
                        int end_y = X_or_Y(scroll_region_end, term_height);
                        int flag = sl < 0 ? LineUpdate::PushBack : LineUpdate::PushFront;
                        int offset = sl < 0 ? start_line : -skip_last_lines;
                        Scroll(term_cursor.y, end_y, sl, clip);
                        GetPrimaryFrameBuffer();
                        for (int i=0, l=abs(sl); i<l; i++)
                            LineUpdate(GetTermLine(sl<0 ? (end_y-l+i+1) : (term_cursor.y+l-i-1)),
                                       &line_fb, flag, offset);
                    }
                } break;
                case 'P': {
                    LineUpdate l(GetCursorLine(), fb_cb);
                    int erase = max(1, parse_csi_argv[0]);
                    l->Erase(term_cursor.x-1, erase);
                } break;
                case 'h': { // set mode
                    int mode = parse_csi_argv[0];
                    if      (parse_csi_argv00 == 0   && mode ==    4) { insert_mode = true;           }
                    else if (parse_csi_argv00 == 0   && mode ==   34) { /* steady cursor */           }
                    else if (parse_csi_argv00 == '?' && mode ==    1) { /* guarded area tx = all */   }
                    else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = true;        }
                    else if (parse_csi_argv00 == '?' && mode ==   47) { /* alternate screen buffer */ }
                    else if (parse_csi_argv00 == '?' && mode == 1049) { /* save screen */             }
                    else ERROR("unhandled CSI-h mode = ", mode, " av00 = ", parse_csi_argv00, " i= ", intermed.str());
                } break;
                case 'l': { // reset mode
                    int mode = parse_csi_argv[0];
                    if      (parse_csi_argv00 == 0   && mode ==    4) { insert_mode = false;                 }
                    else if (parse_csi_argv00 == 0   && mode ==   34) { /* blink cursor */                   }
                    else if (parse_csi_argv00 == '?' && mode ==    1) { /* guarded areax tx = unprot only */ }
                    else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = false;              }
                    else if (parse_csi_argv00 == '?' && mode ==   47) { /* normal screen buffer */           }
                    else if (parse_csi_argv00 == '?' && mode == 1049) { /* restore screen */                 }
                    else ERROR("unhandled CSI-l mode = ", mode, " av00 = ", parse_csi_argv00, " i= ", intermed.str());
                } break;
                case 'm':
                    for (int i=0; i<parse_csi_argc; i++) {
                        int sgr = parse_csi_argv[i]; // select graphic rendition
                        if      (sgr >= 30 && sgr <= 37) Attr::SetFGColorIndex(&cursor.attr, sgr-30);
                        else if (sgr >= 40 && sgr <= 47) Attr::SetBGColorIndex(&cursor.attr, sgr-40);
                        else switch(sgr) {
                            case 0:         cursor.attr  =  default_cursor_attr;    break;
                            case 1:         cursor.attr |=  Attr::Bold;             break;
                            case 3:         cursor.attr |=  Attr::Italic;           break;
                            case 4:         cursor.attr |=  Attr::Underline;        break;
                            case 5: case 6: cursor.attr |=  Attr::Blink;            break;
                            case 7:         cursor.attr |=  Attr::Reverse;          break;
                            case 22:        cursor.attr &= ~Attr::Bold;             break;
                            case 23:        cursor.attr &= ~Attr::Italic;           break;
                            case 24:        cursor.attr &= ~Attr::Underline;        break;
                            case 25:        cursor.attr &= ~Attr::Blink;            break;
                            case 39:        Attr::SetFGColorIndex(&cursor.attr, 7); break;
                            case 49:        Attr::SetBGColorIndex(&cursor.attr, 0); break;
                            default:        ERROR("unhandled SGR ", sgr);
                        }
                    } break;
                case 'r':
                    if (parse_csi_argc == 2) SetScrollRegion(parse_csi_argv[0], parse_csi_argv[1]);
                    else ERROR("invalid scroll region", parse_csi_argc);
                    break;
                default:
                    ERRORf("unhandled CSI %s%c", parse_csi.c_str(), c);
            }
        } else {
            // http://en.wikipedia.org/wiki/C0_and_C1_control_codes#C0_.28ASCII_and_derivatives.29
            bool C0_control = (c >= 0x00 && c <= 0x1f) || c == 0x7f;
            bool C1_control = (c >= 0x80 && c <= 0x9f);
            if (C0_control || C1_control) FlushParseText();
            if (C0_control) switch(c) {
                case '\a':   INFO("bell");                            break; // bell
                case '\b':   term_cursor.x = max(term_cursor.x-1, 1); break; // backspace
                case '\t':   term_cursor.x = term_cursor.x+8;         break; // tab 
                case '\r':   term_cursor.x = 1;                       break; // carriage return
                case '\x1b': parse_state = State::ESC;                break;
                case '\x14': case '\x15': case '\x7f':                break; // shift charset in, out, delete
                case '\n':   case '\v':   case '\f':       Newline(); break; // line feed, vertical tab, form feed
                default:                                   ERRORf("unhandled C0 control %02x", c);
            } else if (0 && C1_control) {
                if (0) {}
                else ERRORf("unhandled C1 control %02x", c);
            } else {
                parse_text += c;
            }
        }
    }
    FlushParseText();
    UpdateCursor();
    line_fb.fb.Release();
    last_fb = 0;
}

void Terminal::FlushParseText() {
    if (parse_text.empty()) return;
    CHECK_GE(term_cursor.x, 1);
    int consumed = 0, write_size = 0;
    String16 input_text = String::ToUTF16(parse_text, &consumed);
    for (int wrote = 0; wrote < input_text.size(); wrote += write_size) {
        if (wrote) Newline(true);
        {
            LineUpdate l(GetCursorLine(), fb_cb);
            int remaining = input_text.size() - wrote;
            write_size = min(remaining, term_width - term_cursor.x + 1);
            l->UpdateText(term_cursor.x-1, String16Piece(input_text.data()+wrote, write_size), cursor.attr, term_width);
        }
    }
    term_cursor.x += write_size;
    parse_text.erase(0, consumed);
}

void Terminal::Newline(bool carriage_return) {
    if (clip && term_cursor.y == scroll_region_end) {
        Scroll(scroll_region_beg, scroll_region_end, -1, clip);
        LineUpdate(GetTermLine(scroll_region_end), GetPrimaryFrameBuffer(), LineUpdate::PushBack, start_line);
    } else if (term_cursor.y == term_height) {
        if (!clip) { TextArea::Write(string(1, '\n'), true, false); last_fb=&line_fb; }
    } else term_cursor.y = min(term_height, term_cursor.y+1);
    if (carriage_return) term_cursor.x = 1;
}

void Terminal::NewTopline() {
    if (clip && term_cursor.y == scroll_region_beg) {
        LineUpdate(line.InsertAt(GetTermLineIndex(term_cursor.y), 1, start_line_adjust),
                   GetPrimaryFrameBuffer(), LineUpdate::PushFront, skip_last_lines);
    } else if (term_cursor.y == 1) {
        if (!clip) LineUpdate(line.InsertAt(-term_height, 1, start_line_adjust),
                              GetPrimaryFrameBuffer(), LineUpdate::PushFront);
    } else term_cursor.y = max(1, term_cursor.y-1);
}

/* Dialog */

void Dialog::Draw() {
    if (moving) box.SetPosition(win_start + screen->mouse - mouse_start);

    Box outline = BoxAndTitle();
    static const int min_width = 50, min_height = 1;
    if (resizing_left)   Typed::MinusPlus(&outline.x, &outline.w, Typed::Max(-outline.w + min_width,            (int)(mouse_start.x - screen->mouse.x)));
    if (resizing_bottom) Typed::MinusPlus(&outline.y, &outline.h, Typed::Max(-outline.h + min_height + title.h, (int)(mouse_start.y - screen->mouse.y)));
    if (resizing_right)  outline.w += Typed::Max(-outline.w + min_width, (int)(screen->mouse.x - mouse_start.x));

    if (!app->input.MouseButton1Down()) {
        if (resizing_left || resizing_right || resizing_top || resizing_bottom) {
            box = Box(outline.x, outline.y, outline.w, outline.h - title.h);
            moving = true;
        }
        if (moving) Layout();
        moving = resizing_left = resizing_right = resizing_top = resizing_bottom = 0;
    }

    screen->gd->FillColor(color);
    box.Draw();

    screen->gd->SetColor(color + Color(0,0,0,(int)(color.A()*.25)));
    (title + box.TopLeft()).Draw();
    screen->gd->SetColor(Color::white);

    if (moving || resizing_left || resizing_right || resizing_top || resizing_bottom)
        BoxOutline().Draw(outline);
}

/* Browsers */

BrowserInterface *CreateDefaultBrowser(Window *W, Asset *a, int w, int h) {
    BrowserInterface *ret = 0;
    if ((ret = CreateQTWebKitBrowser(a)))        return ret;
    if ((ret = CreateBerkeliumBrowser(a, w, h))) return ret;
    return new SimpleBrowser(W, Fonts::Default(), Box(0, 0, w, h));
}

int PercentRefersTo(unsigned short prt, Flow *inline_context) {
    switch (prt) {
        case DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth:           return inline_context->cur_attr.font->max_width;
        case DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight:          return inline_context->cur_attr.font->height;
        case DOM::CSSPrimitiveValue::PercentRefersTo::LineHeight:          return inline_context->layout.line_height;
        case DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight: return inline_context->container->h;
    }; return inline_context->container->w;
}

string DOM::Node::HTML4Style() { 
    string ret;
    if (Node *a = getAttributeNode("align")) {
        const DOMString &v = a->nodeValue();
        if      (StringEquals(v, "right"))  StrAppend(&ret, "margin-left: auto;\n");
        else if (StringEquals(v, "center")) StrAppend(&ret, "margin-left: auto;\nmargin-right: auto;\n");
    }
    if (Node *a = getAttributeNode("background")) StrAppend(&ret, "background-image: ", a->nodeValue(), ";\n");
    if (Node *a = getAttributeNode("bgcolor"))    StrAppend(&ret, "background-color: ", a->nodeValue(), ";\n");
    if (Node *a = getAttributeNode("border"))     StrAppend(&ret, "border-width: ",     a->nodeValue(), "px;\n");
    if (Node *a = getAttributeNode("color"))      StrAppend(&ret, "color: ",            a->nodeValue(), ";\n");
    if (Node *a = getAttributeNode("height"))     StrAppend(&ret, "height: ",           a->nodeValue(), "px;\n");
    if (Node *a = getAttributeNode("text"))       StrAppend(&ret, "color: ",            a->nodeValue(), ";\n");
    if (Node *a = getAttributeNode("width"))      StrAppend(&ret, "width: ",            a->nodeValue(), "px;\n");
    return ret;
}

DOM::Node *DOM::Node::cloneNode(bool deep) {
    FATAL("not implemented");
}

DOM::Renderer *DOM::Node::AttachRender() {
    if (!render) render = AllocatorNew(ownerDocument->alloc, (DOM::Renderer), (this));
    return render;
}

void DOM::Element::setAttribute(const DOMString &name, const DOMString &value) {
    DOM::Attr *attr = AllocatorNew(ownerDocument->alloc, (DOM::Attr), (ownerDocument));
    attr->name = name;
    attr->value = value;
    setAttributeNode(attr);
}

float DOM::CSSPrimitiveValue::ConvertToPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
    switch (u) {
        case CSS_PX:         return          n;
        case CSS_EXS:        return          n  * inline_context->cur_attr.font->max_width;
        case CSS_EMS:        return          n  * inline_context->cur_attr.font->size;
        case CSS_IN:         return          n  * FLAGS_dots_per_inch;
        case CSS_CM:         return CMToInch(n) * FLAGS_dots_per_inch;
        case CSS_MM:         return MMToInch(n) * FLAGS_dots_per_inch;
        case CSS_PC:         return          n  /   6 * FLAGS_dots_per_inch;
        case CSS_PT:         return          n  /  72 * FLAGS_dots_per_inch;
        case CSS_PERCENTAGE: return          n  / 100 * LFL::PercentRefersTo(prt, inline_context);
    }; return 0;
}

float DOM::CSSPrimitiveValue::ConvertFromPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
    switch (u) {
        case CSS_PX:         return          n;
        case CSS_EXS:        return          n / inline_context->cur_attr.font->max_width;
        case CSS_EMS:        return          n / inline_context->cur_attr.font->size;
        case CSS_IN:         return          n / FLAGS_dots_per_inch;
        case CSS_CM:         return InchToCM(n / FLAGS_dots_per_inch);
        case CSS_MM:         return InchToMM(n / FLAGS_dots_per_inch);
        case CSS_PC:         return          n *   6 / FLAGS_dots_per_inch;
        case CSS_PT:         return          n *  72 / FLAGS_dots_per_inch;
        case CSS_PERCENTAGE: return          n * 100 / LFL::PercentRefersTo(prt, inline_context);
    }; return 0;                            
}

int DOM::FontSize::getFontSizeValue(Flow *flow) {
    int base = FLAGS_default_font_size; float scale = 1.2;
    switch (attr) {
        case XXSmall: return base/scale/scale/scale;    case XLarge:  return base*scale*scale;
        case XSmall:  return base/scale/scale;          case XXLarge: return base*scale*scale*scale;
        case Small:   return base/scale;                case Larger:  return flow->cur_attr.font->size*scale;
        case Medium:  return base;                      case Smaller: return flow->cur_attr.font->size/scale;
        case Large:   return base*scale;
    }
    return getPixelValue(flow);
}

void DOM::HTMLTableSectionElement::deleteRow(long index) { return parentTable->deleteRow(index); }
DOM::HTMLElement *DOM::HTMLTableSectionElement::insertRow(long index) { return parentTable->insertRow(index); }

void DOM::Renderer::UpdateStyle(Flow *F) {
    DOM::Node *n = style.node;
    CHECK(n->parentNode);
    style.is_root = n->parentNode == n->ownerDocument;
    CHECK(style.is_root || n->parentNode->render);
    DOM::Attr *inline_style = n->getAttributeNode("style");
    n->ownerDocument->style_context->Match(&style, n, style.is_root ? 0 : &n->parentNode->render->style,
                                           inline_style ? inline_style->nodeValue().c_str() : 0);

    DOM::Display              display        = style.Display();
    DOM::Position             position       = style.Position();
    DOM::Float                _float         = style.Float();
    DOM::Clear                clear          = style.Clear();
    DOM::Visibility           visibility     = style.Visibility();
    DOM::Overflow             overflow       = style.Overflow();
    DOM::TextAlign            textalign      = style.TextAlign();
    DOM::TextTransform        texttransform  = style.TextTransform();
    DOM::TextDecoration       textdecoration = style.TextDecoration();
    DOM::CSSStringValue       bgimage        = style.BackgroundImage();
    DOM::BackgroundRepeat     bgrepeat       = style.BackgroundRepeat();
    DOM::BackgroundAttachment bgattachment   = style.BackgroundAttachment();
    DOM::VerticalAlign        valign         = style.VerticalAlign();
    DOM::CSSAutoNumericValue  zindex         = style.ZIndex();
    DOM::BorderStyle          os             = style.OutlineStyle();
    DOM::BorderStyle          bs_top         = style.BorderTopStyle();
    DOM::BorderStyle          bs_bottom      = style.BorderBottomStyle();
    DOM::BorderStyle          bs_left        = style.BorderLeftStyle();
    DOM::BorderStyle          bs_right       = style.BorderRightStyle();

    display_table_element = display.TableElement();
    display_table         = display.v  == DOM::Display::Table;
    display_inline_table  = display.v  == DOM::Display::InlineTable;
    display_block         = display.v  == DOM::Display::Block;
    display_inline        = display.v  == DOM::Display::Inline;
    display_inline_block  = display.v  == DOM::Display::InlineBlock;
    display_list_item     = display.v  == DOM::Display::ListItem;
    display_none          = display.v  == DOM::Display::None || display.v == DOM::Display::Column || display.v == DOM::Display::ColGroup;

    position_relative = position.v == DOM::Position::Relative;
    position_absolute = position.v == DOM::Position::Absolute;
    position_fixed    = position.v == DOM::Position::Fixed;
    positioned = position_relative || position_absolute || position_fixed;

    if (position_absolute || position_fixed) {
        floating = float_left = float_right = 0;
    } else {
        float_left  = _float.v == DOM::Float::Left;
        float_right = _float.v == DOM::Float::Right;
        floating    = float_left || float_right;
    }

    if (position_absolute) {
        DOM::Rect r = style.Clip();
        if ((clip = !r.Null())) {
            clip_rect = Box(r.left .getPixelValue(F), r.top   .getPixelValue(F),
                            r.right.getPixelValue(F), r.bottom.getPixelValue(F));
            clip_rect.w = max(0, clip_rect.w - clip_rect.x);
            clip_rect.h = max(0, clip_rect.h - clip_rect.y);
        }
    } else clip = 0;

    overflow_auto   = overflow.v == DOM::Overflow::Auto;
    overflow_scroll = overflow.v == DOM::Overflow::Scroll;
    overflow_hidden = overflow.v == DOM::Overflow::Hidden;

    normal_flow = !position_absolute && !position_fixed && !floating && !display_table_element && !style.is_root;
    inline_block = display_inline_block || display_inline_table;
    bool block_level_element = display_block || display_list_item || display_table;
    establishes_block = style.is_root || floating || position_absolute || position_fixed ||
        inline_block || display_table_element ||
        (block_level_element && (overflow_hidden || overflow_scroll || overflow_auto));
    block_level_box = block_level_element | establishes_block;
    if (!block_level_box) overflow_auto = overflow_scroll = overflow_hidden = 0;

    right_to_left    = style.Direction     ().v == DOM::Direction::RTL;
    border_collapse  = style.BorderCollapse().v == DOM::BorderCollapse::Collapse;
    clear_left       = (clear.v == DOM::Clear::Left  || clear.v == DOM::Clear::Both);
    clear_right      = (clear.v == DOM::Clear::Right || clear.v == DOM::Clear::Both);
    hidden           = (visibility.v == DOM::Visibility::Hidden || visibility.v == DOM::Visibility::Collapse);
    valign_top       = (valign.v == DOM::VerticalAlign::Top || valign.v == DOM::VerticalAlign::TextTop);
    valign_mid       = (valign.v == DOM::VerticalAlign::Middle);
    valign_px        = (!valign.Null() && !valign.attr) ? valign.getPixelValue(F) : 0;
    bgrepeat_x       = (bgrepeat.v == DOM::BackgroundRepeat::Repeat || bgrepeat.v == DOM::BackgroundRepeat::RepeatX);
    bgrepeat_y       = (bgrepeat.v == DOM::BackgroundRepeat::Repeat || bgrepeat.v == DOM::BackgroundRepeat::RepeatY);
    bgfixed          = bgattachment  .v == DOM::BackgroundAttachment::Fixed;
    textalign_center = textalign     .v == DOM::TextAlign::Center;
    textalign_right  = textalign     .v == DOM::TextAlign::Right;
    underline        = textdecoration.v == DOM::TextDecoration::Underline;
    overline         = textdecoration.v == DOM::TextDecoration::Overline;
    midline          = textdecoration.v == DOM::TextDecoration::LineThrough;
    blink            = textdecoration.v == DOM::TextDecoration::Blink;
    uppercase        = texttransform .v == DOM::TextTransform::Uppercase;
    lowercase        = texttransform .v == DOM::TextTransform::Lowercase;
    capitalize       = texttransform .v == DOM::TextTransform::Capitalize;

    if (n->nodeType == DOM::TEXT_NODE) {
        color = style.Color().v;
        color.a() = 1.0;
    }
    if (style.bgcolor_not_inherited) {
        background_color = style.BackgroundColor().v;
    }

    os            = os       .Null() ? 0 : os.v;
    bs_t          = bs_top   .Null() ? 0 : bs_top.v;
    bs_b          = bs_bottom.Null() ? 0 : bs_bottom.v;
    bs_l          = bs_left  .Null() ? 0 : bs_left.v;
    bs_r          = bs_right .Null() ? 0 : bs_right.v;
    border_top    = Color(style.BorderTopColor   ().v, 1.0);
    border_left   = Color(style.BorderLeftColor  ().v, 1.0);
    border_right  = Color(style.BorderRightColor ().v, 1.0);
    border_bottom = Color(style.BorderBottomColor().v, 1.0);
    outline       = Color(style.OutlineColor     ().v, 1.0);

    if (!bgimage.Null() && !background_image)
        background_image = n->ownerDocument->ownerBrowser->OpenImage(String::ToUTF8(bgimage.v));

    DOM::CSSNormNumericValue lineheight = style.LineHeight(), charspacing = style.LetterSpacing(), wordspacing = style.WordSpacing();
    lineheight_px  = (!lineheight .Null() && !lineheight. _norm) ? lineheight .getPixelValue(F) : 0;
    charspacing_px = (!charspacing.Null() && !charspacing._norm) ? charspacing.getPixelValue(F) : 0;
    wordspacing_px = (!wordspacing.Null() && !wordspacing._norm) ? wordspacing.getPixelValue(F) : 0;
}

Font *DOM::Renderer::UpdateFont(Flow *F) {
    DOM::FontFamily  font_family  = style.FontFamily();
    DOM::FontSize    font_size    = style.FontSize();
    DOM::FontStyle   font_style   = style.FontStyle();
    DOM::FontWeight  font_weight  = style.FontWeight();
    DOM::FontVariant font_variant = style.FontVariant();

    int font_size_px = font_size.getFontSizeValue(F);
    int font_flag = ((font_weight.v == DOM::FontWeight::_700 || font_weight.v == DOM::FontWeight::_800 || font_weight.v == DOM::FontWeight::_900 || font_weight.v == DOM::FontWeight::Bold || font_weight.v == DOM::FontWeight::Bolder) ? FontDesc::Bold : 0) |
        ((font_style.v == DOM::FontStyle::Italic || font_style.v == DOM::FontStyle::Oblique) ? FontDesc::Italic : 0);
    Font *font = font_family.attr ? Fonts::Get("", String::ToUTF8(font_family.cssText()), font_size_px, Color::white, font_flag) : 0;
    for (int i=0; !font && i<font_family.name.size(); i++) {
        vector<DOM::FontFace> ff;
        style.node->ownerDocument->style_context->FontFaces(font_family.name[i], &ff);
        for (int j=0; j<ff.size(); j++) {
            // INFO(j, " got font face: ", ff[j].family.name.size() ? ff[j].family.name[0] : "<NO1>",
            //      " ", ff[j].source.size() ? ff[j].source[0] : "<NO2>");
        }
    }
    return font ? font : Fonts::Get(FLAGS_default_font, font_size_px, Color::white);
}

void DOM::Renderer::UpdateDimensions(Flow *F) {
    DOM::CSSAutoNumericValue     width = style.   Width(),     height = style.   Height();
    DOM::CSSNoneNumericValue max_width = style.MaxWidth(), max_height = style.MaxHeight();
    DOM::CSSAutoNumericValue ml = style.MarginLeft(), mr = style.MarginRight();
    DOM::CSSAutoNumericValue mt = style.MarginTop(),  mb = style.MarginBottom();

    bool no_auto_margin = position_absolute || position_fixed || display_table_element;
    width_percent = width .IsPercent();
    width_auto    = width ._auto;
    height_auto   = height._auto;
    ml_auto       = ml    ._auto && !no_auto_margin;
    mr_auto       = mr    ._auto && !no_auto_margin;
    mt_auto       = mt    ._auto && !no_auto_margin;
    mb_auto       = mb    ._auto && !no_auto_margin;
    width_px      = width_auto  ? 0 : width .getPixelValue(F);
    height_px     = height_auto ? 0 : height.getPixelValue(F);
    ml_px         = ml_auto     ? 0 : ml    .getPixelValue(F);
    mr_px         = mr_auto     ? 0 : mr    .getPixelValue(F);
    mt_px         = mt_auto     ? 0 : mt    .getPixelValue(F);
    mb_px         = mb_auto     ? 0 : mb    .getPixelValue(F);
    bl_px         = !bs_l ? 0 : style.BorderLeftWidth  ().getPixelValue(F);
    br_px         = !bs_r ? 0 : style.BorderRightWidth ().getPixelValue(F);
    bt_px         = !bs_t ? 0 : style.BorderTopWidth   ().getPixelValue(F);
    bb_px         = !bs_b ? 0 : style.BorderBottomWidth().getPixelValue(F);
     o_px         = style.OutlineWidth     ().getPixelValue(F);
    pl_px         = style.PaddingLeft      ().getPixelValue(F);
    pr_px         = style.PaddingRight     ().getPixelValue(F);
    pt_px         = style.PaddingTop       ().getPixelValue(F);
    pb_px         = style.PaddingBottom    ().getPixelValue(F);

    bool include_floats = block_level_box && establishes_block && !floating;
    if (include_floats && width_auto) box.w = box.CenterFloatWidth(F->p.y, F->cur_line.height) - MarginWidth(); 
    else                              box.w = width_auto ? WidthAuto(F) : width_px;
    if (!max_width._none)             box.w = min(box.w, max_width.getPixelValue(F));

    if (!width_auto) UpdateMarginWidth(F, width_px);

    if (block_level_box && (width_auto || width_percent)) {
        if (normal_flow) shrink = inline_block || style.node->parentNode->render->shrink;
        else             shrink = width_auto && (floating || position_absolute || position_fixed);
    } else               shrink = normal_flow ? style.node->parentNode->render->shrink : 0;
}

void DOM::Renderer::UpdateMarginWidth(Flow *F, int w) {
    if (ml_auto && mr_auto) ml_px = mr_px = MarginCenterAuto(F, w);
    else if       (ml_auto) ml_px =         MarginLeftAuto  (F, w);
    else if       (mr_auto) mr_px =         MarginRightAuto (F, w);
}

int DOM::Renderer::ClampWidth(int w) {
    DOM::CSSNoneNumericValue max_width = style.MaxWidth();
    if (!max_width._none) w = min(w, max_width.getPixelValue(flow));
    return max(w, style.MinWidth().getPixelValue(flow));
}

void DOM::Renderer::UpdateBackgroundImage(Flow *F) {
    int bgpercent_x = 0, bgpercent_y = 0; bool bgdone_x = 0, bgdone_y = 0;
    int iw = background_image->tex.width, ih = background_image->tex.height; 
    DOM::CSSNumericValuePair bgposition = style.BackgroundPosition();
    if (!bgposition.Null()) {
        if (bgposition.first .IsPercent()) bgpercent_x = bgposition.first .v; else { bgposition_x = bgposition.first .getPixelValue(F);      bgdone_x = 0; }
        if (bgposition.second.IsPercent()) bgpercent_y = bgposition.second.v; else { bgposition_y = bgposition.second.getPixelValue(F) + ih; bgdone_y = 0; }
    }
    if (!bgdone_x) { float p=bgpercent_x/100.0; bgposition_x = box.w*p - iw*p; }
    if (!bgdone_y) { float p=bgpercent_y/100.0; bgposition_y = box.h*p + ih*(1-p); }
}

void DOM::Renderer::UpdateFlowAttributes(Flow *F) {
    if (charspacing_px)       F->layout.char_spacing       = charspacing_px;
    if (wordspacing_px)       F->layout.word_spacing       = wordspacing_px;
    if (lineheight_px)        F->layout.line_height        = lineheight_px;
    if (valign_px)            F->layout.valign_offset      = valign_px;
    if (1)                    F->layout.align_center       = textalign_center;
    if (1)                    F->layout.align_right        = textalign_right;
    if (1)                    F->cur_attr.underline        = underline;
    if (1)                    F->cur_attr.overline         = overline;
    if (1)                    F->cur_attr.midline          = midline;
    if (1)                    F->cur_attr.blink            = blink;
    if      (capitalize)      F->layout.word_start_char_tf = ::toupper;
    if      (uppercase)       F->layout.char_tf            = ::toupper;
    else if (lowercase)       F->layout.char_tf            = ::tolower;
    else                      F->layout.char_tf            = 0;
    if (valign_top) {}
    if (valign_mid) {}
}

void DOM::Renderer::PushScissor(const Box &w) {
    if (!tiles) return;
    if (!tile_context_opened) { tile_context_opened=1; tiles->ContextOpen(); }
    TilesPreAdd (tiles, &Tiles::PushScissor, tiles, w);
    TilesPostAdd(tiles, &GraphicsDevice::PopScissor, screen->gd);
}

void DOM::Renderer::Finish() {
    if (tile_context_opened) tiles->ContextClose();
    tile_context_opened = 0;
}

struct MySimpleBrowser {
    struct Handler {
        SimpleBrowser *browser;
        string url;
        virtual ~Handler() {}
        Handler(SimpleBrowser *B, const string &URL) : browser(B), url(URL) {}
        void SetDocumentLayoutDirty() { DOM::Node *html = browser->doc.node->documentElement(); if (html) html->render->layout_dirty = true; }
        void SetDocumentStyleDirty()  { DOM::Node *html = browser->doc.node->documentElement(); if (html) html->render->style_dirty  = true; }
        virtual void Complete(void *self) {
            browser->completed++;
            browser->outstanding.erase(self);
            SetDocumentLayoutDirty();
            delete this;
        }
    };
    struct StyleHandler : public Handler {
        string charset; StyleSheet *style_target;
        StyleHandler(SimpleBrowser *b, const string &url, const DOM::DOMString &cs) : Handler(b, url), charset(String::ToUTF8(cs)),
        style_target(new StyleSheet(browser->doc.node, "", charset.empty() ? "UTF-8" : charset.c_str(), 0, 0)) { b->doc.style_sheet.push_back(style_target); }

        virtual void Complete(void *self) { 
            style_target->Done();
            if (browser->Running(self)) {
                browser->doc.node->style_context->AppendSheet(style_target);
                SetDocumentStyleDirty();
            }
            Handler::Complete(self);
        }
        void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *cb, int cl) {
            if      (!h && (!cb || !cl)) Complete(this);
            else if (!h)                 style_target->Parse(string(cb, cl));
        }
    };
    struct HTMLHandler : public HTMLParser, public StyleHandler {
        vector<DOM::Node*> target;
        HTMLHandler(SimpleBrowser *b, const string &url, DOM::Node *t) : StyleHandler(b, url, "") { target.push_back(t); }

        void ParseChunkCB(const char *content, int content_len) { SetDocumentLayoutDirty(); }
        void WGetContentBegin(Connection *c, const char *h, const string &ct) { browser->doc.content_type=ct; }
        void WGetContentEnd(Connection *c) { StyleHandler::Complete(this); }
        void CloseTag(const String &tag, const KV &attr, const TagStack &stack) {
            if (target.size() && tag == target.back()->nodeName()) target.pop_back();
        }
        void OpenTag(const String &input_tag, const KV &attr, const TagStack &stack) {
            string tag = LFL::String::ToAscii(input_tag);
            DOM::Node *parentNode = target.back();
            DOM::Element *addNode = 0;
            if (tag == "col") { CHECK(parentNode->AsHTMLTableColElement()); }
            if (tag == "td" || tag == "th") {
                DOM::HTMLTableRowElement *row = parentNode->AsHTMLTableRowElement();
                CHECK(row);
                addNode = row->insertCell(row->cells.length());
            } else if (tag == "tr") {
                DOM::HTMLTableElement        *table   = parentNode->AsHTMLTableElement();
                DOM::HTMLTableSectionElement *section = parentNode->AsHTMLTableSectionElement();
                CHECK(table || section);
                if (table) addNode = table  ->insertRow(table               ->rows.length());
                else       addNode = section->insertRow(section->parentTable->rows.length());
            } else if (HTMLParser::TableDependent(tag) && tag != "col") {
                DOM::HTMLTableElement *table = parentNode->AsHTMLTableElement();
                CHECK(table);
                if      (tag == "thead")    addNode = table->createTHead();
                else if (tag == "tbody")    addNode = table->createTBody();
                else if (tag == "tfoot")    addNode = table->createTFoot();
                else if (tag == "caption")  addNode = table->createCaption();
                else if (tag == "colgroup") addNode = table->createColGroup();
            } else {
                addNode = parentNode->document()->createElement(tag);
            }
            if (!addNode) return;
            addNode->tagName = tag;
            addNode->ParseAttributes(attr);
            addNode->AttachRender();
            parentNode->appendChild(addNode);
            target.push_back(addNode);
            if (tag == "body") {
                int ind; string pseudo_style = addNode->AsHTMLBodyElement()->HTML4PseudoClassStyle();
                if (!pseudo_style.empty() && style_target) style_target->Parse(pseudo_style);
            } else if (tag == "img" || (tag == "input" && StringEquals(addNode->getAttribute("type"), "image"))) {
                browser->Open(LFL::String::ToUTF8(addNode->getAttribute("src")), addNode);
            } else if (tag == "link") {
                if (StringEquals       (addNode->getAttribute("rel"),   "stylesheet") &&
                    StringEmptyOrEquals(addNode->getAttribute("media"), "screen", "all")) {
                    browser->Open(LFL::String::ToUTF8(addNode->getAttribute("href")), addNode);
                }
            }
        }
        void Text(const String &input_text, const TagStack &stack) {
            if (!browser->Running(this)) return;
            DOM::Node *parentNode = target.back();
            DOM::Text *ret = parentNode->document()->createTextNode(DOM::DOMString());
            ret->AttachRender();
            parentNode->appendChild(ret);

            bool link = parentNode->htmlElementType == DOM::HTML_ANCHOR_ELEMENT;
            if (link) {
                // ret->render->activate_id.push_back(ret->ownerDocument->gui->mouse.AddClickBox(ret->render->box, MouseController::CB(bind(&SimpleBrowser::AnchorClick, parentNode))));
            }

            if (!inpre) {
                for (const char *p = text.data(), *e = p + text.size(); p < e; /**/) {
                    if (isspace(*p)) {
                        ret->appendData(DOM::StringPiece::Space());
                        while (p<e && isspace(*p)) p++;
                    } else {
                        const char *b = p;
                        while (p<e && !isspace(*p)) p++;
                        ret->appendData(HTMLParser::ReplaceEntitySymbols(string(b, p-b)));
                    }
                }
            } else ret->appendData(text);
        }
        void Script(const String &text, const TagStack &stack) {}
        void Style(const String &text, const TagStack &stack) {
            if (style_target) style_target->Parse(LFL::String::ToUTF8(text));
        }
    };
    struct ImageHandler : public Handler {
        string content; Asset *target;
        ImageHandler(SimpleBrowser *b, const string &url, Asset *t) : Handler(b, url), target(t) {}
        void Complete(const string &content_type) {
            if (!content.empty() && browser->Running(this)) {
                string fn = basename(url.c_str(), 0, 0);
                if      (MIMEType::Jpg(content_type) && !FileSuffix::Jpg(fn)) fn += ".jpg";
                else if (MIMEType::Png(content_type) && !FileSuffix::Png(fn)) fn += ".png";

                if (PrefixMatch(content_type, "text/html")) INFO("ImageHandler content='", content, "'");

                target->Load(content.data(), fn.c_str(), content.size());
                INFO("ImageHandler ", content_type, ": ", url, " ", fn, " ", target->tex.width, " ", target->tex.height);
            }
            Handler::Complete(this);
        }
        void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *cb, int cl) { 
            if      (!h && (!cb || !cl)) Complete(ct);
            else if (!h)                 content.append(cb, cl);
        }
    };
};

void SimpleBrowser::Clear() {
    delete js_context;
    VectorClear(&doc.style_sheet);
    gui.Clear();
    v_scrollbar.LayoutAttached(Viewport());
    h_scrollbar.LayoutAttached(Viewport());
    outstanding.clear();
    alloc.Reset();
    doc.node = AllocatorNew(&alloc, (DOM::HTMLDocument), (this, &alloc, &gui));
    doc.node->style_context = AllocatorNew(&alloc, (StyleContext), (doc.node));
    doc.node->style_context->AppendSheet(StyleSheet::Default());
    js_context = CreateV8JSContext(js_console, doc.node);
}

void SimpleBrowser::Navigate(const string &url) {
    if (layers.empty()) return SystemBrowser::Open(url.c_str());
}

void SimpleBrowser::OpenHTML(const string &content) {
    Clear();
    MySimpleBrowser::HTMLHandler *html_handler = new MySimpleBrowser::HTMLHandler(this, "", doc.node);
    outstanding.insert(html_handler);
    html_handler->WGetCB(0, 0, string(), content.c_str(), content.size());
    html_handler->WGetCB(0, 0, string(), 0,               0);
}

void SimpleBrowser::Open(const string &url, DOM::Frame *frame) {
    Clear();
    doc.node->setURL(url);
    doc.node->setDomain(HTTP::HostURL(url.c_str()).c_str());
    string urldir = url.substr(0, dirnamelen(url.c_str(), 0, 1));
    doc.node->setBaseURI((urldir.size() >= 3 && urldir.substr(urldir.size()-3) == "://") ? url : urldir);
    Open(url, doc.node);
}

void SimpleBrowser::Open(const string &input_url, DOM::Node *target) {
    if (input_url.empty()) return;
    string url = input_url;
    bool data_url = PrefixMatch(url, "data:");
    if (data_url) { /**/
    } else if (PrefixMatch(url, "/")) {
        if (PrefixMatch(url, "//")) url = "http:"                          + url;
        else if (doc.node)          url = String::ToUTF8(doc.node->domain) + url;
    } else if (url.substr(0, 8).find("://") == string::npos) {
        if (doc.node) url = StrCat(doc.node->baseURI, "/", url);
    }

    void *handler = 0;
    HTTPClient::ResponseCB callback;
    DOM::HTMLDocument     *html  = target->AsHTMLDocument();
    DOM::HTMLImageElement *image = target->AsHTMLImageElement();
    DOM::HTMLLinkElement  *link  = target->AsHTMLLinkElement();
    DOM::HTMLInputElement *input = target->AsHTMLInputElement();
    bool input_image = input && StringEquals(input->getAttribute("type"), "image");
    Asset **image_asset = input_image ? &input->image_asset : (image ? &image->asset : 0);

    if (image_asset) {
        CHECK_EQ(*image_asset, 0);
        if ((*image_asset = FindOrNull(image_cache, url))) return;
        *image_asset = new Asset();
        image_cache[url] = *image_asset;
    }

    if (data_url) {
        int semicolon=url.find(";"), comma=url.find(",");
        if (semicolon != string::npos && comma != string::npos && semicolon < comma) {
            string mimetype=url.substr(5, semicolon-5), encoding=url.substr(semicolon+1, comma-semicolon-1);
            if (PrefixMatch(mimetype, "image/") && image_asset) {
                string fn = toconvert(mimetype, tochar<'/', '.'>);
                if (encoding == "base64") {
                    string content = Singleton<Base64>::Get()->Decode(url.c_str()+comma+1, url.size()-comma-1);
                    (*image_asset)->Load(content.data(), fn.c_str(), content.size());
                } else INFO("unhandled: data:url encoding ", encoding);
            } else INFO("unhandled data:url mimetype ", mimetype);
        } else INFO("unhandled data:url ", input_url);
        return;
    }

    if      (html)        { handler = new MySimpleBrowser::HTMLHandler (this, url, html);                          callback = bind(&HTMLParser::WGetCB        ,                    (MySimpleBrowser::HTMLHandler*) handler, _1, _2, _3, _4, _5); }
    else if (image_asset) { handler = new MySimpleBrowser::ImageHandler(this, url, *image_asset);                  callback = bind(&MySimpleBrowser::ImageHandler::WGetResponseCB, (MySimpleBrowser::ImageHandler*)handler, _1, _2, _3, _4, _5); }
    else if (link)        { handler = new MySimpleBrowser::StyleHandler(this, url, link->getAttribute("charset")); callback = bind(&MySimpleBrowser::StyleHandler::WGetResponseCB, (MySimpleBrowser::StyleHandler*)handler, _1, _2, _3, _4, _5); }

    if (handler) {
        INFO("Browser open '", url, "'");
        requested++;
        outstanding.insert(handler);
        Singleton<HTTPClient>::Get()->WGet(url, 0, callback);
    } else {
        ERROR("unknown mimetype for url ", url);
    }
}

Asset *SimpleBrowser::OpenImage(const string &url) {
    DOM::HTMLImageElement image(0);
    Open(url, &image);
    return image.asset;
}

void SimpleBrowser::OpenStyleImport(const string &url) {
    DOM::HTMLLinkElement link(0);
    Open(url, &link);
}

void SimpleBrowser::KeyEvent(int key, bool down) {}
void SimpleBrowser::MouseMoved(int x, int y) { mx=x; my=y; }
void SimpleBrowser::MouseButton(int b, bool d) { gui.Input(b, point(mx, my), d, 1); }
void SimpleBrowser::MouseWheel(int xs, int ys) {}
void SimpleBrowser::AnchorClicked(DOM::HTMLAnchorElement *anchor) {
    anchor->ownerDocument->ownerBrowser->Navigate(String::ToUTF8(anchor->getAttribute("href")));
}

bool SimpleBrowser::Dirty(Box *VP) {
    if (layers.empty()) return true;
    DOM::Node *n = doc.node->documentElement();
    Box viewport = Viewport();
    bool resized = viewport.w != VP->w || viewport.h != VP->h;
    if (resized && n) n->render->layout_dirty = true;
    return n && (n->render->layout_dirty || n->render->style_dirty);
}

void SimpleBrowser::Draw(Box *VP) { return Draw(VP, Dirty(VP)); }
void SimpleBrowser::Draw(Box *VP, bool dirty) {
    if (!VP || !doc.node || !doc.node->documentElement()) return;
    gui.box = *VP;
    bool tiles = layers.size();
    int v_scrolled = v_scrollbar.scrolled * v_scrollbar.doc_height;
    int h_scrolled = h_scrollbar.scrolled * 1000; // v_scrollbar.doc_height;
    if (dirty) {
        DOM::Renderer *html_render = doc.node->documentElement()->render;
        html_render->tiles = tiles ? layers[0] : 0;
        html_render->child_bg.Reset();
        Flow flow(html_render->box.Reset(), font, html_render->child_box.Reset());
        flow.p.y -= tiles ? 0 : v_scrolled;
        Draw(&flow, Viewport().TopLeft());
        doc.height = html_render->box.h;
        if (tiles) layers.Update();
    }
    for (int i=0; i<layers.size(); i++) layers[i]->Draw(Viewport(), !i ? h_scrolled : 0, !i ? -v_scrolled : 0);
}

void SimpleBrowser::Draw(Flow *flow, const point &displacement) {
    initial_displacement = displacement;
    DrawNode(flow, doc.node->documentElement(), initial_displacement);
    if (layers.empty()) DrawScrollbar();
}

void SimpleBrowser::DrawScrollbar() {
    gui.Draw();
    v_scrollbar.SetDocHeight(doc.height);
    v_scrollbar.Update();
    h_scrollbar.Update();
    gui.Activate();
}

void SimpleBrowser::DrawNode(Flow *flow, DOM::Node *n, const point &displacement_in) {
    if (!n) return;
    DOM::Renderer *render = n->render;
    ComputedStyle *style = &render->style;
    bool is_body  = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
    bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
    if (render->style_dirty || render->layout_dirty) { ScissorStack ss; LayoutNode(flow, n, 0); }
    if (render->display_none) return;

    if (!render->done_positioned) {
        render->done_positioned = 1;
        if (render->positioned) {
            DOM::CSSAutoNumericValue pt = style->Top(),  pb = style->Bottom();
            DOM::CSSAutoNumericValue pl = style->Left(), pr = style->Right();
            int ptp = pt.Null() ? 0 : pt.getPixelValue(flow), pbp = pb.Null() ? 0 : pb.getPixelValue(flow);
            int plp = pl.Null() ? 0 : pl.getPixelValue(flow), prp = pr.Null() ? 0 : pr.getPixelValue(flow);
            if (render->position_relative) { 
                if      (ptp) render->box.y -= ptp;
                else if (pbp) render->box.y += pbp;
                if      (plp) render->box.x += plp;
                else if (prp) render->box.x -= prp;
            } else {
                CHECK(!render->floating);
                Box viewport = Viewport();
                const Box *cont =
                    (render->position_fixed || render->absolute_parent == doc.node->documentElement()) ?
                    &viewport : &render->absolute_parent->render->box;

                if (render->height_auto && !pt._auto && !pb._auto) render->box.h = cont->h - render->MarginHeight() - ptp - pbp;
                if (render->width_auto  && !pl._auto && !pr._auto) render->box.w = cont->w - render->MarginWidth () - plp - prp;

                Box margin = Box::AddBorder(render->box, render->MarginOffset());
                if      (!pt._auto) margin.y =  0       - ptp - margin.h;
                else if (!pb._auto) margin.y = -cont->h + pbp;
                if      (!pl._auto) margin.x =  0       + plp;
                else if (!pr._auto) margin.x =  cont->w - prp - margin.w;
                render->box = Box::DelBorder(margin, render->MarginOffset());
            }
        }
    }

    point displacement = displacement_in + (render->block_level_box ? render->box.TopLeft() : point());
    if (render_log)              UpdateRenderLog(n);
    if (render->clip)            render->PushScissor(render->border.TopLeft(render->clip_rect) + displacement);
    if (render->overflow_hidden) render->PushScissor(render->content                           + displacement);
    if (!render->hidden) {
        if (render->tiles) {
            render->tiles->AddBoxArray(render->child_bg,  displacement);
            render->tiles->AddBoxArray(render->child_box, displacement);
        } else {
            render->child_bg .Draw(displacement);
            render->child_box.Draw(displacement);
        }
    }

    if (is_table) {
        DOM::HTMLTableElement *table = n->AsHTMLTableElement();
        HTMLTableRowIter(table)
            if (!row->render->Dirty()) HTMLTableColIter(row) 
                if (!cell->render->Dirty()) DrawNode(render->flow, cell, displacement);
    } else {
        DOM::NodeList *children = &n->childNodes;
        for (int i=0, l=children->length(); i<l; i++) {
            DOM::Node *child = children->item(i);
            if (!child->render->done_floated) DrawNode(render->flow, child, displacement);
        }
    }

    if (render->block_level_box) {
        for (int i=0; i<render->box.float_left .size(); i++) {
            if (render->box.float_left[i].inherited) continue;
            DOM::Node *child = (DOM::Node*)render->box.float_left[i].val;
            if (!child->render->done_positioned) {
                EX_EQ(render->box.float_left[i].w, child->render->margin.w);
                EX_EQ(render->box.float_left[i].h, child->render->margin.h);
                child->render->box.SetPosition(render->box.float_left[i].Position() - child->render->MarginPosition());
            }
            DrawNode(render->flow, child, displacement);
        }
        for (int i=0; i<render->box.float_right.size(); i++) {
            if (render->box.float_right[i].inherited) continue;
            DOM::Node *child = (DOM::Node*)render->box.float_right[i].val;
            if (!child->render->done_positioned) {
                EX_EQ(render->box.float_right[i].w, child->render->margin.w);
                EX_EQ(render->box.float_right[i].h, child->render->margin.h);
                child->render->box.SetPosition(render->box.float_right[i].Position() - child->render->MarginPosition());
            }
            DrawNode(render->flow, child, displacement);
        }
        if (render_log) render_log->indent -= 2;
    }
    render->Finish();
}

DOM::Node *SimpleBrowser::LayoutNode(Flow *flow, DOM::Node *n, bool reflow) {
    if (!n) return 0;
    DOM::Renderer *render = n->render;
    ComputedStyle *style = &render->style;
    bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
    bool table_element = render->display_table_element;

    if (!reflow) {
        if (render->style_dirty) render->UpdateStyle(flow);
        if (!style->is_root) render->tiles = n->parentNode->render->tiles;
        render->done_positioned = render->done_floated = 0;
        render->max_child_i = -1;
    } else if (render->done_floated) return 0;
    if (render->display_none) { render->style_dirty = render->layout_dirty = 0; return 0; }

    if (render->position_absolute) {
        DOM::Node *ap = n->parentNode;
        while (ap && ap->render && (!ap->render->positioned || !ap->render->block_level_box)) ap = ap->parentNode;
        render->absolute_parent = (ap && ap->render) ? ap : doc.node->documentElement();
    } else render->absolute_parent = 0;

    render->parent_flow = render->absolute_parent ? render->absolute_parent->render->flow : flow;
    if (!table_element) {
        render->box = style->is_root ? Viewport() : Box(max(0, flow->container->w), 0);
        render->UpdateDimensions(render->parent_flow);
    }

    render->clear_height = flow->container->AsFloatContainer()->ClearFloats(flow->p.y, flow->cur_line.height, render->clear_left, render->clear_right);
    if (render->clear_height) flow->AppendVerticalSpace(render->clear_height);

    if (style->font_not_inherited || !(render->child_flow.cur_attr.font = flow->cur_attr.font)) render->child_flow.cur_attr.font = render->UpdateFont(flow);
    if (style->is_root) flow->SetFont(render->child_flow.cur_attr.font);

    if (render->block_level_box) {
        if (!table_element) {
            if (!flow->cur_line.fresh && (render->normal_flow || render->position_absolute) && 
                !render->inline_block && !render->floating && !render->position_absolute) flow->AppendNewlines(1);
            render->box.y = flow->p.y;
        }
        render->child_flow = Flow(&render->box, render->child_flow.cur_attr.font, render->child_box.Reset());
        render->flow = &render->child_flow;
        if (!reflow) {
            render->box.Reset();
            if (render->normal_flow && !render->inline_block) render->box.InheritFloats(flow->container->AsFloatContainer());
            if (render->position_fixed && layers.size()) render->tiles = layers[1];
        }
    } else render->flow = render->parent_flow;
    render->UpdateFlowAttributes(render->flow);

    if (n->nodeType == DOM::TEXT_NODE) {
        DOM::Text *T = n->AsText();
        if (1)                            render->flow->SetFGColor(&render->color);
        // if (style->bgcolor_not_inherited) render->flow->SetBGColor(&render->background_color);
        render->flow->AppendText(T->data);
    } else if ((n->htmlElementType == DOM::HTML_IMAGE_ELEMENT) ||
               (n->htmlElementType == DOM::HTML_INPUT_ELEMENT && StringEquals(n->AsElement()->getAttribute("type"), "image"))) {
        Asset *asset = n->htmlElementType == DOM::HTML_IMAGE_ELEMENT ? n->AsHTMLImageElement()->asset : n->AsHTMLInputElement()->image_asset;
        bool missing = !asset || !asset->tex.ID;
        if (missing) asset = &missing_image;

        point dim(n->render->width_px, n->render->height_px);
        if      (!dim.x && !dim.y)                      dim   = point(asset->tex.width, asset->tex.height);
        if      ( dim.x && !dim.y && asset->tex.width ) dim.y = RoundF((float)asset->tex.height/asset->tex.width *dim.x);
        else if (!dim.x &&  dim.y && asset->tex.height) dim.x = RoundF((float)asset->tex.width /asset->tex.height*dim.y);

        bool add_margin = !n->render->floating && !n->render->position_absolute && !n->render->position_fixed;
        Border margin = add_margin ? n->render->MarginOffset() : Border();
        if (missing) {
            point d(max(0, dim.x - (int)asset->tex.width), max(0, dim.y - (int)asset->tex.height));
            margin += Border(d.y/2, d.x/2, d.y/2, d.x/2);
            dim -= d;
        }
        render->flow->SetFGColor(&Color::white);
        render->flow->AppendBox(dim.x, dim.y, &asset->tex);
        render->box = render->flow->out->data.back().box;
    } 
    else if (n->htmlElementType == DOM::HTML_BR_ELEMENT) render->flow->AppendNewlines(1);
    else if (is_table) LayoutTable(render->flow, n->AsHTMLTableElement());

    vector<Flow::RollbackState> child_flow_state;
    DOM::NodeList *children = &n->childNodes;
    for (int i=0, l=children->length(); !is_table && i<l; i++) {
        if (render->block_level_box) child_flow_state.push_back(render->flow->GetRollbackState());

        DOM::Node *child = children->item(i);
        bool reflow_child = i <= render->max_child_i;
        if (render->style_dirty  && !reflow_child) child->render->style_dirty  = 1;
        if (render->layout_dirty && !reflow_child) child->render->layout_dirty = 1;
        DOM::Node *descendent_float = LayoutNode(render->flow, child, reflow_child);
        Typed::Max(&render->max_child_i, i);
        if (!descendent_float) continue;
        if (!render->block_level_box) return descendent_float;

        Box df_margin;
        DOM::Renderer *dfr = descendent_float->render;
        CHECK_EQ(dfr->parent_flow, render->flow);
        int fy = render->flow->p.y - max(0, dfr->margin.h - render->flow->cur_line.height);
        render->box.AddFloat(fy, dfr->margin.w, dfr->margin.h, dfr->float_right, descendent_float, &df_margin);
        dfr->done_floated = 1;

        bool same_line = !render->flow->cur_line.fresh;
        int descendent_float_newlines = render->flow->out->line.size();
        while (child_flow_state.size() && render->flow->out->line.size() == descendent_float_newlines) {
            render->flow->Rollback(PopBack(child_flow_state));
            same_line = (render->flow->out->line.size() == descendent_float_newlines && !render->flow->cur_line.fresh);
        }
        EX_EQ(same_line, false);
        i = child_flow_state.size() - 1;
    }

    if (render->block_level_box) {
        render->flow->Complete();
        int block_height = X_or_Y(render->height_px, max(render->flow->Height(), render->establishes_block ? render->box.FloatHeight() : 0));
        int block_width  = render->shrink ? render->ClampWidth(render->flow->max_line_width) : render->box.w;
        if (render->normal_flow) {
            if (render->inline_block) render->parent_flow->AppendBox  (block_width, block_height, render->MarginOffset(), &render->box);
            else                      render->parent_flow->AppendBlock(block_width, block_height, render->MarginOffset(), &render->box); 
            if (!render->establishes_block) render->box.AddFloatsToParent((FloatContainer*)render->parent_flow->container->AsFloatContainer());
        } else {
            render->box.SetDimension(point(block_width, block_height));
            if (style->is_root || render->position_absolute || table_element) render->box.y -= block_height;
        }
    }

    if (render->background_image) render->UpdateBackgroundImage(flow);
    if (style->bgcolor_not_inherited || render->inline_block || render->clip || render->floating
        || render->bs_t || render->bs_b || render->bs_r || render->bs_l) {
        render->content = Box(0, -render->box.h, render->box.w, render->box.h);
        render->padding = Box::AddBorder(render->content, render->PaddingOffset());
        render->border  = Box::AddBorder(render->content, render->BorderOffset());
        render->margin  = Box::AddBorder(render->content, render->MarginOffset());
    }
    LayoutBackground(n);

    render->style_dirty = render->layout_dirty = 0;
    return (render->floating && !render->done_floated) ? n : 0;
}

void SimpleBrowser::LayoutBackground(DOM::Node *n) {
    bool is_body = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
    DOM::Renderer *render = n->render;
    ComputedStyle *style = &render->style;
    Flow flow(0, 0, render->child_bg.Reset());

    const Box                                                         *box = &render->content;
    if      (is_body)                                                  box = &render->margin;
    else if (render->block_level_box || render->display_table_element) box = &render->padding;

    if (style->bgcolor_not_inherited && render->background_color.A()) {
        if (is_body) screen->gd->ClearColor(Color(render->background_color, 0.0));
        else {
            flow.SetFGColor(&render->background_color);
            flow.out->PushBack(*box, flow.cur_attr, Singleton<BoxFilled>::Get());
        }
    } else if (is_body) screen->gd->ClearColor(Color(1.0, 1.0, 1.0, 0.0));

    if (render->background_image && render->background_image->tex.width && 
                                    render->background_image->tex.height) {
        Box bgi(box->x     + render->bgposition_x, 
                box->top() - render->bgposition_y, 
                render->background_image->tex.width,
                render->background_image->tex.height);

        flow.cur_attr.Clear();
        flow.cur_attr.scissor = box;
        // flow.cur_attr.tex = &render->background_image->tex;

        int start_x = !n->render->bgrepeat_x ? bgi.x       : box->x - bgi.w + (n->render->bgposition_x % bgi.w);
        int start_y = !n->render->bgrepeat_y ? bgi.y       : box->y - bgi.h + (n->render->bgposition_y % bgi.h);
        int end_x   = !n->render->bgrepeat_x ? bgi.right() : box->right();
        int end_y   = !n->render->bgrepeat_y ? bgi.top  () : box->top  ();
        for     (int y = start_y; y < end_y; y += bgi.h) { bgi.y = y;
            for (int x = start_x; x < end_x; x += bgi.w) { bgi.x = x;
                flow.out->PushBack(bgi, flow.cur_attr, &render->background_image->tex);
            }
        }
    }

    flow.cur_attr.Clear();
    if (render->bs_t) {
        flow.SetFGColor(&render->border_top);
        flow.out->PushBack(Box(render->border.x, render->padding.top(), render->border.w, render->border.top() - render->padding.top()),
                           flow.cur_attr, Singleton<BoxFilled>::Get());
    }
    if (render->bs_b) {
        flow.SetFGColor(&render->border_bottom);
        flow.out->PushBack(Box(render->border.x, render->border.y, render->border.w, render->padding.y - render->border.y),
                           flow.cur_attr, Singleton<BoxFilled>::Get());
    }
    if (render->bs_r) {
        flow.SetFGColor(&render->border_right);
        flow.out->PushBack(Box(render->padding.right(), render->border.y, render->border.right() - render->padding.right(), render->border.h),
                           flow.cur_attr, Singleton<BoxFilled>::Get());
    }
    if (render->bs_l) {
        flow.SetFGColor(&render->border_left);
        flow.out->PushBack(Box(render->border.x, render->border.y, render->padding.x - render->border.x, render->border.h),
                           flow.cur_attr, Singleton<BoxFilled>::Get());
    }
}

void SimpleBrowser::LayoutTable(Flow *flow, DOM::HTMLTableElement *n) {
    UpdateTableStyle(flow, n);
    TableFlow table(flow);
    table.Select();

    HTMLTableIterColGroup(n)
        table.SetMinColumnWidth(j, col->render->MarginBoxWidth(), X_or_1(atoi(col->getAttribute("span"))));

    HTMLTableRowIter(n) {
        HTMLTableColIter(row) {
            DOM::Renderer *cr = cell->render;
            TableFlow::Column *cj = table.SetCellDim(j, cr->MarginBoxWidth(), cr->cell_colspan, cr->cell_rowspan);
            if (!n->render->width_auto || !cr->width_auto) continue;
            cr->box = Box(0,0); LayoutNode(flow, cell, 0); Typed::Max(&cj->max_width, cr->flow->max_line_width + cr->MarginWidth()); 
            cr->box = Box(1,0); LayoutNode(flow, cell, 0); Typed::Max(&cj->min_width, cr->flow->max_line_width + cr->MarginWidth());
        }
        table.NextRowDim();
    }

    n->render->box.w = table.ComputeWidth(n->render->width_auto ? 0 : n->render->width_px);
    if (n->render->width_auto) n->render->UpdateMarginWidth(n->render->parent_flow, n->render->box.w);

    HTMLTableRowIter(n) {
        HTMLTableColIter(row) {
            DOM::Renderer *cr = cell->render;
            table.AppendCell(j, &cr->box, cr->cell_colspan);
            cr->box.DelBorder(cr->MarginOffset().LeftRight());
            LayoutNode(flow, cell, 0);
            table.SetCellHeight(j, cr->box.h + cr->MarginHeight(), cell, cr->cell_colspan, cr->cell_rowspan);
        }
        row->render->row_height = table.AppendRow();
        row->render->layout_dirty = 0;
    }
}

void SimpleBrowser::UpdateTableStyle(Flow *flow, DOM::Node *n) {
    for (int i=0, l=n->childNodes.length(); i<l; i++) {
        DOM::Node *child = n->childNodes.item(i);
        DOM::HTMLTableCellElement *cell = child->AsHTMLTableCellElement();
        if (n->render->style_dirty) child->render->style_dirty = 1;
        if (!child->AsHTMLTableRowElement() && !child->AsHTMLTableSectionElement() &&
            !child->AsHTMLTableColElement() && !child->AsHTMLTableCaptionElement() && !cell) continue;

        if (child->render->style_dirty) {
            child->render->UpdateStyle(flow);
            child->render->display_table_element = 1;
            child->render->tiles = child->parentNode->render->tiles;
            child->render->UpdateDimensions(flow);
            if (cell) {
                cell->render->cell_colspan = X_or_1(atoi(cell->getAttribute("colspan")));
                cell->render->cell_rowspan = X_or_1(atoi(cell->getAttribute("rowspan")));
            }
        }
        UpdateTableStyle(flow, child);
    }
    n->render->style_dirty = 0;
}

void SimpleBrowser::UpdateRenderLog(DOM::Node *n) {
    DOM::Renderer *render = n->render;
    if (render_log->data.empty()) {
        CHECK(render->style.is_root);
        Box viewport = Viewport();
        render_log->indent = 2;
        render_log->data = StrCat("layer at (0,0) size ",          viewport.w, "x", viewport.h,
                                  "\n  RenderView at (0,0) size ", viewport.w, "x", viewport.h,
                                  "\nlayer at (0,0) size ", render->box.w, "x", render->box.h, "\n");
    }
    if (render->block_level_box) {
        StrAppend(&render_log->data, string(render_log->indent, ' '), "RenderBlock {", toupper(n->nodeName()), "} at (",
                  render->box.x, ",", ScreenToWebKitY(render->box), ") size ", render->box.w, "x", render->box.h, "\n");
        render_log->indent += 2;
    }
    if (n->nodeType == DOM::TEXT_NODE) {
        StrAppend(&render_log->data, string(render_log->indent, ' '),
                  "text run at (", render->box.x, ",",    ScreenToWebKitY(render->box),
                  ") width ",      render->box.w, ": \"", n->nodeValue(), "\"\n");
    }
}

struct BrowserController : public InputController {
    BrowserInterface *browser;
    BrowserController(BrowserInterface *B) : browser(B) {}
    void Input(InputEvent::Id event, bool down) {
        int key = InputEvent::GetKey(event);
        if (key)                                browser->KeyEvent(key, down);
        else if (event == Mouse::Event::Motion) browser->MouseMoved(screen->mouse.x, screen->mouse.y);
        else if (event == Mouse::Event::Wheel)  browser->MouseWheel(0, down*32);
        else                                    browser->MouseButton(event, down);
    }
};

#ifdef LFL_QT
extern QApplication *lfl_qapp;
class QTWebKitBrowser : public QObject, public BrowserInterface {
    Q_OBJECT
  public:
    Asset* asset;
    int W, H, lastx, lasty, lastd;
    QWebPage page;

    QTWebKitBrowser(Asset *a) : lastx(0), lasty(0), lastd(0) {
        asset = a;
        Resize(screen->width, screen->height);
        lfl_qapp->connect(&page, SIGNAL(repaintRequested(const QRect&)),           this, SLOT(ViewUpdate(const QRect &)));
        lfl_qapp->connect(&page, SIGNAL(scrollRequested(int, int, const QRect &)), this, SLOT(Scroll(int, int, const QRect &)));
    }
    void Resize(int win, int hin) {
        W = win; H = hin;
        page.setViewportSize(QSize(W, H));
        if (!asset->tex.ID) asset->tex.CreateBacked(W, H);
        else                asset->tex.Resize(W, H);
    }
    void Open(const string &url) { page.mainFrame()->load(QUrl(url.c_str())); }
    void MouseMoved(int x, int y) {
        lastx = x; lasty = screen->height - y; 
        QMouseEvent me(QEvent::MouseMove, QPoint(lastx, lasty), Qt::NoButton,
                       lastd ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(&page, &me);
    }
    void MouseButton(int b, bool d) {
        lastd = d;
        QMouseEvent me(QEvent::MouseButtonPress, QPoint(lastx, lasty), Qt::LeftButton,
                       lastd ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(&page, &me);
    }
    void MouseWheel(int xs, int ys) {}
    void KeyEvent(int key, bool down) {}
    void BackButton() { page.triggerAction(QWebPage::Back); }
    void ForwardButton() { page.triggerAction(QWebPage::Forward);  }
    void RefreshButton() { page.triggerAction(QWebPage::Reload); }
    string GetURL() { return page.mainFrame()->url().toString().toLocal8Bit().data(); }
    void Draw(Box *w) {
        asset->tex.DrawCrimped(*w, 1, 0, 0);
        QPoint scroll = page.currentFrame()->scrollPosition();
        QWebHitTestResult hit_test = page.currentFrame()->hitTestContent(QPoint(lastx, lasty));
        QWebElement hit = hit_test.element();
        QRect rect = hit.geometry();
        Box hitwin(rect.x() - scroll.x(), w->h-rect.y()-rect.height()+scroll.y(), rect.width(), rect.height());
        screen->gd->SetColor(Color::blue);
        hitwin.Draw();
    }
  public slots:
    void Scroll(int dx, int dy, const QRect &rect) { ViewUpdate(QRect(0, 0, W, H)); }
    void ViewUpdate(const QRect &dirtyRect) {
        QImage image(page.viewportSize(), QImage::Format_ARGB32);
        QPainter painter(&image);
        page.currentFrame()->render(&painter, dirtyRect);
        painter.end();

        int instride = image.bytesPerLine(), bpp = Pixel::size(asset->tex.pf);
        int mx = dirtyRect.x(), my = dirtyRect.y(), mw = dirtyRect.width(), mh = dirtyRect.height();
        const unsigned char *in = (const unsigned char *)image.bits();
        unsigned char *buf = asset->tex.buf;
        screen->gd->BindTexture(GraphicsDevice::Texture2D, asset->tex.ID);
        for (int j = 0; j < mh; j++) {
            int src_ind = (j + my) * instride + mx * bpp;
            int dst_ind = ((j + my) * asset->tex.width + mx) * bpp;
            if (true) VideoResampler::RGB2BGRCopyPixels(buf+dst_ind, in+src_ind, mw, bpp);
            else                                 memcpy(buf+dst_ind, in+src_ind, mw*bpp);
        }
        asset->tex.UpdateGL(0, my, asset->tex.width, dirtyRect.height());
    }
};

BrowserInterface *CreateQTWebKitBrowser(Asset *a) { return new QTWebKitBrowser(a); }

#include "gui.moc"

/* Dialogs */

void Dialog::MessageBox(const string &n) {
    Mouse::ReleaseFocus();
    QMessageBox *msg = new QMessageBox();
    msg->setAttribute(Qt::WA_DeleteOnClose);
    msg->setText("MesssageBox");
    msg->setInformativeText(n.c_str());
    msg->setModal(false);
    msg->open();
}
void Dialog::TextureBox(const string &n) {}

#else /* LFL_QT */
BrowserInterface *CreateQTWebKitBrowser(Asset *a) { return 0; }

void Dialog::MessageBox(const string &n) {
    Mouse::ReleaseFocus();
    new MessageBoxDialog(n);
}
void Dialog::TextureBox(const string &n) {
    Mouse::ReleaseFocus();
    new TextureBoxDialog(n);
}
#endif /* LFL_QT */

#ifdef LFL_BERKELIUM
struct BerkeliumModule : public Module {
    BerkeliumModule() { app->LoadModule(this); }
    int Frame(unsigned t) { Berkelium::update(); return 0; }
    int Free() { Berkelium::destroy(); return 0; }
    int Init() {
        const char *homedir = LFAppDownloadDir();
        INFO("berkelium init");
        Berkelium::init(
#ifdef _WIN32
            homedir ? Berkelium::FileString::point_to(wstring(homedir).c_str(), wstring(homedir).size()) :
#else
            homedir ? Berkelium::FileString::point_to(homedir, strlen(homedir)) :
#endif
            Berkelium::FileString::empty());
        return 0;
    }
};

struct BerkeliumBrowser : public BrowserInterface, public Berkelium::WindowDelegate {
    Asset *asset; int W, H; Berkelium::Window *window;
    BerkeliumBrowser(Asset *a, int win, int hin) : asset(a), window(0) { 
        Singleton<BerkeliumModule>::Get();
        Berkelium::Context* context = Berkelium::Context::create();
        window = Berkelium::Window::create(context);
        delete context;
        window->setDelegate(this);
        resize(win, hin); 
    }

    void Draw(Box *w) { glOverlay(*w, asset, 1); }
    void Open(const string &url) { window->navigateTo(Berkelium::URLString::point_to(url.data(), url.length())); }
    void MouseMoved(int x, int y) { window->mouseMoved((float)x/screen->width*W, (float)(screen->height-y)/screen->height*H); }
    void MouseButton(int b, bool down) {
        if (b == BIND_MOUSE1 && down) {
            int click_x = screen->mouse.x/screen->width*W, click_y = (screen->height - screen->mouse.y)/screen->height*H;
            string js = WStringPrintf(L"lfapp_browser_click(document.elementFromPoint(%d, %d).innerHTML)", click_x, click_y);
            window->executeJavascript(Berkelium::WideString::point_to((wchar_t *)js.c_str(), js.size()));
        }
        window->mouseButton(b == BIND_MOUSE1 ? 0 : 1, down);
    }
    void MouseWheel(int xs, int ys) { window->mouseWheel(xs, ys); }
    void KeyEvent(int key, bool down) {
        int bk = -1;
        switch (key) {
            case Key::PageUp:    bk = 0x21; break;
            case Key::PageDown:  bk = 0x22; break;
            case Key::Left:      bk = 0x25; break;
            case Key::Up:        bk = 0x26; break;
            case Key::Right:     bk = 0x27; break;
            case Key::Down:      bk = 0x28; break;
            case Key::Delete:    bk = 0x2E; break;
            case Key::Backspace: bk = 0x2E; break;
            case Key::Return:    bk = '\r'; break;
            case Key::Tab:       bk = '\t'; break;
        }

        int mods = ctrl_key_down() ? Berkelium::CONTROL_MOD : 0 | shift_key_down() ? Berkelium::SHIFT_MOD : 0;
        window->keyEvent(down, mods, bk != -1 ? bk : key, 0);
        if (!down || bk != -1) return;

        wchar_t wkey[2] = { key, 0 };
        window->textEvent(wkey, 1);
    }

    void Resize(int win, int hin) {
        W = win; H = hin;
        window->resize(W, H);
        if (!asset->texID) PixelBuffer::create(W, H, asset);
        else               asset->tex.resize(W, H, asset->texcoord);
    }

    virtual void onPaint(Berkelium::Window* wini, const unsigned char *in_buf, const Berkelium::Rect &in_rect,
        size_t num_copy_rects, const Berkelium::Rect* copy_rects, int dx, int dy, const Berkelium::Rect& scroll_rect) {
        unsigned char *buf = asset->tex.buf
        int bpp = Pixel::size(asset->tex.pf);
        screen->gd->BindTexture(GL_TEXTURE_2D, asset->texID);

        if (dy < 0) {
            const Berkelium::Rect &r = scroll_rect;
            for (int j = -dy, h = r.height(); j < h; j++) {
                int src_ind = ((j + r.top()     ) * asset->tex.w + r.left()) * bpp;
                int dst_ind = ((j + r.top() + dy) * asset->tex.w + r.left()) * bpp;
                memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
            }
        } else if (dy > 0) {
            const Berkelium::Rect &r = scroll_rect;
            for (int j = r.height() - dy; j >= 0; j--) {
                int src_ind = ((j + r.top()     ) * asset->tex.w + r.left()) * bpp;
                int dst_ind = ((j + r.top() + dy) * asset->tex.w + r.left()) * bpp;                
                memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
            }
        }
        if (dx) {
            const Berkelium::Rect &r = scroll_rect;
            for (int j = 0, h = r.height(); j < h; j++) {
                int src_ind = ((j + r.top()) * asset->tex.w + r.left()     ) * bpp;
                int dst_ind = ((j + r.top()) * asset->tex.w + r.left() - dx) * bpp;
                memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
            }
        }
        for (int i = 0; i < num_copy_rects; i++) {
            const Berkelium::Rect &r = copy_rects[i];
            for(int j = 0, h = r.height(); j < h; j++) {
                int dst_ind = ((j + r.top()) * asset->tex.w + r.left()) * bpp;
                int src_ind = ((j + (r.top() - in_rect.top())) * in_rect.width() + (r.left() - in_rect.left())) * bpp;
                memcpy(buf+dst_ind, in_buf+src_ind, r.width()*bpp);
            }
            asset->tex.flush(0, r.top(), asset->tex.w, r.height());        
        }

        // for (Berkelium::Window::FrontToBackIter iter = wini->frontIter(); 0 && iter != wini->frontEnd(); iter++) { Berkelium::Widget *w = *iter; }
    }

    virtual void onCreatedWindow(Berkelium::Window *win, Berkelium::Window *newWindow, const Berkelium::Rect &initialRect) { newWindow->setDelegate(this); }
    virtual void onAddressBarChanged(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onAddressBarChanged: ", newURL.data()); }
    virtual void onStartLoading(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onStartLoading: ", newURL.data()); }
    virtual void onLoad(Berkelium::Window *win) { INFO("onLoad"); }
    virtual void onJavascriptCallback(Berkelium::Window *win, void* replyMsg, Berkelium::URLString url, Berkelium::WideString funcName, Berkelium::Script::Variant *args, size_t numArgs) {
        string fname = WideStringToString(funcName);
        if (fname == "lfapp_browser_click" && numArgs == 1 && args[0].type() == args[0].JSSTRING) {
            string a1 = WideStringToString(args[0].toString());
            INFO("jscb: ", fname, " ", a1);
        } else {
            INFO("jscb: ", fname);
        }
        if (replyMsg) win->synchronousScriptReturn(replyMsg, numArgs ? args[0] : Berkelium::Script::Variant());
    }
    virtual void onRunFileChooser(Berkelium::Window *win, int mode, Berkelium::WideString title, Berkelium::FileString defaultFile) { win->filesSelected(NULL); }
    virtual void onNavigationRequested(Berkelium::Window *win, Berkelium::URLString newURL, Berkelium::URLString referrer, bool isNewWindow, bool &cancelDefaultAction) {}

    static string WideStringToString(const Berkelium::WideString& in) {
        string out; out.resize(in.size());
        for (int i = 0; i < in.size(); i++) out[i] = in.data()[i];     
        return out;
    }
};

BrowserInterface *CreateBerkeliumBrowser(Asset *a, int W, int H) {
    BerkeliumBrowser *browser = new BerkeliumBrowser(a, W, H);
    Berkelium::WideString click = Berkelium::WideString::point_to(L"lfapp_browser_click");
    browser->window->bind(click, Berkelium::Script::Variant::bindFunction(click, true));
    return browser;
}
#else /* LFL_BERKELIUM */
BrowserInterface *CreateBerkeliumBrowser(Asset *a, int W, int H) { return 0; }
#endif /* LFL_BERKELIUM */

}; // namespace LFL
#ifdef LFL_BOOST
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/simple_point.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/progress.hpp>
#include <boost/shared_ptr.hpp>

using namespace boost;

struct BoostForceDirectedLayout {
    typedef adjacency_list<listS, vecS, undirectedS, property<vertex_name_t, std::string> > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;

    typedef boost::rectangle_topology<> topology_type;
    typedef topology_type::point_type point_type;
    typedef std::vector<point_type> PositionVec;
    typedef iterator_property_map<PositionVec::iterator, property_map<Graph, vertex_index_t>::type> PositionMap;
    
    Graph g;
    vector<Vertex> v;

    void Clear() { g.clear(); v.clear(); }
    void AssignVertexPositionToTargetCenter(const HelperGUI *gui, PositionMap *position, int vertex_offset) {
        for (int i = 0; i < gui->label.size(); ++i) {
            (*position)[v[i*2+vertex_offset]][0] = gui->label[i].target_center.x;
            (*position)[v[i*2+vertex_offset]][1] = gui->label[i].target_center.y;
        }
    }
    void AssignVertexPositionToLabelCenter(const HelperGUI *gui, PositionMap *position, int vertex_offset) {
        for (int i = 0; i < gui->label.size(); ++i) {
            (*position)[v[i*2+vertex_offset]][0] = gui->label[i].label_center.x;
            (*position)[v[i*2+vertex_offset]][1] = gui->label[i].label_center.y;
        }
    }
    void Layout(HelperGUI *gui) {
        Clear();

        for (int i = 0; i < gui->label.size()*2; ++i) v.push_back(add_vertex(StringPrintf("%d", i), g));
        for (int i = 0; i < gui->label.size()  ; ++i) add_edge(v[i*2], v[i*2+1], g);

        PositionVec position_vec(num_vertices(g));
        PositionMap position(position_vec.begin(), get(vertex_index, g));

        minstd_rand gen;
        topology_type topo(gen, 0, 0, screen->width, screen->height);
        if (0) random_graph_layout(g, position, topo);
        else AssignVertexPositionToLabelCenter(gui, &position, 1);

        for (int i = 0; i < 300; i++) {
            AssignVertexPositionToTargetCenter(gui, &position, 0);
            fruchterman_reingold_force_directed_layout(g, position, topo, cooling(linear_cooling<double>(1)));
        }

        for (int i = 0; i < gui->label.size(); ++i) {
            HelperGUI::Label *l = &gui->label[i];
            l->label_center.x = position[v[i*2+1]][0];
            l->label_center.y = position[v[i*2+1]][1];
            l->AssignLabelBox();
        }
    }
};
#endif // LFL_BOOST
namespace LFL {

void HelperGUI::ForceDirectedLayout() {
#ifdef LFL_BOOST
    BoostForceDirectedLayout().Layout(this);
#endif
}

void HelperGUI::Draw() {
    for (auto i = label.begin(); i != label.end(); ++i) {
        glLine(i->label_center.x, i->label_center.y, i->target_center.x, i->target_center.y, &font->fg);
        screen->gd->FillColor(Color::black);
        Box::AddBorder(i->label, 4, 0).Draw();
        font->Draw(i->description, point(i->label.x, i->label.y));
    }
}

}; // namespace LFL
