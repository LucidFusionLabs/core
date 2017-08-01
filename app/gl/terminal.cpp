/*
 * $Id: terminal.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "core/app/gl/view.h"
#include "core/app/gl/terminal.h"

#ifdef  LFL_TERMINAL_DEBUG
#define TerminalDebug(...) ERRORf(__VA_ARGS__)
#define TerminalTrace(...) DebugPrintf("%s", StrCat(logtime(Now()), " ", StringPrintf(__VA_ARGS__)).c_str())
#else
#define TerminalDebug(...)
#define TerminalTrace(...)
#endif

namespace LFL {
Terminal::StandardVGAColors::StandardVGAColors() { 
  c[0] = Color(  0,   0,   0);  c[ 8] = Color( 85,  85,  85);
  c[1] = Color(170,   0,   0);  c[ 9] = Color(255,  85,  85);
  c[2] = Color(  0, 170,   0);  c[10] = Color( 85, 255,  85);
  c[3] = Color(170,  85,   0);  c[11] = Color(255, 255,  85);
  c[4] = Color(  0,   0, 170);  c[12] = Color( 85,  85, 255);
  c[5] = Color(170,   0, 170);  c[13] = Color(255,  85, 255);
  c[6] = Color(  0, 170, 170);  c[14] = Color( 85, 255, 255);
  c[7] = Color(170, 170, 170);  c[15] = Color(255, 255, 255);
  c[normal_index]     = c[7];
  c[bold_index]       = c[15];
  c[background_index] = c[0];
}

/// Solarized palette by Ethan Schoonover
Terminal::SolarizedDarkColors::SolarizedDarkColors() { 
  c[0] = Color(  7,  54,  66); /*base02*/   c[ 8] = Color(  0,  43,  54); /*base03*/
  c[1] = Color(220,  50,  47); /*red*/      c[ 9] = Color(203,  75,  22); /*orange*/
  c[2] = Color(133, 153,   0); /*green*/    c[10] = Color( 88, 110, 117); /*base01*/
  c[3] = Color(181, 137,   0); /*yellow*/   c[11] = Color(101, 123, 131); /*base00*/
  c[4] = Color( 38, 139, 210); /*blue*/     c[12] = Color(131, 148, 150); /*base0*/
  c[5] = Color(211,  54, 130); /*magenta*/  c[13] = Color(108, 113, 196); /*violet*/
  c[6] = Color( 42, 161, 152); /*cyan*/     c[14] = Color(147, 161, 161); /*base1*/
  c[7] = Color(238, 232, 213); /*base2*/    c[15] = Color(253, 246, 227); /*base3*/
  c[normal_index]     = c[12];
  c[bold_index]       = c[14];
  c[background_index] = c[8];
}

Terminal::SolarizedLightColors::SolarizedLightColors() { 
  c[0] = Color(238, 232, 213); /*base2*/    c[ 8] = Color(253, 246, 227); /*base3*/
  c[1] = Color(220,  50,  47); /*red*/      c[ 9] = Color(203,  75,  22); /*orange*/
  c[2] = Color(133, 153,   0); /*green*/    c[10] = Color(147, 161, 161); /*base1*/
  c[3] = Color(181, 137,   0); /*yellow*/   c[11] = Color(131, 148, 150); /*base0*/
  c[4] = Color( 38, 139, 210); /*blue*/     c[12] = Color(101, 123, 131); /*base00*/
  c[5] = Color(211,  54, 130); /*magenta*/  c[13] = Color(108, 113, 196); /*violet*/
  c[6] = Color( 42, 161, 152); /*cyan*/     c[14] = Color( 88, 110, 117); /*base01*/
  c[7] = Color(  7,  54,  66); /*base02*/   c[15] = Color(  0,  43,  54); /*base03*/
  c[normal_index]     = c[12];
  c[bold_index]       = c[14];
  c[background_index] = c[8];
}

Terminal::Terminal(ByteSink *O, Window *W, const FontRef &F, const point &dim) :
  TextArea(W, F, 200, 0), sink(O), fb_cb(bind(&Terminal::GetFrameBuffer, this, _1)) {
  CHECK(style.font->fixed_width || (style.font->flag & FontDesc::Mono));
  wrap_lines = write_newline = insert_mode = 0;
  line.SetAttrSource(&style);
  line_fb.align_top_or_bot = cmd_fb.align_top_or_bot = false;
  line_fb.partial_last_line = cmd_fb.partial_last_line = false;
  SetColors(Singleton<SolarizedDarkColors>::Get());
  cursor.attr = default_attr;
  token_processing = 1;
  cmd_prefix = "";
  SetTerminalDimension(dim.x, dim.y);
  Activate();
}

int Terminal::GetFrameY(int y) const { return (term_height - y + 1) * style.font->Height(); }
int Terminal::GetCursorY(int y) const { return GetFrameY(y) + (line_fb.align_top_or_bot ? extra_height : 0); }
int Terminal::GetCursorX(int x, int y) const {
  const Line *l = GetTermLine(y);
  return x <= l->Size() ? l->data->glyphs.Position(x-1).x : ((x-1) * style.font->FixedWidth());
}

Terminal::LinesFrameBuffer *Terminal::GetFrameBuffer(const Line *l) {
  int i = line.IndexOf(l);
  return ((-i-1 < start_line || term_height+i < skip_last_lines) ? cmd_fb : line_fb).Attach(&last_fb);
}

void Terminal::Resized(const Box &b, bool font_size_changed) {
  int old_term_width = term_width, old_term_height = term_height;
  SetTerminalDimension(b.w / style.font->FixedWidth(), b.h / style.font->Height());
  TerminalDebug("Resized %d, %d <- %d, %d", term_width, term_height, old_term_width, old_term_height);
  bool grid_changed = term_width != old_term_width || term_height != old_term_height;
  if (grid_changed || first_resize) if (sink) sink->IOCtlWindowSize(term_width, term_height); 

  int height_dy = term_height - old_term_height;
  if (!line_fb.only_grow) {
    if      (height_dy > 0) TextArea::Write(string(height_dy, '\n'), 0);
    else if (height_dy < 0 && term_cursor.y < old_term_height)
      line.PopBack(min(-height_dy, old_term_height - term_cursor.y));
  } else {
    if      (height_dy > 0) { for (int i=0; i<height_dy; i++) line.PushFront(); }
    else if (height_dy < 0) line.PopFront(-height_dy);
    term_cursor.y += height_dy;
  }

  term_cursor.x = min(term_cursor.x, max(1, term_width));
  term_cursor.y = min(term_cursor.y, max(1, term_height));
  TextArea::Resized(b, font_size_changed);
  if (clip) clip = UpdateClipBorder();
  ResizedLeftoverRegion(b.w, b.h);
}

void Terminal::ResizedLeftoverRegion(int w, int h, bool update_fb) {
  if (!cmd_fb.SizeChanged(w, h, style.font, bg_color)) return;
  if (update_fb) {
    for (int i=0; i<start_line;      i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-i-1],           point(0, GetFrameY(term_height-i)), LinesFrameBuffer::Flag::Flush);
    for (int i=0; i<skip_last_lines; i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-term_height+i], point(0, GetFrameY(i+1)),           LinesFrameBuffer::Flag::Flush);
  }
  cmd_fb.SizeChangedDone();
  last_fb = 0;
}

void Terminal::MoveToOrFromScrollRegion(TextBox::LinesFrameBuffer *fb, TextBox::Line *l, const point &p, int flag) {
  int plpy = l->p.y;
  fb->Update(l, p, flag);
  bool last_outside_scroll_region = l->data->outside_scroll_region;
  if ((l->data->outside_scroll_region = fb != &line_fb) != last_outside_scroll_region)
    l->data->AddControlsDelta(plpy - l->p.y + line_fb.h * (l->data->outside_scroll_region ? -1 : 1));
}

void Terminal::SetScrollRegion(int b, int e, bool release_fb) {
  if (b<0 || e<0 || e>term_height || b>e) { TerminalDebug("%d-%d outside 1-%d", b, e, term_height); return; }
  int prev_region_beg = scroll_region_beg, prev_region_end = scroll_region_end, font_height = style.font->Height();
  scroll_region_beg = b;
  scroll_region_end = e;
  bool no_region = !scroll_region_beg || !scroll_region_end || (scroll_region_beg == 1 && scroll_region_end == term_height);
  skip_last_lines = no_region ? 0 : scroll_region_beg - 1;
  start_line_adjust = start_line = no_region ? 0 : term_height - scroll_region_end;
  clip = no_region ? 0 : UpdateClipBorder();
  ResizedLeftoverRegion(line_fb.w, line_fb.h);
  int   prev_beg_or_1=X_or_1(prev_region_beg),     prev_end_or_ht=X_or_Y(  prev_region_end, term_height);
  int scroll_beg_or_1=X_or_1(scroll_region_beg), scroll_end_or_ht=X_or_Y(scroll_region_end, term_height);

  if (scroll_beg_or_1 != prev_beg_or_1 || prev_end_or_ht != scroll_end_or_ht) GetPrimaryFrameBuffer();
  for (int i =  scroll_beg_or_1; i <    prev_beg_or_1; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i),   line_fb.BackPlus(point(0, (term_height-i+1)*font_height)), LinesFrameBuffer::Flag::NoLayout);
  for (int i =   prev_end_or_ht; i < scroll_end_or_ht; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i+1), line_fb.BackPlus(point(0, (term_height-i)  *font_height)), LinesFrameBuffer::Flag::NoLayout);

  if (prev_beg_or_1 < scroll_beg_or_1 || scroll_end_or_ht < prev_end_or_ht) GetSecondaryFrameBuffer();
  for (int i =    prev_beg_or_1; i < scroll_beg_or_1; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i),   point(0, GetFrameY(i)),   LinesFrameBuffer::Flag::NoLayout);
  for (int i = scroll_end_or_ht; i <  prev_end_or_ht; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i+1), point(0, GetFrameY(i+1)), LinesFrameBuffer::Flag::NoLayout);
  if (release_fb) { cmd_fb.fb.Release(); last_fb=0; }
}

void Terminal::SetTerminalDimension(int w, int h) {
  term_width  = max(line_fb.only_grow ? term_width  : 0, w);
  term_height = max(line_fb.only_grow ? term_height : 0, h);
  ScopedClearColor scc(line_fb.fb.gd, bg_color);
  if (!line.Size()) TextArea::Write(string(term_height, '\n'), 0);
}

Border *Terminal::UpdateClipBorder() {
  int font_height = style.font->Height();
  clip_border.bottom = font_height * start_line_adjust;
  clip_border.top    = font_height * skip_last_lines;
  if (line_fb.align_top_or_bot) { if (clip_border.bottom) clip_border.bottom += extra_height; }
  else                          { if (clip_border.top)    clip_border.top    += extra_height; }
  return &clip_border;
}

void Terminal::MoveLines(int sy, int ey, int dy, bool move_fb_p) {
  if (sy == ey) return;
  CHECK_LT(sy, ey);
  int line_ind = GetTermLineIndex(sy), scroll_lines = ey - sy + 1, ady = abs(dy), sdy = (dy > 0 ? 1 : -1);
  Move(line, line_ind + (dy>0 ? dy : 0), line_ind + (dy<0 ? -dy : 0), scroll_lines - ady, move_fb_p ? line.movep_cb : line.move_cb);
  for (int i = 0, cy = (dy>0 ? sy : ey); i < ady; i++) GetTermLine(cy + i*sdy)->Clear();
}

void Terminal::Scroll(int sl) {
  bool up = sl<0;
  if (!clip) return up ? PushBackLines(-sl) : PushFrontLines(sl);
  int beg_y = scroll_region_beg, offset = up ? start_line           : skip_last_lines;
  int end_y = scroll_region_end, flag   = up ? LineUpdate::PushBack : LineUpdate::PushFront;
  MoveLines(beg_y, end_y, sl, true);
  GetPrimaryFrameBuffer();
  for (int i=0, l=abs(sl); i<l; i++)
    LineUpdate(GetTermLine(up ? (end_y-l+i+1) : (beg_y+l-i-1)), &line_fb, flag, offset);
}

void Terminal::UpdateToken(Line *L, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox> *token) {
  CHECK_LE(word_offset + word_len, L->data->glyphs.Size());
  Line *BL = L, *EL = L, *NL;
  int in_line_ind = -line.IndexOf(L)-1, beg_line_ind = in_line_ind, end_line_ind = in_line_ind;
  int beg_offset = word_offset, end_offset = word_offset + word_len - 1, new_offset, gs;

  for (; !beg_offset && beg_line_ind < term_height; beg_line_ind++, beg_offset = term_width - new_offset, BL = NL) {
    const DrawableBoxArray &glyphs = (NL = &line[-beg_line_ind-1-1])->data->glyphs;
    if ((gs = glyphs.Size()) != term_width || !(new_offset = RLengthChar(&glyphs[gs-1], notspace, gs))) break;
  }
  for (; end_offset == term_width-1 && end_line_ind >= 0; end_line_ind--, end_offset = new_offset - 1, EL = NL) {
    const DrawableBoxArray &glyphs = (NL = &line[-beg_line_ind-1+1])->data->glyphs;
    if (!(new_offset = LengthChar(glyphs.data.data(), notspace, glyphs.Size()))) break;
  }

  string text = CopyText(beg_line_ind, beg_offset, end_line_ind, end_offset, 0);
  if (update_type < 0) UpdateLongToken(BL, beg_offset, EL, end_offset, text, update_type);

  if (BL != L && !word_offset && token->lbw != token->nlbw)
    UpdateLongToken(BL, beg_offset, &line[-in_line_ind-1-1], term_width-1,
                    CopyText(beg_line_ind, beg_offset, in_line_ind+1, term_width-1, 0), update_type * -10);
  if (EL != L && word_offset + word_len == term_width && token->lew != token->nlew)
    UpdateLongToken(&line[-in_line_ind-1+1], 0, EL, end_offset,
                    CopyText(in_line_ind-1, 0, end_line_ind, end_offset, 0), update_type * -11);

  if (update_type > 0) UpdateLongToken(BL, beg_offset, EL, end_offset, text, update_type);
}

void Terminal::Draw(const Box &b, int flag, Shader *shader) {
  float scale = 0;
  GraphicsContext gc(root->gd);
  TextArea::Draw(b, flag & ~DrawFlag::DrawCursor, shader);
  if (shader) {
    scale = shader->scale;
    shader->SetUniform2f("iChannelScroll", 0, XY_or_Y(scale, -line_fb.align_top_or_bot * extra_height) - b.y);
  }
  if (clip) {
    if (scale) { Box ub(b); ub.y /= scale; ub = Box::TopBorder(ub, *clip); ub.y *= scale; Scissor s(gc.gd, ub); cmd_fb.DrawAligned(ub, point()); }
    else       { Scissor s(gc.gd, Box::TopBorder(b, *clip, scale)); cmd_fb.DrawAligned(b, point()); }
    if (1)     { Scissor s(gc.gd, Box::BotBorder(b, *clip, scale)); cmd_fb.DrawAligned(b, point()); }
    if (hover_control) DrawHoverLink(b);
  }
  if (flag & DrawFlag::DrawCursor) TextBox::DrawCursor(b.Position() + GetCursorPosition());
  if (extra_height && !shader) {
    ScopedFillColor c(gc.gd, *bg_color);
    gc.DrawTexturedBox(Box(b.x, b.y + (line_fb.align_top_or_bot ? 0 : line_fb.h), b.w, extra_height));
  }
  if (selection.changing) DrawSelection();
}

void Terminal::Write(const StringPiece &s, bool update_fb, bool release_fb) {
  if (!app->MainThread()) return app->RunInMainThread(bind(&Terminal::WriteCB, this, s.str(), update_fb, release_fb));
  TerminalTrace("Terminal: Write('%s', %zd)", CHexEscapeNonAscii(s.str()).c_str(), s.size());
  line_fb.fb.gd->DrawMode(DrawMode::_2D, 0);
  ScopedClearColor scc(line_fb.fb.gd, bg_color);
  last_fb = 0;
  for (int i = 0; i < s.len; i++) {
    const unsigned char c = *(s.begin() + i);
    if (c == 0x18 || c == 0x1a) { /* CAN or SUB */ parse_state = State::TEXT; continue; }
    if (parse_state == State::ESC) {
      parse_state = State::TEXT; // default next state
      TerminalTrace("ESC %c", c);
      if (c >= '(' && c <= '+') {
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
        case '=': case '>':                        break; // application or normal keypad
        case 'c': ResetTerminal();                 break;
        case 'D': Newline();                       break;
        case 'M': NewTopline();                    break;
        case '7': saved_term_cursor = term_cursor; break;
        case '8': term_cursor = point(Clamp(saved_term_cursor.x, 1, term_width),
                                      Clamp(saved_term_cursor.y, 1, term_height));
        default: TerminalDebug("unhandled escape %c (%02x)", c, c);
      }
    } else if (parse_state == State::CHARSET) {
      TerminalTrace("charset G%d %c", 1+parse_charset-'(', c);
      parse_state = State::TEXT;

    } else if (parse_state == State::OSC) {
      if (!parse_osc_escape) {
        if (c == 0x1b) { parse_osc_escape = 1; continue; }
        if (c != 0x07) { parse_osc       += c; continue; }
      }
      else if (c != 0x5c) { TerminalDebug("within-OSC-escape %c (%02x)", c, c); parse_state = State::TEXT; continue; }
      parse_state = State::TEXT;

      if (parse_osc.size() > 2 && parse_osc[1] == ';' && Within(parse_osc[0], '0', '2')) root->SetCaption(parse_osc.substr(2));
      else TerminalDebug("unhandled OSC %s", parse_osc.c_str());

    } else if (parse_state == State::CSI) {
      // http://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
      if (c < 0x40 || c > 0x7e) { parse_csi += c; continue; }
      TerminalTrace("CSI %s%c (cur=%d,%d)", parse_csi.c_str(), c, term_cursor.x, term_cursor.y);
      parse_state = State::TEXT;

      int parsed_csi=0, parse_csi_argc=0, parse_csi_argv[16];
      unsigned char parse_csi_argv00 = parse_csi.empty() ? 0 : (isdigit(parse_csi[0]) ? 0 : parse_csi[0]);
      for (/**/; Within<int>(parse_csi[parsed_csi], 0x20, 0x2f); parsed_csi++) {}
      StringPiece intermed(parse_csi.data(), parsed_csi);

      memzeros(parse_csi_argv);
      bool parse_csi_arg_done = 0;
      // http://www.inwap.com/pdp10/ansicode.txt 
      for (/**/; Within<int>(parse_csi[parsed_csi], 0x30, 0x3f); parsed_csi++) {
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
          l->UpdateText(term_cursor.x-1, StringPiece(string(X_or_1(parse_csi_argv[0]), ' ')), cursor.attr, term_width, 0, 1);
        } break;
        case 'A': term_cursor.y = max(term_cursor.y - X_or_1(parse_csi_argv[0]), 1);           break;
        case 'B': term_cursor.y = min(term_cursor.y + X_or_1(parse_csi_argv[0]), term_height); break;
        case 'C': term_cursor.x = min(term_cursor.x + X_or_1(parse_csi_argv[0]), term_width);  break;
        case 'D': term_cursor.x = max(term_cursor.x - X_or_1(parse_csi_argv[0]), 1);           break;
        case 'd': term_cursor.y = Clamp(parse_csi_argv[0], 1, term_height);                    break;
        case 'G': term_cursor.x = Clamp(parse_csi_argv[0], 1, term_width);                     break;
        case 'H': term_cursor = point(Clamp(parse_csi_argv[1], 1, term_width),
                                      Clamp(parse_csi_argv[0], 1, term_height)); break;
        case 'Z': TabPrev(parse_csi_argv[0]); break;
        case 'J': {
          int clear_beg_y = 1, clear_end_y = term_height;
          if      (parse_csi_argv[0] == 0) { LineUpdate(GetCursorLine(), fb_cb)->Erase(term_cursor.x-1);  clear_beg_y = term_cursor.y; }
          else if (parse_csi_argv[0] == 1) { LineUpdate(GetCursorLine(), fb_cb)->Erase(0, term_cursor.x); clear_end_y = term_cursor.y; }
          else if (parse_csi_argv[0] == 2) { ClearTerminal(); break; }
          for (int i = clear_beg_y; i <= clear_end_y; i++) LineUpdate(GetTermLine(i), fb_cb)->Clear();
        } break;
        case 'K': {
          LineUpdate l(GetCursorLine(), fb_cb);
          if      (parse_csi_argv[0] == 0) l->Erase(term_cursor.x-1);
          else if (parse_csi_argv[0] == 1) l->Erase(0, term_cursor.x);
          else if (parse_csi_argv[0] == 2) l->Clear();
        } break;
        case 'S': case 'T': Scroll((c == 'T' ? 1 : -1) * X_or_1(parse_csi_argv[0])); break;
        case 'L': case 'M': { /* insert or delete lines */
          int sl = (c == 'L' ? 1 : -1) * X_or_1(parse_csi_argv[0]);
          int beg_y = term_cursor.y, end_y = X_or_Y(scroll_region_end, term_height);
          if (beg_y == X_or_1(scroll_region_beg)) { Scroll(sl); break; }
          if (clip && beg_y < scroll_region_beg)
          { TerminalDebug("y=%s scrollregion=%d-%d", term_cursor.DebugString().c_str(), scroll_region_beg, scroll_region_end); break; }
          MoveLines(beg_y, end_y, sl, false);
          GetPrimaryFrameBuffer();
          for (int i=beg_y; i<=end_y; i++) LineUpdate(GetTermLine(i), &line_fb);
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
          else if (parse_csi_argv00 == '?' && mode ==    7) { /* wrap around mode */        }
          else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = true;        }
          else if (parse_csi_argv00 == '?' && mode ==   47) { /* alternate screen buffer */ }
          else if (parse_csi_argv00 == '?' && mode == 1034) { /* meta mode: 8th bit on */   }
          else if (parse_csi_argv00 == '?' && mode == 1049) { /* save screen */             }
          else TerminalDebug("unhandled CSI-h mode = %d av00 = %c i= %s", mode, parse_csi_argv00, intermed.str().c_str());
        } break;
        case 'l': { // reset mode
          int mode = parse_csi_argv[0];
          if      (parse_csi_argv00 == 0   && mode ==    3) { /* 80 column mode */                 }
          else if (parse_csi_argv00 == 0   && mode ==    4) { insert_mode = false;                 }
          else if (parse_csi_argv00 == 0   && mode ==   34) { /* blink cursor */                   }
          else if (parse_csi_argv00 == '?' && mode ==    1) { /* guarded area tx = unprot only */  }
          else if (parse_csi_argv00 == '?' && mode ==    7) { /* no wrap around mode */            }
          else if (parse_csi_argv00 == '?' && mode ==   12) { /* steady cursor */                  }
          else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = false;              }
          else if (parse_csi_argv00 == '?' && mode ==   47) { /* normal screen buffer */           }
          else if (parse_csi_argv00 == '?' && mode == 1049) { /* restore screen */                 }
          else TerminalDebug("unhandled CSI-l mode = %d av00 = %c i= %s", mode, parse_csi_argv00, intermed.str().c_str());
        } break;
        case 'm':
        for (int i=0; i<parse_csi_argc; i++) {
          int sgr = parse_csi_argv[i]; // select graphic rendition
          if      (sgr >= 30 && sgr <= 37) cursor.attr = Style::SetFGColorIndex(cursor.attr, sgr-30);
          else if (sgr >= 40 && sgr <= 47) cursor.attr = Style::SetBGColorIndex(cursor.attr, sgr-40);
          else switch(sgr) {
            case 0:         cursor.attr  =  default_attr;    break;
            case 1:         cursor.attr |=  Style::Bold;      break;
            case 3:         cursor.attr |=  Style::Italic;    break;
            case 4:         cursor.attr |=  Style::Underline; break;
            case 5: case 6: cursor.attr |=  Style::Blink;     break;
            case 7:         cursor.attr |=  Style::Reverse;   break;
            case 22:        cursor.attr &= ~Style::Bold;      break;
            case 23:        cursor.attr &= ~Style::Italic;    break;
            case 24:        cursor.attr &= ~Style::Underline; break;
            case 25:        cursor.attr &= ~Style::Blink;     break;
            case 27:        cursor.attr &= ~Style::Reverse;   break;
            case 39:        cursor.attr = Style::SetFGColorIndex(cursor.attr, style.colors->normal_index);     break;
            case 49:        cursor.attr = Style::SetBGColorIndex(cursor.attr, style.colors->background_index); break;
            default:        TerminalDebug("unhandled SGR %d", sgr);
          }
        } break;
        case 'p':
          if (parse_csi_argv00 == '!') { /* soft reset http://vt100.net/docs/vt510-rm/DECSTR */
            insert_mode = false;
            SetScrollRegion(1, term_height);
          }
          else TerminalDebug("Unhandled CSI-p %c", parse_csi_argv00);
          break;
        case 'r':
          if (parse_csi_argc == 2) SetScrollRegion(parse_csi_argv[0], parse_csi_argv[1]);
          else TerminalDebug("invalid scroll region argc %d", parse_csi_argc);
          break;
        default:
          TerminalDebug("unhandled CSI %s%c", parse_csi.c_str(), c);
      }
    } else {
      // http://en.wikipedia.org/wiki/C0_and_C1_control_codes#C0_.28ASCII_and_derivatives.29
      bool C0_control = (c >= 0x00 && c <= 0x1f) || c == 0x7f;
      bool C1_control = (c >= 0x80 && c <= 0x9f);
      if (C0_control || C1_control) {
        TerminalTrace("C0/C1 control: %02x", c);
        FlushParseText();
      }
      if (C0_control) switch(c) {
        case '\a':   TerminalDebug("%s", "bell");                     break; // bell
        case '\b':   term_cursor.x = max(term_cursor.x-1, 1);         break; // backspace
        case '\t':   TabNext(1);                                      break; // tab 
        case '\r':   term_cursor.x = 1;                               break; // carriage return
        case '\x1b': parse_state = State::ESC;                        break;
        case '\x14': case '\x15': case '\x7f':                        break; // shift charset in, out, delete
        case '\n':   case '\v':   case '\f':   Newline(newline_mode); break; // line feed, vertical tab, form feed
        default:                               TerminalDebug("unhandled C0 control %02x", c);
      } else if (0 && C1_control) {
        if (0) {}
        else TerminalDebug("unhandled C1 control %02x", c);
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
  bool append = 0;
  int consumed = 0, write_size = 0, update_size = 0;
  CHECK_GT(term_cursor.x, 0);
  style.font.ptr = style.GetAttr(cursor.attr)->font;
  String16 input_text = String::ToUTF16(parse_text, &consumed);
  TerminalTrace("Terminal: (cur=%d,%d) FlushParseText('%s').size = [%zd, %d]", term_cursor.x, term_cursor.y,
                CHexEscapeNonAscii(StringPiece(parse_text.data(), consumed)).c_str(), input_text.size(), consumed);
  for (int wrote = 0; wrote < input_text.size(); wrote += write_size) {
    if (wrote) Newline(true);
    else if (term_cursor.x > term_width) { GetCursorLine()->data->wrapped = true; Newline(true); }
    Line *l = GetCursorLine();
    LinesFrameBuffer *fb = GetFrameBuffer(l);
    int remaining = input_text.size() - wrote, o = term_cursor.x-1, tl = term_width - o;
    write_size = min(remaining, tl);
    if (write_size == tl && remaining > write_size) l->data->wrapped = true;
    String16Piece input_piece(input_text.data() + wrote, write_size);
    update_size = l->UpdateText(o, input_piece, cursor.attr, term_width, &append);
    TerminalTrace("Terminal: FlushParseText: UpdateText(%d, %d, '%s').size = [%d, %d] attr=%d",
                  term_cursor.x, term_cursor.y, String::ToUTF8(input_piece).c_str(), write_size, update_size, cursor.attr);
    if (!update_size) continue;
    l->data->outside_scroll_region = fb != &line_fb;
    l->Layout();
    if (!fb->lines) continue;
    int s = l->Size(), ol = s - o, sx = l->data->glyphs.LeftBound(o), ex = l->data->glyphs.RightBound(s-1);
    if (append) l->Draw(l->p, -1, o, ol);
    else LinesFrameBuffer::Paint(l, point(sx, l->p.y), Box(-sx, 0, ex - sx, fb->font_height), o, ol);
  }
  term_cursor.x += update_size;
  parse_text.erase(0, consumed);
}

void Terminal::Newline(bool carriage_return) {
  if (clip && term_cursor.y == scroll_region_end) Scroll(-1);
  else if (term_cursor.y == term_height) { if (!clip) PushBackLines(1); }
  else term_cursor.y = min(term_height, term_cursor.y+1);
  if (carriage_return) term_cursor.x = 1;
}

void Terminal::NewTopline() {
  if (clip && term_cursor.y == scroll_region_beg) Scroll(1);
  else if (term_cursor.y == 1) { if (!clip) PushFrontLines(1); }
  else term_cursor.y = max(1, term_cursor.y-1);
}

void Terminal::TabNext(int n) { term_cursor.x = min(NextMultipleOfN(term_cursor.x, tab_width) + 1, term_width); }
void Terminal::TabPrev(int n) {
  if (tab_stop.size()) TerminalDebug("%s", "variable tab stop not implemented");
  else if ((term_cursor.x = max(PrevMultipleOfN(term_cursor.x - max(0, n-1)*tab_width - 2, tab_width), 1)) != 1) term_cursor.x++;
}

void Terminal::Redraw(bool attach, bool relayout) {
  bool prev_clip = clip;
  int prev_scroll_beg = scroll_region_beg, prev_scroll_end = scroll_region_end;
  SetScrollRegion(1, term_height, true);
  TextArea::Redraw(true, relayout);
  if (prev_clip) SetScrollRegion(prev_scroll_beg, prev_scroll_end, true);
}

void Terminal::ResetTerminal() {
  term_cursor.x = term_cursor.y = 1;
  scroll_region_beg = scroll_region_end = 0;
  clip = 0;
  ClearTerminal();
}

void Terminal::ClearTerminal() {
  for (int i=1; i<=term_height; ++i) GetTermLine(i)->Clear();
  Redraw(true);
}

}; // namespace LFL
