/*
 * $Id: terminal.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef LFL_CORE_APP_GL_TERMINAL_H__
#define LFL_CORE_APP_GL_TERMINAL_H__
namespace LFL {

struct Terminal : public TextArea {
  struct State { enum { TEXT=0, ESC=1, CSI=2, OSC=3, CHARSET=4 }; };
  struct ByteSink {
    virtual ~ByteSink() {}
    virtual int Write(const StringPiece &b) = 0;
    virtual void IOCtlWindowSize(int w, int h) {}
  };
  struct Controller : public ByteSink {
    bool ctrl_down=0, alt_down=0, frame_on_keyboard_input=0;
    virtual ~Controller() {}
    virtual int Open(TextArea*) = 0;
    virtual StringPiece Read() = 0;
    virtual void Close() {}
    virtual void Dispose() {}
  };
  struct TerminalColors : public Colors {
    Color c[16 + 3];
    TerminalColors() { normal_index=16; bold_index=17; background_index=18; }
    const Color *GetColor(int n) const { CHECK_RANGE(n, 0, sizeofarray(c)); return &c[n]; }
  };
  struct StandardVGAColors       : public TerminalColors { StandardVGAColors(); };
  struct SolarizedDarkColors     : public TerminalColors { SolarizedDarkColors(); };
  struct SolarizedLightColors    : public TerminalColors { SolarizedLightColors(); };

  ByteSink *sink=0;
  int term_width=0, term_height=0, parse_state=State::TEXT;
  int scroll_region_beg=0, scroll_region_end=0, tab_width=8;
  string parse_text, parse_csi, parse_osc;
  unsigned char parse_charset=0;
  bool parse_osc_escape=0, first_resize=1, newline_mode=0;
  char erase_char = 0x7f, enter_char = '\r'; 
  point term_cursor=point(1,1), saved_term_cursor=point(1,1);
  LinesFrameBuffer::FromLineCB fb_cb;
  LinesFrameBuffer *last_fb=0;
  Border clip_border;
  set<int> tab_stop;

  Terminal(ByteSink *O, Window *W, const FontRef &F=FontRef(), const point &dim=point(1,1));
  virtual ~Terminal() {}
  virtual void Resized(const Box &b, bool font_size_changed=false);
  virtual void ResizedLeftoverRegion(int w, int h, bool update_fb=true);
  virtual void SetScrollRegion(int b, int e, bool release_fb=false);
  virtual void SetTerminalDimension(int w, int h);
  virtual void Draw(const Box &b, int flag=DrawFlag::Default, Shader *shader=0);
  virtual void Write(const StringPiece &s, bool update_fb=true, bool release_fb=true);
  virtual void Input(char k) {                       sink->Write(StringPiece(&k, 1)); }
  virtual void Erase      () {                       sink->Write(StringPiece(&erase_char, 1)); }
  virtual void Enter      () {                       sink->Write(StringPiece(&enter_char, 1)); }
  virtual void Tab        () { char k = '\t';        sink->Write(StringPiece(&k, 1)); }
  virtual void Escape     () { char k = 0x1b;        sink->Write(StringPiece(&k, 1)); }
  virtual void HistUp     () { char k[] = "\x1bOA";  sink->Write(StringPiece( k, 3)); }
  virtual void HistDown   () { char k[] = "\x1bOB";  sink->Write(StringPiece( k, 3)); }
  virtual void CursorRight() { char k[] = "\x1bOC";  sink->Write(StringPiece( k, 3)); }
  virtual void CursorLeft () { char k[] = "\x1bOD";  sink->Write(StringPiece( k, 3)); }
  virtual void PageUp     () { char k[] = "\x1b[5~"; sink->Write(StringPiece( k, 4)); }
  virtual void PageDown   () { char k[] = "\x1b[6~"; sink->Write(StringPiece( k, 4)); }
  virtual void Home       () { char k[] = "\x1bOH";  sink->Write(StringPiece( k, 3)); }
  virtual void End        () { char k[] = "\x1bOF";  sink->Write(StringPiece( k, 3)); }
  virtual void MoveToOrFromScrollRegion(LinesFrameBuffer *fb, Line *l, const point &p, int flag);
  // virtual int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) { return 0; }
  virtual void UpdateCursor() { cursor.p = point(GetCursorX(term_cursor.x, term_cursor.y), GetCursorY(term_cursor.y)); }
  virtual void UpdateToken(Line*, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox>*);
  virtual bool GetGlyphFromCoords(const point &p, Selection::Point *out) { return GetGlyphFromCoordsOffset(p, out, clip ? 0 : start_line, 0); }
  virtual void ScrollUp  () { TextArea::PageDown(); }
  virtual void ScrollDown() { TextArea::PageUp(); }
  int GetFrameY(int y) const;
  int GetCursorY(int y) const;
  int GetCursorX(int x, int y) const;
  int GetTermLineIndex(int y) const { return -term_height + y-1; }
  const Line *GetTermLine(int y) const { return &line[GetTermLineIndex(y)]; }
  /**/  Line *GetTermLine(int y)       { return &line[GetTermLineIndex(y)]; }
  Line *GetCursorLine() { return GetTermLine(term_cursor.y); }
  LinesFrameBuffer *GetPrimaryFrameBuffer()   { return line_fb.Attach(&last_fb); }
  LinesFrameBuffer *GetSecondaryFrameBuffer() { return cmd_fb .Attach(&last_fb); }
  LinesFrameBuffer *GetFrameBuffer(const Line *l);
  void PushBackLines (int n) { TextArea::Write(string(n, '\n'), true, false); }
  void PushFrontLines(int n) { for (int i=0; i<n; ++i) LineUpdate(line.InsertAt(-term_height, 1, start_line_adjust), GetPrimaryFrameBuffer(), LineUpdate::PushFront); }
  point GetCursorPosition() const { return point(0, -scrolled_lines * style.font->Height()) + cursor.p; }
  Border *UpdateClipBorder();
  void MoveLines(int sy, int ey, int dy, bool move_fb_p);
  void Scroll(int sl);
  void FlushParseText();
  void Newline(bool carriage_return=false);
  void NewTopline();
  void TabNext(int n);
  void TabPrev(int n);
  void Redraw(bool attach=true, bool relayout=false);
  void ResetTerminal();
  void ClearTerminal();
};

}; // namespace LFL
#endif // LFL_CORE_APP_GL_TERMINAL_H__
