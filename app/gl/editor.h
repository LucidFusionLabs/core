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

#ifndef LFL_CORE_APP_GL_EDITOR_H__
#define LFL_CORE_APP_GL_EDITOR_H__
namespace LFL {

struct Editor : public TextView {
  struct LineOffset { 
    long long file_offset=-1;
    int file_size=0, wrapped_lines=0, annotation_ind=-1, main_tu_line=-1, next_tu_line=-1;
    SyntaxMatch::ListPointer syntax_region_start_parent={0,0}, syntax_region_end_parent={0,0};
    vector<SyntaxMatch::State> syntax_region_start_sig, syntax_region_end_sig;
    vector<SyntaxMatch::List> syntax_region_ancestor_list_storage;
    bool colored=0;
    LineOffset(int O=0, int S=0, int WL=1, int AI=-1) :
      file_offset(O), file_size(S), wrapped_lines(WL), annotation_ind(AI) {}

    static string GetString(const LineOffset *v) { return StrCat(v->file_offset); }
    static int    GetLines (const LineOffset *v) { return v->wrapped_lines; }
    static int VectorGetLines(const vector<LineOffset> &v, int i) { return v[i].wrapped_lines; }
  };
  typedef PrefixSumKeyedRedBlackTree<int, LineOffset> LineMap;
  struct SyntaxColors : public Colors, public SyntaxStyleInterface {
    struct Rule { string name; Color fg, bg; int style; };
    string name;
    vector<Color> color;
    unordered_map<string, int> style;
    SyntaxColors(const string &n, const vector<Rule>&);
    virtual const Color *GetColor(int n) const override { CHECK_RANGE(n, 0, color.size()); return &color[n]; }
    virtual int GetSyntaxStyle(const string &n, int da) override { return FindOrDefault(style, n, da); }
    const Color *GetFGColor(const string &n) { return GetColor(Style::GetFGColorIndex(GetSyntaxStyle(n, SetDefaultAttr(0)))); }
    const Color *GetBGColor(const string &n) { return GetColor(Style::GetBGColorIndex(GetSyntaxStyle(n, SetDefaultAttr(0)))); }
  };
  struct Base16DefaultDarkSyntaxColors : public SyntaxColors { Base16DefaultDarkSyntaxColors(); };
  struct Modification { point p; bool erase; String16 data; };
  struct VersionNumber {
    int major, offset;
    bool operator==(const VersionNumber &x) const { return major == x.major && offset == x.offset; }
    bool operator!=(const VersionNumber &x) const { return major != x.major || offset != x.offset; }
  };

  shared_ptr<File> file;
  LineMap file_line;
  FreeListVector<String16> edits;
  Time modified=Time(0);
  Callback modified_cb, newline_cb, tab_cb;
  vector<Modification> version;
  VersionNumber version_number={0,0}, saved_version_number={0,0}, cached_text_version_number={-1,0};
  function<const DrawableAnnotation*(const LineMap::Iterator&, const String16&, bool, int, int)> annotation_cb;
  shared_ptr<BufferFile> cached_text;
  Line *cursor_glyphs=0;
  LineOffset *cursor_offset=0;
  int cursor_anchor=0, cursor_line_index=0, cursor_start_line_number=0, cursor_start_line_number_offset=0;
  int syntax_parsed_anchor=0, syntax_parsed_line_index=0;
  bool opened=0;
  virtual ~Editor();
  Editor(Window *W, const FontRef &F=FontRef(), unique_ptr<File> I=unique_ptr<File>());

  bool Init(unique_ptr<File> I) { return (opened = (file = shared_ptr<File>(I.release())) && file->Opened()); }
  void Input(char k) override { Modify(k,    false); }
  void Enter()       override { Modify('\n', false); }
  void Erase()       override { Modify(0,    true); }
  void CursorLeft()  override { UpdateCursorX(max(cursor.i.x-1, 0)); }
  void CursorRight() override { UpdateCursorX(min(cursor.i.x+1, CursorGlyphsSize())); }
  void Home()        override { UpdateCursorX(0); }
  void End()         override { UpdateCursorX(CursorGlyphsSize()); }
  void Tab()         override { if (tab_cb) tab_cb(); }
  void HistUp()      override;
  void HistDown()    override;
  void SelectionCB(const Selection::Point&);
  bool Empty() const override { return !file_line.size(); }
  void UpdateMapping(int width, int flag=0) override;
  int UpdateMappedLines(pair<int, int>, bool, bool, bool, bool, bool) override;
  int UpdateLines(float vs, int *first_ind, int *first_offset, int *first_len) override;

  int CursorGlyphsSize() const { return cursor_glyphs ? cursor_glyphs->Size() : 0; }
  uint16_t CursorGlyph() const { String16 v = CursorLineGlyphs(cursor.i.x, 1); return v.empty() ? 0 : v[0]; }
  String16 CursorLineGlyphs(size_t o, size_t l) const { return cursor_glyphs ? cursor_glyphs->data->glyphs.Text16(o, l) : String16(); }
  void MarkCursorLineFirstDirty() { syntax_parsed_line_index=cursor_line_index; syntax_parsed_anchor=cursor_anchor; }
  const String16 *ReadLine(const Editor::LineMap::Iterator &ui, String16 *buf);
  void SetWrapMode(const string &n);
  void SetShouldWrap(bool v, bool word_break);
  void UpdateCursor() override;
  void UpdateCursorLine();
  void UpdateCursorX(int x) override;
  int CursorLinesChanged(const String16 &b, int add_lines=0);
  int ModifyCursorLine();
  void Modify(char16_t, bool erase, bool undo_or_redo=false);
  int Save();
  int SaveTo(File *out);
  bool CacheModifiedText(bool force=false);
  void RecordModify(const point &p, bool erase, char16_t c);
  bool WalkUndo(bool backwards);
  bool ScrollTo(int line_index, int x);
};

struct EditorDialog : public TextViewDialogT<Editor> {
  struct Flag { enum { Wrap=Dialog::Flag::Next }; };
  EditorDialog(Window *W, const FontRef &F, unique_ptr<File> I, float w=.5, float h=.5, int flag=0) :
    TextViewDialogT(W, "EditorDialog", F, w, h, flag) {
    if (I) { title_text = BaseName(I->Filename()); view.Init(move(I)); }
    view.line_fb.wrap = flag & Flag::Wrap; 
  }
};

}; // namespace LFL
#endif // LFL_CORE_APP_GL_EDITOR_H__
