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

#include "core/app/gl/view.h"
#include "core/app/gl/editor.h"

namespace LFL {
Editor::SyntaxColors::SyntaxColors(const string &n, const vector<Rule> &rules) :
  name(n), color(3, Color()) {
  normal_index=0; bold_index=1; background_index=2;
  map<Color, pair<int, int>> seen_fg, seen_bg;
  for (auto &r : rules) {
    auto fs = &seen_fg[r.fg], bs = &seen_bg[r.bg];
    if (!fs->second++) fs->first = PushBackIndex(color, r.fg);
    if (!bs->second++) bs->first = PushBackIndex(color, r.bg);
    style[r.name] = Style::SetColorIndex(r.style, fs->first, bs->first);
  }
  auto i = style.find("Normal");
  if (i != style.end()) {
    color[normal_index] = color[bold_index] = color[Style::GetFGColorIndex(i->second)];
    color[background_index]                 = color[Style::GetBGColorIndex(i->second)];
  } else {
    ERROR("SyntaxColors ", name, " no Normal");
    int max_fg=0, max_bg=0, max_fg_ind=0, max_bg_ind=0;
    for (auto &f : seen_fg) if (Max(&max_fg, f.second.second)) max_fg_ind = f.second.first;
    for (auto &b : seen_bg) if (Max(&max_bg, b.second.second)) max_bg_ind = b.second.first;
    color[normal_index] = color[bold_index] = color[max_fg_ind];
    color[background_index]                 = color[max_bg_ind];
  }
  for (auto &c : color) if (c == Color::clear) c = color[background_index];
}

// base16 by chriskempson
Editor::Base16DefaultDarkSyntaxColors::Base16DefaultDarkSyntaxColors() :
  SyntaxColors("Base16DefaultDarkSyntaxColors", {
    { "Boolean",      Color("dc9656"), Color("181818"), 0 },
    { "Character",    Color("ab4642"), Color("181818"), 0 },
    { "Comment",      Color("585858"), Color("181818"), 0 },
    { "Conditional",  Color("ba8baf"), Color("181818"), 0 },
    { "Constant",     Color("dc9656"), Color("181818"), 0 },
    { "Define",       Color("ba8baf"), Color("181818"), 0 },
    { "Delimiter",    Color("a16946"), Color("181818"), 0 },
    { "Float",        Color("dc9656"), Color("181818"), 0 },
    { "Function",     Color("7cafc2"), Color("181818"), 0 },
    { "Identifier",   Color("ab4642"), Color("181818"), 0 },
    { "Include",      Color("7cafc2"), Color("181818"), 0 },
    { "Keyword",      Color("ba8baf"), Color("181818"), 0 },
    { "Label",        Color("f7ca88"), Color("181818"), 0 },
    { "Normal",       Color("d8d8d8"), Color("181818"), 0 },
    { "Number",       Color("dc9656"), Color("181818"), 0 },
    { "Operator",     Color("d8d8d8"), Color("181818"), 0 },
    { "PreCondition", Color("ba8baf"), Color("181818"), 0 },
    { "PreProc",      Color("f7ca88"), Color("181818"), 0 },
    { "Repeat",       Color("f7ca88"), Color("181818"), 0 },
    { "Special",      Color("86c1b9"), Color("181818"), 0 },
    { "SpecialChar",  Color("a16946"), Color("181818"), 0 },
    { "Statement",    Color("ab4642"), Color("181818"), 0 },
    { "StatusLine",   Color("b8b8b8"), Color("383838"), 0 },
    { "StorageClass", Color("f7ca88"), Color("181818"), 0 },
    { "String",       Color("a1b56c"), Color("181818"), 0 },
    { "Structure",    Color("ba8baf"), Color("181818"), 0 },
    { "Tag",          Color("f7ca88"), Color("181818"), 0 },
    { "Todo",         Color("f7ca88"), Color("282828"), 0 },
    { "Type",         Color("f7ca88"), Color("181818"), 0 },
    { "Typedef",      Color("f7ca88"), Color("181818"), 0 }
  }) {}

Editor::~Editor() {}
Editor::Editor(Window *W, const FontRef &F, unique_ptr<File> I) : TextView(W, F),
  file_line(&LineOffset::GetLines, &LineOffset::GetString),
  annotation_cb([](const LineMap::ConstIterator&, const String16&, bool, int cs, int){ static DrawableAnnotation a; return cs ? nullptr : &a; }) {
  cmd_color = Color(Color::black, .5);
  edits.free_func = [](String16 *v) { v->clear(); };
  selection_cb = bind(&Editor::SelectionCB, this, _1);
  if (I) Init(move(I));
}

const String16 *Editor::ReadLine(const Editor::LineMap::Iterator &i, String16 *buf) {
  if (int e = (max(0, -i.val->file_size))) return &edits[e-1];
  *buf = String::ToUTF16(file->ReadString(i.val->file_offset, i.val->file_size));
  return buf;
}

void Editor::SetWrapMode(const string &n) {
  bool lines_mode = (n == "lines"), words_mode = (n == "words"), should_wrap = lines_mode || words_mode;
  SetShouldWrap(should_wrap, should_wrap ? words_mode : layout.word_break);
}

void Editor::SetShouldWrap(bool w, bool word_break) {
  layout.word_break = word_break;
  line_fb.wrap = w;
  Reload();
  Redraw(true);
}

void Editor::HistUp() {
  if (cursor.i.y > 0) { if (!--cursor.i.y && start_line_adjust) return AddVScroll(start_line_adjust); }
  else {
    cursor.i.y = 0;
    LineOffset *lo = 0;
    if (Wrap()) lo = (--file_line.SecondBound(cursor_start_line_number+1)).val;
    return AddVScroll((lo ? -lo->wrapped_lines : -1) + start_line_adjust);
  }
  UpdateCursorLine();
  UpdateCursor();
}

void Editor::HistDown() {
  bool last=0;
  if (cursor_start_line_number_offset + cursor_glyphs->Lines() < line_fb.lines) {
    last = ++cursor.i.y >= line.Size()-1;
  } else {
    AddVScroll(end_line_cutoff + 1);
    cursor.i.y = line.Size() - 1;
    last = 1; 
  }
  UpdateCursorLine();
  UpdateCursor();
  if (last) UpdateCursorX(cursor.i.x);
}

void Editor::SelectionCB(const Selection::Point &p) {
  cursor.i.y = p.line_ind;
  UpdateCursorLine();
  cursor.i.x = p.char_ind >= 0 ? p.char_ind : CursorGlyphsSize();
  UpdateCursor();
}

void Editor::UpdateMapping(int width, int flag) {
  wrapped_lines = syntax_parsed_anchor = syntax_parsed_line_index = 0;
  last_update_mapping_flag = flag;
  file_line.Clear();
  file->Reset();
  NextRecordReader nr(file.get());
  int ind = 0, offset = 0, wrap = Wrap(), ll;
  for (const char *l = nr.NextLineRaw(&offset); l; l = nr.NextLineRaw(&offset)) {
    wrapped_lines += (ll = wrap ? TextArea::style.font->Lines(l, width, layout.word_break) : 1);
    file_line.val.Insert(LineOffset(offset, nr.record_len, ll, flag ? ind++ : -1));
  }
  file_line.LoadFromSortedVal();
  // XXX update cursor position
}

int Editor::UpdateMappedLines(pair<int, int> read_lines, bool up, bool head_read, bool tail_read,
                              bool short_read, bool shorten_read) {
  bool wrap = Wrap();
  int read_len = 0;
  IOVector rv;
  LineMap::Iterator lib, lie;
  if (read_lines.second) {
    CHECK((lib = file_line.SecondBound(read_lines.first+1)).val);
    if (wrap) {
      if (head_read) start_line_adjust = min(0, lib.key - last_first_line);
      if (short_read && tail_read && end_line_cutoff) ++lib;
    }
    int last_read_line = read_lines.first + read_lines.second - 1;
    for (lie = lib; lie.val && lie.key <= last_read_line; ++lie) {
      auto v = lie.val;
      if (shorten_read && !(lie.key + v->wrapped_lines <= last_read_line)) break;
      if (v->file_size >= 0) read_len += rv.Append({v->file_offset, v->file_size+1});
    }
    if (wrap && tail_read) {
      LineMap::Iterator i = lie.ind ? lie : file_line.RBegin();
      end_line_cutoff = max(0, (--i).key + i.val->wrapped_lines - last_first_line - last_fb_lines);
    }
  }

  string buf(read_len, 0);
  if (read_len) CHECK_EQ(buf.size(), file->ReadIOV(&buf[0], rv.data(), rv.size()));

  int added = 0, bo = 0, width = wrap ? GetFrameBuffer()->w : 0, ll, l, e;
  bool first_line = true;
  Line *L = 0;
  if (up) {
    vector<tuple<String16, const String16*, const DrawableAnnotation*, LineMap::Iterator>> line_data;
    for (auto li = lie; li != lib; bo += l + !e) {
      l = (e = max(0, -(--li).val->file_size)) ? 0 : li.val->file_size;
      if (e) line_data.emplace_back(String16(), &edits[e-1], nullptr, li);
      else   line_data.emplace_back(String::ToUTF16(buf.data()+read_len-bo-l-1, l), nullptr, nullptr, li);
    }
    for (auto ldi = line_data.rbegin(), lde = line_data.rend(); ldi != lde; ++ldi) {
      auto t = tuple_get<1>(*ldi);
      tuple_get<2>(*ldi) = annotation_cb(tuple_get<3>(*ldi), t ? *t : tuple_get<0>(*ldi), first_line, 0, 0);
      first_line = 0;
    }
    added = line_data.size();
    for (auto &ld : line_data) {
      auto t = tuple_get<1>(ld);
      int removed = (wrap && line.ring.Full()) ? line.Front()->Lines() : 0;
      (L = line.PushBack())->AssignText(t ? *t : tuple_get<0>(ld), *tuple_get<2>(ld));
      fb_wrapped_lines += (ll = L->Layout(width, true)) - removed;
      if (removed) line.ring.DecrementSize(1);
      CHECK_EQ(tuple_get<3>(ld).val->wrapped_lines, ll);
    }
  } else {
    for (auto li = lib; li != lie; ++li, bo += l + !e, added++, first_line = 0) {
      l = (e = max(0, -li.val->file_size)) ? 0 : li.val->file_size;
      int removed = (wrap && line.ring.Full()) ? line.Back()->Lines() : 0;
      if (e) { const auto &t = edits[e-1];                 (L = line.PushFront())->AssignText(t, *annotation_cb(li, t, first_line, 0, 0)); }
      else   { auto t = String::ToUTF16(buf.data()+bo, l); (L = line.PushFront())->AssignText(t, *annotation_cb(li, t, first_line, 0, 0)); }
      fb_wrapped_lines += (ll = L->Layout(width, true)) - removed;
      if (removed) line.ring.DecrementSize(1);
      CHECK_EQ(li.val->wrapped_lines, ll);
    }
  }
  return added;
}

int Editor::UpdateLines(float vs, int *first_ind, int *first_offset, int *first_len) {
  if (!opened) return 0;
  int ret = TextView::UpdateLines(vs, first_ind, first_offset, first_len);
  if (!file_line.size()) {
    line.Resize((wrapped_lines = 1));
    line.PushBack()->Clear();
    file_line.Insert(1, LineOffset());
  }
  UpdateCursorLine();
  UpdateCursor();
  return ret;
}

void Editor::UpdateCursorLine() {
  int capped_y = min(cursor.i.y, min(line.Size(), wrapped_lines - last_first_line) - 1);
  cursor_start_line_number = last_first_line + start_line_adjust;
  if (!Wrap()) cursor_start_line_number += (cursor.i.y = capped_y);
  else for (cursor.i.y = 0; cursor.i.y < capped_y && cursor_start_line_number < wrapped_lines-1; cursor.i.y++)
    cursor_start_line_number += line[-1-cursor.i.y].Lines();

  cursor_glyphs = &line[-1-cursor.i.y];
  cursor.i.x = min(cursor.i.x, CursorGlyphsSize());
  cursor_start_line_number_offset = cursor_start_line_number - last_first_line;
  if (!Wrap()) { CHECK_EQ(cursor.i.y, cursor_start_line_number_offset); }

  auto it = file_line.SecondBound(cursor_start_line_number+1);
  if (!it.ind) FATAL("missing line number: ", cursor_start_line_number, " size=", file_line.size(), ", wl=", wrapped_lines); 
  cursor_line_index = it.GetIndex();
  cursor_anchor = it.GetAnchor();
  cursor_offset = it.val;
}

void Editor::UpdateCursor() {
  cursor.p = !cursor_glyphs ? point() : cursor_glyphs->data->glyphs.Position(cursor.i.x) +
    point(0, box.h - cursor_start_line_number_offset * style.font->Height());
}

void Editor::UpdateCursorX(int x) {
  cursor.i.x = min(x, CursorGlyphsSize());
  if (!Wrap()) return UpdateCursor();
  int wl = cursor_start_line_number_offset + cursor_glyphs->data->glyphs.GetLineFromIndex(x);
  if (wl >= line_fb.lines) { AddVScroll(wl-line_fb.lines+1); cursor.i=point(x, line.Size()-1); }
  else if (wl < 0)         { AddVScroll(wl);                 cursor.i=point(x, 0); }
  UpdateCursorLine();
  UpdateCursor();
}

int Editor::CursorLinesChanged(const String16 &b, int add_lines) {
  cursor_glyphs->AssignText(b);
  int ll = cursor_glyphs->Layout(GetFrameBuffer()->w, true);
  int d = ChangedDiff(&cursor_offset->wrapped_lines, ll);
  if (d) { CHECK_EQ(cursor_offset, file_line.Update(cursor_start_line_number, *cursor_offset)); }
  if (int a = d + add_lines) wrapped_lines = AddWrappedLines(wrapped_lines, a);
  return d;
}

int Editor::ModifyCursorLine() {
  CHECK(cursor_offset);
  if (cursor_offset->file_size >= 0) cursor_offset->file_size = -edits.Insert(cursor_glyphs->Text16())-1;
  return -cursor_offset->file_size-1;
}

void Editor::Modify(char16_t c, bool erase, bool undo_or_redo) {
  if (!cursor_glyphs || !cursor_offset) return;
  int edit_id = ModifyCursorLine();
  String16 *b = &edits[edit_id];
  CHECK_LE(cursor.i.x, cursor_glyphs->Size());
  CHECK_LE(cursor.i.x, b->size());
  CHECK_EQ(cursor_offset->wrapped_lines, cursor_glyphs->Lines());
  bool wrap = Wrap(), erase_line = erase && !cursor.i.x;
  if (!cursor_start_line_number && erase_line) return;

  if (!undo_or_redo && !erase_line)
    RecordModify(point(cursor.i.x, cursor_line_index), erase, !erase ? c : (*b)[cursor.i.x-1]);
  modified = Now();
  if (modified_cb) modified_cb();
  if (!erase_line && cursor_line_index < syntax_parsed_line_index) MarkCursorLineFirstDirty();
  cursor_offset->colored = false;

  if (erase_line) {
    file_line.Erase(cursor_start_line_number);
    wrapped_lines = AddWrappedLines(wrapped_lines, -cursor_glyphs->Lines());
    HistUp();
    if (cursor_line_index < syntax_parsed_line_index) MarkCursorLineFirstDirty();
    b = &edits[ModifyCursorLine()];
    bool chomped = !undo_or_redo && b->size() && b->back() == '\r';
    if (chomped) b->pop_back();
    int x = b->size();
    b->append(edits[edit_id]);
    edits.Erase(edit_id);
    if (wrap) CursorLinesChanged(*b);
    UpdateCursorX(x);
    RefreshLines();
    if (!undo_or_redo) {
      if (1)       RecordModify(point(cursor.i.x + chomped, cursor_line_index), true, '\n');
      if (chomped) RecordModify(point(cursor.i.x + 1,       cursor_line_index), true, '\r');
    }
    Redraw(true);
  } else if (!erase && c == '\n') {
    String16 a = b->substr(cursor.i.x);
    b->resize(cursor.i.x);
    if (wrap) CursorLinesChanged(*b, 1);
    else wrapped_lines = AddWrappedLines(wrapped_lines, 1);
    file_line.Insert(cursor_start_line_number + cursor_offset->wrapped_lines, LineOffset());
    if (wrap) RefreshLines();
    cursor.i.x = 0;
    HistDown();
    CHECK_EQ(0, cursor_offset->file_offset);
    CHECK_EQ(0, cursor_offset->file_size);
    CHECK_EQ(1, cursor_offset->wrapped_lines);
    if (wrap) CursorLinesChanged(a);
    swap(edits[ModifyCursorLine()], a);
    RefreshLines();
    if (newline_cb) newline_cb();
    Redraw(true);
  } else {
    if (erase) b->erase(cursor.i.x-1, 1);
    else       b->insert(cursor.i.x, 1, c);
    const DrawableAnnotation *annotation = nullptr;
    if (!wrap && !(annotation = annotation_cb
                   (file_line.GetAnchorIter(cursor_anchor), *b, 0, erase ? -1 : 1, cursor.i.x))) {
      int attr_id = cursor.i.x ? cursor_glyphs->data->glyphs[cursor.i.x-1].attr_id : default_attr;
      if (erase)  cursor_glyphs->Erase          (cursor.i.x-1, 1);
      else if (0) cursor_glyphs->OverwriteTextAt(cursor.i.x, String16(1, c), attr_id);
      else        cursor_glyphs->InsertTextAt   (cursor.i.x, String16(1, c), attr_id);
      UpdateLineFB(cursor_glyphs, GetFrameBuffer(), 0);
    } 
    if (erase) CursorLeft();
    if (wrap) {
      CursorLinesChanged(*b);
      RefreshLines();
      Redraw(true);
    } else if (annotation) {
      // XXX keep reparsing till reach same signature or last displayed line, then only redraw those
      RefreshLines();
      Redraw(true);
    }
    if (!erase) CursorRight();
  }
}

int Editor::Save() {
  IOVector rv;
  for (LineMap::ConstIterator i = file_line.Begin(); i.ind; ++i)
    rv.Append({ i.val->file_offset, i.val->file_size + (i.val->file_size >= 0) });
  int ret = file->Rewrite(ArrayPiece<IOVec>(rv.data(), rv.size()),
    function<string(int)>([&](int i){ return String::ToUTF8(edits.data[i]) + "\n"; }));
  Reload();
  Redraw(true);
  saved_version_number = version_number;
  return ret;
}

int Editor::SaveTo(File *out) {
  IOVector rv;
  for (LineMap::ConstIterator i = file_line.Begin(); i.ind; ++i)
    rv.Append({ i.val->file_offset, i.val->file_size + (i.val->file_size >= 0) });
  return file->Rewrite(ArrayPiece<IOVec>(rv.data(), rv.size()),
    function<string(int)>([&](int i){ return String::ToUTF8(edits.data[i]) + "\n"; }), out);
}

bool Editor::CacheModifiedText(bool force) {
  if (!force && version_number == saved_version_number) return false;
  if (version_number != cached_text_version_number) {
    cached_text = make_shared<BufferFile>(string());
    SaveTo(cached_text.get());
    cached_text_version_number = version_number;
  }
  return true;
}

void Editor::RecordModify(const point &p, bool erase, char16_t c) {
  if (version_number.offset < version.size()) { version_number.major++; version.resize(version_number.offset); }
  if (version.size() && c != '\n') {
    Modification &m = version.back();
    if (m.p.y == p.y && m.erase == erase && !(m.data.size() == 1 && m.data[0] == '\n') &&
        m.p.x + m.data.size() * (erase ? -1 : 1) == p.x) { m.data.append(1, c); return; }
  }
  version.push_back({ p, erase, String16(1, c) });
  version_number.offset = version.size();
}

bool Editor::WalkUndo(bool backwards) {
  if (backwards) { if (!version_number.offset)                  return false; }
  else           { if (version_number.offset >= version.size()) return false; }

  const Modification &m = version[backwards ? --version_number.offset : version_number.offset++];
  bool nl = m.data.size() == 1 && m.data[0] == '\n';
  point target = (nl && (backwards != m.erase)) ? point(0, m.p.y + 1) :
    point(m.p.x + ((backwards && !nl) ? (m.erase ? -m.data.size() : m.data.size()) : 0), m.p.y);
  CHECK(ScrollTo(target.y, target.x));
  CHECK_EQ(target.y, cursor_line_index);
  CHECK_EQ(target.x, cursor.i.x);

  if (backwards) for (auto i=m.data.rbegin(), e=m.data.rend(); i!=e; ++i) Modify(*i, !m.erase, true);
  else           for (auto i=m.data.begin(),  e=m.data.end();  i!=e; ++i) Modify(*i,  m.erase, true);
  return true;
}

bool Editor::ScrollTo(int line_index, int x) {
  int target_offset = min(line_index, last_fb_lines/2), wrapped_line_no = -1;
  if (!Wrap()) {
    if (line_index < 0 || line_index >= file_line.size()) return false;
    wrapped_line_no = line_index;
  } else {
    LineMap::ConstIterator li = file_line.FindIndex(line_index);
    if (!li.val) return false;
    wrapped_line_no = li.GetBegKey();
  }

  cursor.i.y = 0;
  if (wrapped_line_no >= last_first_line &&
      wrapped_line_no < last_first_line + last_fb_lines) target_offset = wrapped_line_no - last_first_line;
  else SetVScroll(wrapped_line_no - target_offset);

  if (!Wrap()) cursor.i.y = target_offset;
  else for (int i=start_line, o=start_line_adjust; o<target_offset; i++, cursor.i.y++) o += line[-1-i].Lines();

  UpdateCursorLine();
  UpdateCursorX(x);
  return true;
}
 
}; // namespace LFL
