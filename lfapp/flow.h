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

#ifndef LFL_LFAPP_FLOW_H__
#define LFL_LFAPP_FLOW_H__

#define FlowDebug(...) INFO(__VA_ARGS__)

namespace LFL {

struct DrawableBoxRun {
  const Drawable::Attr *attr;
  ArrayPiece<DrawableBox> data;
  const Box *line;
  DrawableBoxRun(const DrawableBox *buf=0, int len=0)                                        : attr(0), data(buf, len), line(0) {}
  DrawableBoxRun(const DrawableBox *buf,   int len, const Drawable::Attr *A, const Box *L=0) : attr(A), data(buf, len), line(L) {}

  int                Size () const { return data.size(); }
  const DrawableBox &First() const { return data[0]; }
  const DrawableBox &Last () const { return data[Size()-1]; }

  template <class K> basic_string<K> Text(int i, int l) const {
    basic_string<K> t(l, 0);
    auto bi = &data.buf[i];
    for (auto ti = t.begin(), te = t.end(); ti != te; ++ti, ++bi) *ti = bi->Id();
    return t;
  }
  string   Text  ()             const { return Text<char>    (0, data.size()); }
  String16 Text16()             const { return Text<char16_t>(0, data.size()); }
  string   Text  (int i, int l) const { return Text<char>    (i, l); }
  String16 Text16(int i, int l) const { return Text<char16_t>(i, l); }
  string   DebugString()        const { return StrCat("BoxRun='", Text(), "'"); }

  typedef function<void    (const Drawable *,  const Box &,  const Drawable::Attr *)> DrawCB;
  static void DefaultDrawCB(const Drawable *d, const Box &w, const Drawable::Attr *a) { d->Draw(w, a); }
  point Draw(point p, DrawCB = &DefaultDrawCB) const;
  void draw(point p) const { Draw(p); }

  typedef function<void              (const Box &)> DrawBackgroundCB;
  static void DefaultDrawBackgroundCB(const Box &w) { w.Draw(); }
  void DrawBackground(point p, DrawBackgroundCB = &DefaultDrawBackgroundCB) const;
};

struct DrawableBoxArray {
  vector<DrawableBox> data;
  Drawable::AttrVec attr;
  vector<Box> line;
  vector<int> line_ind;
  int height;
  DrawableBoxArray() : height(0) { Clear(); }

  /**/  DrawableBox& operator[](int i)       { return data[i]; }
  const DrawableBox& operator[](int i) const { return data[i]; }
  const DrawableBox& Back() const { return data.back(); }
  int Size() const { return data.size(); }

  string   Text  ()             const { return data.size() ? DrawableBoxRun(&data[0], data.size()).Text  ()     : string();   }
  String16 Text16()             const { return data.size() ? DrawableBoxRun(&data[0], data.size()).Text16()     : String16(); }
  string   Text  (int i, int l) const { return data.size() ? DrawableBoxRun(&data[0], data.size()).Text  (i, l) : string();   }
  String16 Text16(int i, int l) const { return data.size() ? DrawableBoxRun(&data[0], data.size()).Text16(i, l) : String16(); }

  point Position(int o) const;
  int LeftBound (int o) const { const DrawableBox &b = data[o]; return b.LeftBound (attr.GetAttr(b.attr_id)); }
  int RightBound(int o) const { const DrawableBox &b = data[o]; return b.RightBound(attr.GetAttr(b.attr_id)); }
  int BoundingWidth(const DrawableBox &b, const DrawableBox &e) const;

  void Clear() { data.clear(); attr.clear(); line.clear(); line_ind.clear(); height=0; }
  DrawableBoxArray *Reset() { Clear(); return this; }

  DrawableBox &PushBack(const Box &box, const Drawable::Attr &a,       Drawable *drawable, int *ind_out=0) { return PushBack(box, attr.GetAttrId(a), drawable, ind_out); }
  DrawableBox &PushBack(const Box &box, int                   attr_id, Drawable *drawable, int *ind_out=0);

  void InsertAt(int o, const DrawableBoxArray &x);
  void InsertAt(int o, const vector<DrawableBox> &x);
  void OverwriteAt(int o, const vector<DrawableBox> &x);
  void Erase(int o, size_t l=UINT_MAX, bool shift=false);
  point Draw(point p, int glyph_start=0, int glyph_len=-1) const;
  string DebugString() const;

  int GetLineFromCoords(const point &p) { return 0; }
  int GetLineFromIndex(int n) { return upper_bound(line_ind.begin(), line_ind.end(), n) - line_ind.begin(); }
  bool GetGlyphFromCoords(const point &p, int *index_out, Box *box_out) { return GetGlyphFromCoords(p, index_out, box_out, GetLineFromCoords(p)); }
  bool GetGlyphFromCoords(const point &p, int *index_out, Box *box_out, int li);

  struct RollbackState { size_t data_size, attr_size, line_size; int height; };
  RollbackState GetRollbackState() const { return { data.size(), attr.size(), line.size(), height }; }
  void Rollback(const RollbackState &s) { data.resize(s.data_size); attr.resize(s.attr_size); line.resize(s.line_size); height=s.height; }
};

struct FloatContainer : public Box {
  struct Float : public Box {
    bool inherited, stacked; void *val;
    Float() : inherited(0), stacked(0), val(0) {}
    Float(const point &p, int W=0, int H=0, void *V=0) : Box(p, W, H), inherited(0), stacked(0), val(V) {}
    Float(const Box   &w,                   void *V=0) : Box(w),       inherited(0), stacked(0), val(V) {}
    Float(const Float &f, const point &p) : Box(p, f.w, f.h), inherited(f.inherited), stacked(f.stacked), val(f.val) {}
    virtual string DebugString() const { return StrCat("Float{", Box::DebugString(), ", inherited=", inherited, ", stacked=", stacked, ", val=", (void*)val, "}"); }
    static void MarkInherited(vector<Float> *t) { for (auto i = t->begin(); i != t->end(); ++i) i->inherited=1; }
  };
  vector<Float> float_left, float_right;
  FloatContainer() {}
  FloatContainer(const Box &W) : Box(W) {}
  FloatContainer &operator=(const Box &W) { x=W.x; y=W.y; w=W.w; h=W.h; return *this; }

  virtual string DebugString() const;
  virtual const FloatContainer *AsFloatContainer() const { return this; }
  virtual       FloatContainer *AsFloatContainer()       { return this; }
  virtual float baseleft(float py, float ph, int *adjacent_out=0) const;
  virtual float baseright(float py, float ph, int *adjacent_out=0) const;
  void basedir(float py, float ph, const vector<Float> *float_target, int *adjacent_out, function<bool (const Box&)> filter_cb) const;

  int CenterFloatWidth(int fy, int fh) const { return baseright(fy, fh) - baseleft(fy, fh); }
  int FloatHeight() const;
  int ClearFloats(int fy, int fh, bool clear_left, bool clear_right) const;

  FloatContainer *Reset() { Clear(); return this; }
  void Clear() { float_left.clear(); float_right.clear(); }

  void AddFloat(int fy, int fw, int fh, bool right_or_left, LFL::DOM::Node *v, Box *out_box);
  int InheritFloats(const FloatContainer *parent);
  int AddFloatsToParent(FloatContainer *parent);

  static bool Compare(const Box &lw, const Box &rw) { return pair<int,int>(lw.top(), lw.h) > pair<int,int>(rw.top(), rw.h); }
  static int Copy(const vector<Float> &s, vector<Float> *d, const point &dc, bool copy_inherited, bool mark_inherited);
};

struct Flow {
  enum class State { OK=1, NEW_WORD=2, NEW_LINE=3 };
  struct TextAnnotation : public ArrayPiece<pair<int, int>> {
    const Drawable::AttrSource *attr_source=0;
    TextAnnotation() {}
    TextAnnotation(const pair<int,int> *a, const PieceIndex &p)                     : ArrayPiece<pair<int, int>>(a, p) {} 
    TextAnnotation(const vector<pair<int,int>> &a, const Drawable::AttrSource *s=0) : ArrayPiece<pair<int, int>>(a), attr_source(s) {}
  };
  struct Layout {
    bool wrap_lines=1, word_break=1, align_center=0, align_right=0, ignore_newlines=0, pad_wide_chars=0;
    int char_spacing=0, word_spacing=0, line_height=0, valign_offset=0;
    int (*char_tf)(int)=0, (*word_start_char_tf)(int)=0;
  };
  struct CurrentLine { int out_ind=0, beg=0, end=0; short height=0, ascent=0, descent=0, base=0; bool fresh=0; };
  struct CurrentWord { int len=0;                                                                bool fresh=0; };

  Layout layout;
  point p; 
  DrawableBoxArray *out;
  const Box *container;
  Drawable::Attr cur_attr;
  CurrentLine cur_line;
  CurrentWord cur_word;
  int adj_float_left=-1, adj_float_right=-1;
  State state=State::OK;
  int max_line_width=0;

  Flow(DrawableBoxArray *O) : Flow(0, 0, O) {}
  Flow(const Box *W=0, Font *F=0, DrawableBoxArray *O=0, Layout *L=0) :
    layout(*(L?L:Singleton<Layout>::Get())), out(O), container(W?W:Singleton<Box>::Get())
    { cur_line.out_ind=O?O->Size():0; SetFont(F); SetCurrentLineBounds(); cur_line.fresh=1; }

  struct RollbackState {
    point p; Drawable::Attr attr; CurrentLine line; CurrentWord word; State state; int max_line_width;
    DrawableBoxArray::RollbackState out_state; 
  };
  RollbackState GetRollbackState() { return { p, cur_attr, cur_line, cur_word, state, max_line_width, out->GetRollbackState() }; }
  void Rollback(const RollbackState &s) { p=s.p; cur_attr=s.attr; cur_line=s.line; cur_word=s.word; state=s.state; max_line_width=s.max_line_width; out->Rollback(s.out_state); }
  string DebugString() const { return StrCat("Flow{ p=", p.DebugString(), ", container=", container->DebugString(), "}"); }

  void SetFGColor(const Color *C) { cur_attr.fg = C; }
  void SetBGColor(const Color *C) { cur_attr.bg = C; }
  void SetAtlas(Font *F) { cur_attr.font = F; }
  void SetFont(Font *F);
  void SetMinimumAscent(short line_ascent);
  void UpdateCurrentLine(int height_delta, int ascent_delta, int descent_delta);

  int Height() const { return -p.y - (cur_line.fresh ? cur_line.height : 0); }
  Box CurrentLineBox() const { return Box(cur_line.beg, p.y, p.x - cur_line.beg, cur_line.height); }
  int LayoutLineHeight() const { return X_or_Y(layout.line_height, cur_attr.font ? cur_attr.font->Height() : 0); }

  void AppendVerticalSpace(int h);
  void AppendBlock(int w, int h, Box *box_out);
  void AppendBlock(int w, int h, const Border &b, Box *box_out);
  void AppendRow(float x=0, float w=0, Box *box_out=0) { AppendBox(x, container->w*w, cur_line.height, box_out); }
  void AppendBoxArrayText(const DrawableBoxArray &in);
  int AppendBox(float x, int w, int h, Drawable *drawable) { p.x=container->w*x; return AppendBox(w, h, drawable); }
  int AppendBox(/**/     int w, int h, Drawable *drawable);
  void AppendBox(float x, int w, int h, Box *box_out) { p.x=container->w*x; AppendBox(w, h, box_out); }
  void AppendBox(/**/     int w, int h, Box *box_out);
  void AppendBox(int w, int h, const Border &b, Box *box_out);
  void AppendBox(DrawableBox *box);

  /**/               int AppendText(float x, const string          &text) { p.x=container->w*x; return AppendText(StringPiece           (text), 0); }
  /**/               int AppendText(float x, const String16        &text) { p.x=container->w*x; return AppendText(String16Piece         (text), 0); }
  template <class X> int AppendText(float x, const X               *text) { p.x=container->w*x; return AppendText(StringPiece::Unbounded(text), 0); }
  template <class X> int AppendText(float x, const StringPieceT<X> &text) { p.x=container->w*x; return AppendText<X>(                    text,  0); }

  /**/               int AppendText(const string          &text, int attr_id=0) { return AppendText(StringPiece           (text), attr_id); }
  /**/               int AppendText(const String16        &text, int attr_id=0) { return AppendText(String16Piece         (text), attr_id); }
  template <class X> int AppendText(const X               *text, int attr_id=0) { return AppendText(StringPiece::Unbounded(text), attr_id); }
  template <class X> int AppendText(const StringPieceT<X> &text, int attr_id=0) { return AppendText(text, TextAnnotation(), attr_id); }

  /**/               int AppendText(const string          &text, const TextAnnotation &attr, int da=0) { return AppendText(StringPiece           (text), attr, da); }
  /**/               int AppendText(const String16        &text, const TextAnnotation &attr, int da=0) { return AppendText(String16Piece         (text), attr, da); }
  template <class X> int AppendText(const X               *text, const TextAnnotation &attr, int da=0) { return AppendText(StringPiece::Unbounded(text), attr, da); }
  template <class X> int AppendText(const StringPieceT<X> &text, const TextAnnotation &attr, int da=0) {
    int start_size = out->data.size();
    out->data.reserve(start_size + text.size());
    int initial_out_lines = out->line.size(), line_start_ind = 0, c_bytes = 0, ci_bytes = 0, c, ci;
    int attr_id = (!attr.len && !da && !out->attr.source) ? out->attr.GetAttrId(cur_attr) : da;
    auto a = attr.buf, ae = a + attr.len;

    for (const X *b = text.data(), *p = b; !text.Done(p); p += c_bytes) {
      if (a != ae && p - b == a->first) {
        if (out->attr.source) cur_attr.font = out->attr.GetAttr((attr_id = a++->second))->font;
        else attr_id = out->attr.GetAttrId((cur_attr = *attr.attr_source->GetAttr(a++->second)));
      }
      if (!(c = UTF<X>::ReadGlyph(text, p, &c_bytes, true))) FlowDebug("null glyph");
      if (c == Unicode::zero_width_non_breaking_space) continue;
      if (AppendChar(c, attr_id, &PushBack(out->data, DrawableBox())) == State::NEW_WORD) {
        for (const X *pi=p; !text.Done(pi) && notspace(*pi); pi += ci_bytes) {
          if (!(ci = UTF<X>::ReadGlyph(text, pi, &ci_bytes, true))) FlowDebug("null glyph");
          cur_word.len += cur_attr.font->GetGlyphWidth(ci);
        }
        AppendChar(c, attr_id, &out->data.back());
      }
    }
    return out->data.size() - start_size;
  }

  State AppendChar(int c, int attr_id, DrawableBox *box);
  State AppendBoxOrChar(int c, DrawableBox *box, int h);

  void AppendNewlines(int n) { for (int i=0; i<n; i++) AppendNewline(); }
  State AppendNewline(int need_height=0, bool next_glyph_preadded=0);

  void AlignCurrentLine();
  void MoveCurrentLine(const point &dx);
  void GetCurrentLineExtents(int *min_x, int *max_x);
  void SetCurrentLineBounds();

  void Complete() { if (!cur_line.fresh) AppendNewline(); }
};

#define TableFlowColIter(t) for (int j=0, cols=(t)->column.size(); j<cols; j++) if (TableFlow::Column *cj = &(t)->column[j])
struct TableFlow {
  struct Column {
    int width=0, min_width=0, max_width=0, last_ended_y=0, remaining_rowspan=0, remaining_height=0; void *remaining_val=0;
    void ResetHeight() { last_ended_y=remaining_rowspan=remaining_height=0; remaining_val=0; }
    void AddHeight(int height, int rowspan, void *val=0) {
      CHECK(!remaining_height && !remaining_rowspan);
      remaining_height=height; remaining_rowspan=rowspan; remaining_val=val;
    }
  };
  Flow *flow;
  vector<Column> column;
  int col_skipped=0, cell_width=0, max_cell_height=0, split_cell_height=0;
  TableFlow(Flow *F=0) : flow(F) {}

  void Select() { flow->layout.wrap_lines=0; }
  void SetMinColumnWidth(int j, int width, int colspan=1);
  Column *SetCellDim(int j, int width, int colspan=1, int rowspan=1);
  void NextRowDim() { col_skipped=0; TableFlowColIter(this) if (cj->remaining_rowspan) cj->remaining_rowspan--; }
  int ComputeWidth(int fixed_width);
  void AppendCell(int j, Box *out, int colspan=1);
  void SetCellHeight(int j, int cellheight, void *cell, int colspan=1, int rowspan=1);
  int AppendRow();
};

}; // namespace LFL
#endif // LFL_LFAPP_FLOW_H__
