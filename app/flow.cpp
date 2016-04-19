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

#include "core/app/gui.h"

namespace LFL {
point DrawableBoxRun::Draw(point p, DrawCB cb) const {
  Box w;
  DrawBackground(p);
  if (attr->tex) attr->tex->Bind();
  if (attr->tex || attr->font) screen->gd-> SetColor(attr->fg ? *attr->fg : Color::white);
  else                         screen->gd->FillColor(attr->fg ? *attr->fg : Color::white);
  if (attr->font) attr->font->Select();
  else if (attr->tex) screen->gd->EnableLayering();
  if (attr->blend) screen->gd->EnableBlend();
  // else if (!attr->font) screen->gd->DisableBlend();
  if (attr->scissor) screen->gd->PushScissor(*attr->scissor + p);
  for (auto i = data.buf, e = data.end(); i != e; ++i) if (i->drawable) cb(i->drawable, (w = i->box + p), attr);
  if (attr->scissor) screen->gd->PopScissor();
  return point(w.x + w.w, w.y);
}

void DrawableBoxRun::DrawBackground(point p, DrawBackgroundCB cb) const {
  if (attr->bg) screen->gd->FillColor(*attr->bg);
  if (!attr->bg || !data.size()) return;
  int line_height = line ? line->h : (attr->font ? attr->font->Height() : 0);
  if (!line_height) return;
  int left = data[0].LeftBound(attr), right = data.back().RightBound(attr);
  cb(Box(p.x + left, p.y - line_height, right - left, line_height));
}

point DrawableBoxArray::Position(int o) const {
  if (!Size()) return point();
  CHECK_GE(o, 0);
  bool last = o >= Size();
  const DrawableBox &b = last ? data.back() : data[o];
  const Drawable::Attr *a = attr.GetAttr(b.attr_id);
  return point(last ? b.RightBound(a) : b.LeftBound(a), b.TopBound(a));
}

int DrawableBoxArray::BoundingWidth(const DrawableBox &b, const DrawableBox &e) const {
  CHECK_LE(b.box.x, e.box.x);
  return e.RightBound(attr.GetAttr(e.attr_id)) - b.LeftBound(attr.GetAttr(b.attr_id));
}

DrawableBox& DrawableBoxArray::PushBack(const Box &box, int attr_id, Drawable *drawable, int *ind_out) {
  if (ind_out) *ind_out = data.size();
  return LFL::PushBack(data, DrawableBox(box, drawable, attr_id, line.size()));
}

void DrawableBoxArray::InsertAt(int o, const DrawableBoxArray &x) {
  if (!Size() && !o) *this = x;
  else InsertAt(o, x.data);
}

void DrawableBoxArray::InsertAt(int o, const vector<DrawableBox> &x) {
  CHECK_EQ(0, line_ind.size());
  point p(x.size() ? BoundingWidth(x.front(), x.back()) : 0, 0);
  data.insert(data.begin() + o, x.begin(), x.end());
  for (auto i = data.begin() + o + x.size(); i != data.end(); ++i) i->box += p;
}

void DrawableBoxArray::OverwriteAt(int o, const vector<DrawableBox> &x) {
  if (!x.size()) return;
  CHECK_LE(o + x.size(), data.size());
  auto i = data.begin() + o, e = i + x.size();
  point p(BoundingWidth(x.front(), x.back()) - (data.size() ? BoundingWidth(*i, *(e-1)) : 0), 0);
  for (auto xi = x.begin(); i != e; ++i, ++xi) *i = *xi;
  if (p.x) for (i = e, e = data.end(); i != e; ++i) i->box += p;
}

void DrawableBoxArray::Erase(int o, size_t l, bool shift) { 
  if (!l || data.size() <= o) return;
  if (shift) CHECK_EQ(0, line_ind.size());
  vector<DrawableBox>::iterator b = data.begin() + o, e = data.begin() + min(o+l, data.size());
  point p(shift ? BoundingWidth(*b, *(e-1)) : 0, 0);
  auto i = data.erase(b, e);
  if (shift) for (; i != data.end(); ++i) i->box -= p;
}

point DrawableBoxArray::Draw(point p, int glyph_start, int glyph_len) const {
  point e;
  if (!data.size()) return e;
  for (DrawableBoxIterator iter(&data[glyph_start], Xge0_or_Y(glyph_len, data.size())); !iter.Done(); iter.Increment())
    e = DrawableBoxRun(iter.Data(), iter.Length(), attr.GetAttr(iter.cur_attr1), VectorGet(line, iter.cur_attr2)).Draw(p);
  return e;
}

string DrawableBoxArray::DebugString() const {
  string ret = StrCat("BoxArray ", Void(this), " H=", height, " line_ind ", line_ind.size(), " { ");
  for (auto i : line_ind) StrAppend(&ret, i,  ", ");
  StrAppend(&ret, " } size = ", data.size(), ", runs = [ ");

  for (DrawableBoxIterator iter(data); !iter.Done(); iter.Increment()) 
    StrAppend(&ret, "R", iter.i, "(", DrawableBoxRun(iter.Data(), iter.Length()).DebugString(), "), ");
  StrAppend(&ret, "], lines = [\n");

  for (int i=0, start=0; i<=line_ind.size() && start<data.size(); i++) {
    int end = i<line_ind.size() ? line_ind[i] : data.size();
    StrAppend(&ret, "\"", Text(start, end-start), "\",\n");
    start = end;
  }

  return ret;
}

bool DrawableBoxArray::GetGlyphFromCoords(const point &p, int *index_out, Box *box_out, int li) {
  vector<DrawableBox>::const_iterator gb, ge, it;
  gb = data.begin() + ((li && line_ind.size() && li <= line_ind.size()) ? line_ind[li-1] : 0);
  ge = li < line_ind.size() ? (data.begin() + line_ind[li]) : data.end();
  it = LesserBound(gb, ge, DrawableBox(Box(p,0,0)), true);
  if (it == data.end()) { *index_out = data.size(); *box_out = BackOrDefault(data).box; return false; }
  else                  { *index_out = it - data.begin(); *box_out = it->box; return true; }
}

string FloatContainer::DebugString() const {
  string ret = StrCat(Box::DebugString(), " fl{");
  for (int i=0; i<float_left.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_left[i].DebugString());
  StrAppend(&ret, "} fr{");
  for (int i=0; i<float_right.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_right[i].DebugString());
  return ret + "}";
}

float FloatContainer::baseleft(float py, float ph, int *adjacent_out) const {
  int max_left = x;
  basedir(py, ph, &float_left, adjacent_out, [&](const Box &b){ return Max(&max_left, b.right()); });
  return max_left;
}

float FloatContainer::baseright(float py, float ph, int *adjacent_out) const { 
  int min_right = x + w;
  basedir(py, ph, &float_right, adjacent_out, [&](const Box &b){ return Min(&min_right, b.x); });
  return min_right;
}

void FloatContainer::basedir(float py, float ph, const vector<Float> *float_target, int *adjacent_out, function<bool (const Box&)> filter_cb) const {
  if (adjacent_out) *adjacent_out = -1;
  for (int i = 0; i < float_target->size(); i++) {
    const Float &f = (*float_target)[i];
    if ((f.y + 0  ) >= (py + ph)) continue;
    if ((f.y + f.h) <= (py + 0 )) break;
    if (filter_cb(f) && adjacent_out) *adjacent_out = i; 
  }
}

int FloatContainer::FloatHeight() const {
  int min_y = 0;
  for (auto i = float_left .begin(); i != float_left .end(); ++i) if (!i->inherited) Min(&min_y, i->y);
  for (auto i = float_right.begin(); i != float_right.end(); ++i) if (!i->inherited) Min(&min_y, i->y);
  return -min_y;
}

int FloatContainer::ClearFloats(int fy, int fh, bool clear_left, bool clear_right) const {
  if (!clear_left && !clear_right) return 0;
  int fl = -1, fr = -1, sy = fy, ch;
  while (clear_left || clear_right) {
    if (clear_left)  { baseleft (fy, fh, &fl); if (fl >= 0) Min(&fy, float_left [fl].Position().y - fh); }
    if (clear_right) { baseright(fy, fh, &fr); if (fr >= 0) Min(&fy, float_right[fr].Position().y - fh); }
    if ((!clear_left || fl<0) && (!clear_right || fr<0)) break;
  }
  return max(0, sy - fy);
}

void FloatContainer::AddFloat(int fy, int fw, int fh, bool right_or_left, LFL::DOM::Node *v, Box *out_box) {
  for (;;) {
    int adjacent_ind = -1, opposite_ind = -1;
    int base_left  = baseleft (fy, fh, !right_or_left ? &adjacent_ind : &opposite_ind);
    int base_right = baseright(fy, fh,  right_or_left ? &adjacent_ind : &opposite_ind);
    int fx = right_or_left ? (base_right - fw) : base_left;
    Float *adjacent_float = (adjacent_ind < 0) ? 0 : &(!right_or_left ? float_left : float_right)[adjacent_ind];
    Float *opposite_float = (opposite_ind < 0) ? 0 : &( right_or_left ? float_left : float_right)[opposite_ind];
    if (((adjacent_float || opposite_float) && (fx < base_left || (fx + fw) > base_right)) ||
        (adjacent_float && adjacent_float->stacked)) {
      if (adjacent_float) adjacent_float->stacked = 1;
      point afp((X_or_Y(adjacent_float, opposite_float)->Position()));
      fy = afp.y - fh;
      continue;
    }
    *out_box = Box(fx, fy, fw, fh);
    break;
  }
  vector<Float> *float_target = right_or_left ? &float_right : &float_left;
  float_target->push_back(Float(out_box->Position(), out_box->w, out_box->h, v));
  sort(float_target->begin(), float_target->end(), FloatContainer::Compare);
}

int FloatContainer::InheritFloats(const FloatContainer *parent) {
  Copy(parent->float_left,  &float_left,  -TopLeft(), 1, 1);
  Copy(parent->float_right, &float_right, -TopLeft(), 1, 1);
  return parent->float_left.size() + parent->float_right.size();
}

int FloatContainer::AddFloatsToParent(FloatContainer *parent) {
  int count = 0;
  count += Copy(float_left,  &parent->float_left,  TopLeft(), 0, 0);
  count += Copy(float_right, &parent->float_right, TopLeft(), 0, 0);
  Float::MarkInherited(&float_left);
  Float::MarkInherited(&float_right);
  return count;
}

int FloatContainer::Copy(const vector<Float> &s, vector<Float> *d, const point &dc, bool copy_inherited, bool mark_inherited) {
  int count = 0;
  if (!s.size()) return count;
  for (int i=0; i<s.size(); i++) {
    if (!copy_inherited && s[i].inherited) continue;
    d->push_back(Float(s[i], s[i].Position() + dc));
    if (mark_inherited) (*d)[d->size()-1].inherited = 1;
    count++;
  }
  sort(d->begin(), d->end(), Compare);
  return count;
}

void Flow::SetFont(Font *F) {
  if (!(cur_attr.font = F)) return;
  int prev_height = cur_line.height, prev_ascent = cur_line.ascent, prev_descent = cur_line.descent;
  Max(&cur_line.height,  F->Height());
  Max(&cur_line.ascent,  F->ascender);
  Max(&cur_line.descent, F->descender);
  UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, cur_line.descent-prev_descent);
}

void Flow::SetMinimumAscent(short line_ascent) {
  int prev_height = cur_line.height, prev_ascent = cur_line.ascent;
  Max(&cur_line.ascent, line_ascent);
  Max(&cur_line.height, int16_t(cur_line.ascent + cur_line.descent));
  UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, 0);
}

void Flow::UpdateCurrentLine(int height_delta, int ascent_delta, int descent_delta) {
  p.y -= height_delta;
  if (out && !layout.append_only) MoveCurrentLine(point(0, -ascent_delta));
}

void Flow::AppendVerticalSpace(int h) {
  if (h <= 0) return;
  if (!cur_line.fresh) AppendNewline();
  p.y -= h;
  SetCurrentLineBounds();
}

void Flow::AppendBlock(int w, int h, Box *box_out) {
  Max(&max_line_width, w);
  AppendVerticalSpace(h);
  *box_out = Box(0, p.y + cur_line.height, w, h);
}

void Flow::AppendBlock(int w, int h, const Border &b, Box *box_out) {
  AppendBlock(w + b.Width(), h + (h ? b.Height() : 0), box_out);
  *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
}

void Flow::AppendBoxArrayText(const DrawableBoxArray &in) {
  bool attr_fwd = in.attr.source;
  TextAnnotation annotation(&in.attr);
  for (DrawableBoxRawIterator iter(in.data); !iter.Done(); iter.Increment())
    annotation.emplace_back(iter.cur_start, iter.cur_attr);
  AppendText(DrawableBoxRun(&in[0], in.Size()).Text16(), annotation);
}

int Flow::AppendBox(int w, int h, Drawable *drawable) { 
  AppendBox(&out->PushBack(Box(0,0,w,h), cur_attr, drawable));
  return out->data.size()-1;
}

void Flow::AppendBox(int w, int h, Box *box_out) {
  static bool add_non_drawable = true;
  if (out && add_non_drawable) {
    AppendBox(&out->PushBack(Box(0,0,w,h), cur_attr, NULL));
    if (box_out) *box_out = out->data.back().box;
  } else {
    DrawableBox box(Box(0,0,w,h), 0, out ? out->attr.GetAttrId(cur_attr) : 0, out ? out->line.size() : -1);
    AppendBox(&box);
    if (box_out) *box_out = box.box;
  }
}

void Flow::AppendBox(int w, int h, const Border &b, Box *box_out) {
  AppendBox(w + b.Width(), h + (h ? b.Height() : 0), box_out);
  if (box_out) *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
}

void Flow::AppendBox(DrawableBox *box) {
  point bp = box->box.Position();
  SetMinimumAscent(box->box.h);
  if (!box->box.w) box->box.SetPosition(p);
  else {
    box->box.SetPosition(bp);
    cur_word.len = box->box.w;
    cur_word.fresh = 1;
    CHECK_EQ(State::OK, AppendBoxOrChar(0, box, box->box.h));
  }
  cur_word.len = 0;
}

Flow::State Flow::AppendChar(int c, int attr_id, DrawableBox *box) {
  if (layout.char_tf) c = layout.char_tf(c);
  if (state == State::NEW_WORD && layout.word_start_char_tf) c = layout.word_start_char_tf(c);
  Max(&cur_line.height, cur_attr.font->Height());
  box->drawable = cur_attr.font->FindGlyph(c);
  box->attr_id = attr_id;
  box->line_id = out ? out->line.size() : -1;
  return AppendBoxOrChar(c, box, cur_attr.font->Height());
}

Flow::State Flow::AppendBoxOrChar(int c, DrawableBox *box, int h) {
  bool space = isspace(c), drawable = box->drawable;
  if (space) cur_word.len = 0;
  int max_line_shifts = 1000;
  for (; layout.wrap_lines && max_line_shifts; max_line_shifts--) {
    bool wrap = 0;
    if (!cur_word.len) cur_word.fresh = 1;
    if (!layout.word_break) {
      int box_width = drawable ? box->drawable->Advance(&box->box, &cur_attr) : box->box.w;
      wrap = cur_line.end && p.x + box_width > cur_line.end;
    } else if (cur_word.fresh && !space) {
      if (!cur_word.len) return (state = State::NEW_WORD);
      wrap = cur_word.len && cur_line.end && (p.x + cur_word.len > cur_line.end);
    }
    if (wrap && !(cur_line.fresh && adj_float_left == -1 && adj_float_right == -1)) {
      if (cur_line.fresh) { /* clear floats */ } 
      AppendNewline(h, true);
      continue;
    }
    break;
  }
  CHECK(max_line_shifts);
  cur_line.fresh = 0;
  cur_word.fresh = 0;
  if (c == '\n') { if (!layout.ignore_newlines) AppendNewline(); return State::OK; }

  int advance = drawable ? box->drawable->Layout(&box->box, &cur_attr) : box->box.w;
  box->box.y += cur_line.descent;
  box->box += p;
  p.x += advance;
  state = State::OK;

  if (layout.pad_wide_chars && drawable && box->drawable->Wide()) {
    Glyph *nbsp = cur_attr.font->FindGlyph(Unicode::zero_width_non_breaking_space);
    out->data.emplace_back(Box(p.x, p.y + cur_line.descent, 0, nbsp->tex.height), nbsp, box->attr_id, box->line_id);
  }

  return state;
}

Flow::State Flow::AppendNewline(int need_height, bool next_glyph_preadded) {
  if (out) {        
    AlignCurrentLine();
    out->line.push_back(CurrentLineBox());
    out->line_ind.push_back(out ? max<int>(0, out->data.size()-next_glyph_preadded) : 0);
    out->height += out->line.back().h;
    if (out->data.size() > cur_line.out_ind)
      Max(&max_line_width, out->data.back().box.right() - out->data[cur_line.out_ind].box.x);
  }
  cur_line.fresh = 1;
  cur_line.height = cur_line.ascent = cur_line.descent = 0;
  cur_line.out_ind = out ? out->data.size() : 0;
  SetMinimumAscent(max(need_height, LayoutLineHeight()));
  SetCurrentLineBounds();
  if (cur_attr.font) cur_line.descent = cur_attr.font->descender;
  return state = State::NEW_LINE;
}

void Flow::AlignCurrentLine() {
  if (cur_line.out_ind >= out->data.size() || (!layout.align_center && !layout.align_right)) return;
  int line_size = cur_line.end - cur_line.beg, line_min_x, line_max_x;
  GetCurrentLineExtents(&line_min_x, &line_max_x);
  int line_len = line_max_x - line_min_x, align = 0;
  if      (layout.align_center) align = (line_size - line_len)/2;
  else if (layout.align_right)  align = (line_size - line_len);
  if (align) MoveCurrentLine(point(align, 0));
}

void Flow::MoveCurrentLine(const point &dx) { 
  for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) i->box += dx;
}

void Flow::GetCurrentLineExtents(int *min_x, int *max_x) { 
  *min_x=INT_MAX; *max_x=INT_MIN;
  for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) { Min(min_x, i->box.x); Max(max_x, i->box.right()); } 
}

void Flow::SetCurrentLineBounds() {
  cur_line.beg = container->baseleft (p.y, cur_line.height, &adj_float_left)  - container->x;
  cur_line.end = container->baseright(p.y, cur_line.height, &adj_float_right) - container->x;
  p.x = cur_line.beg;
}

void TableFlow::SetMinColumnWidth(int j, int width, int colspan) {
  EnsureSize(column, j+colspan);
  if (width) for (int v=width/colspan, k=j; k<j+colspan; k++) Max(&column[k].width, v);
}

TableFlow::Column *TableFlow::SetCellDim(int j, int width, int colspan, int rowspan) {
  while (VectorEnsureElement(column, j+col_skipped)->remaining_rowspan) col_skipped++;
  SetMinColumnWidth(j+col_skipped, width, colspan);
  for (int k = 0; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan += rowspan;
  return &column[j+col_skipped];
}

int TableFlow::ComputeWidth(int fixed_width) {
  int table_width = 0, auto_width_cols = 0, sum_column_width = 0;
  TableFlowColIter(this) { cj->ResetHeight(); if (cj->width) sum_column_width += cj->width; else auto_width_cols++; }
  if (fixed_width) {
    table_width = max(fixed_width, sum_column_width);
    int remaining = table_width - sum_column_width;
    if (remaining > 0) {
      if (auto_width_cols) { TableFlowColIter(this) if (!cj->width) cj->width += remaining/auto_width_cols; }
      else                 { TableFlowColIter(this)                 cj->width += remaining/cols; }
    }
  } else { 
    int min_table_width = 0, max_table_width = 0;
    TableFlowColIter(this) { 
      min_table_width += max(cj->width, cj->min_width);
      max_table_width += max(cj->width, cj->max_width);
    }
    bool maxfits = max_table_width < flow->container->w;
    table_width = maxfits ? max_table_width : min_table_width;
    TableFlowColIter(this) {
      cj->width = maxfits ? max(cj->width, cj->max_width) : max(cj->width, cj->min_width);
    }
  }
  return table_width;
}

void TableFlow::AppendCell(int j, Box *out, int colspan) {
  TableFlow::Column *cj = 0;
  for (;;col_skipped++) {
    if (!(cj = VectorCheckElement(column, j+col_skipped))->remaining_rowspan) break;
    flow->AppendBox(cj->width, 0, NullPointer<Box>());
  }
  cell_width = 0;
  CHECK_LE(j+col_skipped+colspan, column.size());
  for (int k=j+col_skipped, l=k+colspan; k<l; k++) cell_width += column[k].width;
  flow->AppendBox(cell_width, 0, out);
}

void TableFlow::SetCellHeight(int j, int cellheight, void *cell, int colspan, int rowspan) {
  column[j+col_skipped].AddHeight(cellheight, rowspan, cell);
  for (int k = 1; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan = rowspan;
  col_skipped += colspan-1;

  if (rowspan == 1)   max_cell_height = max(max_cell_height,   cellheight);
  else              split_cell_height = max(split_cell_height, cellheight / rowspan);
}

int TableFlow::AppendRow() {
  if (!max_cell_height) max_cell_height = split_cell_height;
  TableFlowColIter(this) {
    cj->remaining_rowspan = max(0, cj->remaining_rowspan - 1);
    if (!cj->remaining_rowspan) Max(&max_cell_height, cj->remaining_height);
  }
  TableFlowColIter(this) {
    int subtracted = min(max_cell_height, cj->remaining_height);
    if (subtracted) cj->remaining_height -= subtracted;
  }
  flow->AppendBox(1, max_cell_height, NullPointer<Box>());
  flow->AppendNewline();
  int ret = max_cell_height;
  col_skipped = cell_width = max_cell_height = split_cell_height = 0;
  return ret;
}

}; // namespace LFL
