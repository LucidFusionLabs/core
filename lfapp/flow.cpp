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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "../crawler/html.h"
#include "../crawler/document.h"

namespace LFL {
float FloatContainer::baseleft(float py, float ph, int *adjacent_out) const {
    int max_left = x;
    basedir(py, ph, &float_left, adjacent_out, [&](const Box &b){ return Max(&max_left, b.right()); });
    return max_left - x;
}

float FloatContainer::baseright(float py, float ph, int *adjacent_out) const { 
    int min_right = x + w;
    basedir(py, ph, &float_right, adjacent_out, [&](const Box &b){ return Min(&min_right, b.x); });
    return min_right - x;
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
            if (!cur_word.len) return state = State::NEW_WORD;
            wrap = cur_word.len && cur_line.end && (p.x + cur_word.len > cur_line.end);
        }
        if (wrap && !(cur_line.fresh && adj_float_left == -1 && adj_float_right == -1)) {
            if (cur_line.fresh) { /* clear floats */ } 
            AppendNewline(h);
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
        int fw = advance / 2;
        Glyph *nbsp = cur_attr.font->FindGlyph(Unicode::non_breaking_space);
        out->data.emplace_back(Box(p.x - fw, p.y + cur_line.descent, fw, nbsp->tex.height), nbsp, box->attr_id, box->line_id);
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
        flow->AppendBox(cj->width, 0, (Box*)0);
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
    flow->AppendBox(1, max_cell_height, (Box*)0);
    flow->AppendNewline();
    int ret = max_cell_height;
    col_skipped = cell_width = max_cell_height = split_cell_height = 0;
    return ret;
}

}; // namespace LFL
