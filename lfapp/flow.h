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

#ifndef __LFL_LFAPP_FLOW_H__
#define __LFL_LFAPP_FLOW_H__

namespace LFL {

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
    virtual float baseleft(float py, float ph, int *adjacent_out=0) const {
        int max_left = x;
        basedir(py, ph, &float_left, adjacent_out, [&](const Box &b){ return Max(&max_left, b.right()); });
        return max_left - x;
    }
    virtual float baseright(float py, float ph, int *adjacent_out=0) const { 
        int min_right = x + w;
        basedir(py, ph, &float_right, adjacent_out, [&](const Box &b){ return Min(&min_right, b.x); });
        return min_right - x;
    }
    void basedir(float py, float ph, const vector<Float> *float_target, int *adjacent_out, function<bool (const Box&)> filter_cb) const {
        if (adjacent_out) *adjacent_out = -1;
        for (int i = 0; i < float_target->size(); i++) {
            const Float &f = (*float_target)[i];
            if ((f.y + 0  ) >= (py + ph)) continue;
            if ((f.y + f.h) <= (py + 0 )) break;
            if (filter_cb(f) && adjacent_out) *adjacent_out = i; 
        }
    }

    int CenterFloatWidth(int fy, int fh) const { return baseright(fy, fh) - baseleft(fy, fh); }
    int FloatHeight() const {
        int min_y = 0;
        for (auto i = float_left .begin(); i != float_left .end(); ++i) if (!i->inherited) Min(&min_y, i->y);
        for (auto i = float_right.begin(); i != float_right.end(); ++i) if (!i->inherited) Min(&min_y, i->y);
        return -min_y;
    }
    int ClearFloats(int fy, int fh, bool clear_left, bool clear_right) const {
        if (!clear_left && !clear_right) return 0;
        int fl = -1, fr = -1, sy = fy, ch;
        while (clear_left || clear_right) {
            if (clear_left)  { baseleft (fy, fh, &fl); if (fl >= 0) Min(&fy, float_left [fl].Position().y - fh); }
            if (clear_right) { baseright(fy, fh, &fr); if (fr >= 0) Min(&fy, float_right[fr].Position().y - fh); }
            if ((!clear_left || fl<0) && (!clear_right || fr<0)) break;
        }
        return max(0, sy - fy);
    }

    FloatContainer *Reset() { Clear(); return this; }
    void Clear() { float_left.clear(); float_right.clear(); }

    void AddFloat(int fy, int fw, int fh, bool right_or_left, LFL::DOM::Node *v, Box *out_box) {
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

    int InheritFloats(const FloatContainer *parent) {
        Copy(parent->float_left,  &float_left,  -TopLeft(), 1, 1);
        Copy(parent->float_right, &float_right, -TopLeft(), 1, 1);
        return parent->float_left.size() + parent->float_right.size();
    }
    int AddFloatsToParent(FloatContainer *parent) {
        int count = 0;
        count += Copy(float_left,  &parent->float_left,  TopLeft(), 0, 0);
        count += Copy(float_right, &parent->float_right, TopLeft(), 0, 0);
        Float::MarkInherited(&float_left);
        Float::MarkInherited(&float_right);
        return count;
    }

    static bool Compare(const Box &lw, const Box &rw) { return pair<int,int>(lw.top(), lw.h) > pair<int,int>(rw.top(), rw.h); }
    static int Copy(const vector<Float> &s, vector<Float> *d, const point &dc, bool copy_inherited, bool mark_inherited) {
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
};

struct Flow {
    struct Layout {
        bool wrap_lines=1, word_break=1, align_center=0, align_right=0, ignore_newlines=0;
        int char_spacing=0, word_spacing=0, line_height=0, valign_offset=0;
        int (*char_tf)(int)=0, (*word_start_char_tf)(int)=0;
    } layout;
    point p; 
    BoxArray *out;
    const Box *container;
    Drawable::Attr cur_attr;
    int adj_float_left=-1, adj_float_right=-1;
    struct CurrentLine { int out_ind, beg, end; short height, ascent, descent, base; bool fresh; } cur_line;
    struct CurrentWord { int out_ind, len;                                           bool fresh; } cur_word;
    enum class State { OK=1, NEW_WORD=2, NEW_LINE=3 } state=State::OK;
    int max_line_width=0;

    Flow(BoxArray *O) : Flow(0, 0, O) {}
    Flow(const Box *W=0, Font *F=0, BoxArray *O=0, Layout *L=0) :
        layout(*(L?L:Singleton<Layout>::Get())), out(O), container(W?W:Singleton<Box>::Get())
        { memzero(cur_line); memzero(cur_word); SetFont(F); SetCurrentLineBounds(); cur_line.fresh=1; }

    struct RollbackState {
        point p; Drawable::Attr attr; CurrentLine line; CurrentWord word; State state; int max_line_width;
        BoxArray::RollbackState out_state; 
    };
    RollbackState GetRollbackState() { return { p, cur_attr, cur_line, cur_word, state, max_line_width, out->GetRollbackState() }; }
    void Rollback(const RollbackState &s) { p=s.p; cur_attr=s.attr; cur_line=s.line; cur_word=s.word; state=s.state; max_line_width=s.max_line_width; out->Rollback(s.out_state); }
    string DebugString() const {
        return StrCat("Flow{ p=", p.DebugString(), ", container=", container->DebugString(), "}");
    }

    void SetFGColor(const Color *C) { cur_attr.fg = C; }
    void SetBGColor(const Color *C) { cur_attr.bg = C; }
    void SetAtlas(Font *F) { cur_attr.font = F; }
    void SetFont(Font *F) {
        if (!(cur_attr.font = F)) return;
        int prev_height = cur_line.height, prev_ascent = cur_line.ascent, prev_descent = cur_line.descent;
        Max(&cur_line.height,  F->Height());
        Max(&cur_line.ascent,  F->ascender);
        Max(&cur_line.descent, F->descender);
        UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, cur_line.descent-prev_descent);
    }
    void SetMinimumAscent(short line_ascent) {
        int prev_height = cur_line.height, prev_ascent = cur_line.ascent;
        Max(&cur_line.ascent, line_ascent);
        Max(&cur_line.height, (short)(cur_line.ascent + cur_line.descent));
        UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, 0);
    }
    void UpdateCurrentLine(int height_delta, int ascent_delta, int descent_delta) {
        p.y -= height_delta;
        if (out) MoveCurrentLine(point(0, -ascent_delta));
    }

    int Height() const { return -p.y - (cur_line.fresh ? cur_line.height : 0); }
    Box CurrentLineBox() const { return Box(cur_line.beg, p.y, p.x - cur_line.beg, cur_line.height); }
    int LayoutLineHeight() const { return X_or_Y(layout.line_height, cur_attr.font ? cur_attr.font->Height() : 0); }

    void AppendVerticalSpace(int h) {
        if (h <= 0) return;
        if (!cur_line.fresh) AppendNewline();
        p.y -= h;
        SetCurrentLineBounds();
    }
    void AppendBlock(int w, int h, Box *box_out) {
        AppendVerticalSpace(h);
        *box_out = Box(0, p.y + cur_line.height, w, h);
    }
    void AppendBlock(int w, int h, const Border &b, Box *box_out) {
        AppendBlock(w + b.Width(), h + (h ? b.Height() : 0), box_out);
        *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
    }
    void AppendRow(float x=0, float w=0, Box *box_out=0) { AppendBox(x, container->w*w, cur_line.height, box_out); }
    void AppendBoxArrayText(const BoxArray &in) {
        bool attr_fwd = in.attr.source;
        for (Drawable::Box::RawIterator iter(in.data); !iter.Done(); iter.Increment()) {
            if (!attr_fwd) cur_attr = *in.attr.GetAttr(iter.cur_attr);
            AppendText(BoxRun(iter.Data(), iter.Length()).Text(), attr_fwd ? iter.cur_attr : 0);
        }
    }

    int AppendBox(float x, int w, int h, Drawable *drawable) { p.x=container->w*x; return AppendBox(w, h, drawable); }
    int AppendBox(/**/     int w, int h, Drawable *drawable) { 
        AppendBox(&out->PushBack(Box(0,0,w,h), cur_attr, drawable));
        return out->data.size()-1;
    }

    void AppendBox(float x, int w, int h, Box *box_out) { p.x=container->w*x; AppendBox(w, h, box_out); }
    void AppendBox(/**/     int w, int h, Box *box_out) {
        Drawable::Box box(Box(0,0,w,h), 0, out ? out->attr.GetAttrId(cur_attr) : 0, out ? out->line.size() : -1);
        AppendBox(&box);
        if (box_out) *box_out = box.box;
    }
    void AppendBox(int w, int h, const Border &b, Box *box_out) {
        AppendBox(w + b.Width(), h + (h ? b.Height() : 0), box_out);
        if (box_out) *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
    }

    void AppendBox(Drawable::Box *box) {
        point bp = box->box.Position();
        SetMinimumAscent(box->box.h);
        if (!box->box.w) box->box.SetPosition(p);
        else {
            box->box.SetPosition(bp);
            cur_word.len = box->box.w;
            cur_word.fresh = 1;
            AppendBoxOrChar(0, box, box->box.h);
        }
        cur_word.len = 0;
    }

    /**/               void AppendText(float x, const string          &text) { p.x=container->w*x; AppendText(StringPiece           (text), 0); }
    /**/               void AppendText(float x, const String16        &text) { p.x=container->w*x; AppendText(String16Piece         (text), 0); }
    template <class X> void AppendText(float x, const X               *text) { p.x=container->w*x; AppendText(StringPiece::Unbounded(text), 0); }
    template <class X> void AppendText(float x, const StringPieceT<X> &text) { p.x=container->w*x; AppendText<X>(                    text,  0); }

    /**/               void AppendText(const string          &text, int attr_id=0) { AppendText(StringPiece           (text), attr_id); }
    /**/               void AppendText(const String16        &text, int attr_id=0) { AppendText(String16Piece         (text), attr_id); }
    template <class X> void AppendText(const X               *text, int attr_id=0) { AppendText(StringPiece::Unbounded(text), attr_id); }
    template <class X> void AppendText(const StringPieceT<X> &text, int attr_id=0) {
        if (!attr_id) attr_id = out->attr.GetAttrId(cur_attr);
        out->data.reserve(out->data.size() + text.size());
        int initial_out_lines = out->line.size(), line_start_ind = 0, c_bytes = 0, ci_bytes = 0;
        for (const X *p = text.data(); !text.Done(p); p += c_bytes) {
            int c = UTF<X>::ReadGlyph(text, p, &c_bytes);
            if (AppendChar(c, attr_id, &PushBack(out->data, Drawable::Box())) == State::NEW_WORD) {
                for (const X *pi=p; !text.Done(pi) && notspace(*pi); pi += ci_bytes)
                    cur_word.len += cur_attr.font->GetGlyphWidth(UTF<X>::ReadGlyph(text, pi, &ci_bytes));
                AppendChar(c, attr_id, &out->data.back());
            }
        }
    }

    State AppendChar(int c, int attr_id, Drawable::Box *box) {
        if (layout.char_tf) c = layout.char_tf(c);
        if (state == State::NEW_WORD && layout.word_start_char_tf) c = layout.word_start_char_tf(c);
        Max(&cur_line.height, cur_attr.font->Height());
        box->drawable = cur_attr.font->FindGlyph(c);
        box->attr_id = attr_id;
        box->line_id = out ? out->line.size() : -1;
        return AppendBoxOrChar(c, box, cur_attr.font->Height());
    }
    State AppendBoxOrChar(int c, Drawable::Box *box, int h) {
        bool space = isspace(c);
        if (space) cur_word.len = 0;
        int max_line_shifts = 1000;
        for (; layout.wrap_lines && max_line_shifts; max_line_shifts--) {
            bool wrap = 0;
            if (!cur_word.len) cur_word.fresh = 1;
            if (!layout.word_break) {
                int box_width = box->drawable ? box->drawable->Advance(&box->box, &cur_attr) : box->box.w;
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

        int advance = box->drawable ? box->drawable->Layout(&box->box, &cur_attr) : box->box.w;
        box->box.y += cur_line.descent;
        box->box += p;
        p.x += advance;
        return state = State::OK;
    }

    void AppendNewlines(int n) { for (int i=0; i<n; i++) AppendNewline(); }
    State AppendNewline(int need_height=0, bool next_glyph_preadded=1) {
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

    void AlignCurrentLine() {
        if (cur_line.out_ind >= out->data.size() || (!layout.align_center && !layout.align_right)) return;
        int line_size = cur_line.end - cur_line.beg, line_min_x, line_max_x;
        GetCurrentLineExtents(&line_min_x, &line_max_x);
        int line_len = line_max_x - line_min_x, align = 0;
        if      (layout.align_center) align = (line_size - line_len)/2;
        else if (layout.align_right)  align = (line_size - line_len);
        if (align) MoveCurrentLine(point(align, 0));
    }
    void MoveCurrentLine(const point &dx) { 
        for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) i->box += dx;
    }
    void GetCurrentLineExtents(int *min_x, int *max_x) { 
        *min_x=INT_MAX; *max_x=INT_MIN;
        for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) { Min(min_x, i->box.x); Max(max_x, i->box.right()); } 
    }
    void SetCurrentLineBounds() {
        cur_line.beg = container->baseleft (p.y, cur_line.height, &adj_float_left)  - container->x;
        cur_line.end = container->baseright(p.y, cur_line.height, &adj_float_right) - container->x;
        p.x = cur_line.beg;
    }
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
    void SetMinColumnWidth(int j, int width, int colspan=1) {
        EnsureSize(column, j+colspan);
        if (width) for (int v=width/colspan, k=j; k<j+colspan; k++) Max(&column[k].width, v);
    }
    Column *SetCellDim(int j, int width, int colspan=1, int rowspan=1) {
        while (VectorEnsureElement(column, j+col_skipped)->remaining_rowspan) col_skipped++;
        SetMinColumnWidth(j+col_skipped, width, colspan);
        for (int k = 0; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan += rowspan;
        return &column[j+col_skipped];
    }
    void NextRowDim() { col_skipped=0; TableFlowColIter(this) if (cj->remaining_rowspan) cj->remaining_rowspan--; }
    int ComputeWidth(int fixed_width) {
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
    void AppendCell(int j, Box *out, int colspan=1) {
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
    void SetCellHeight(int j, int cellheight, void *cell, int colspan=1, int rowspan=1) {
        column[j+col_skipped].AddHeight(cellheight, rowspan, cell);
        for (int k = 1; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan = rowspan;
        col_skipped += colspan-1;

        if (rowspan == 1)   max_cell_height = max(max_cell_height,   cellheight);
        else              split_cell_height = max(split_cell_height, cellheight / rowspan);
    }
    int AppendRow() {
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
};

}; // namespace LFL
#endif // __LFL_LFAPP_FLOW_H__
