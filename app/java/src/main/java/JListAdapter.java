package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.RelativeLayout;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.BaseAdapter;
import android.widget.ListView;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.app.ActionBar;
import android.app.AlertDialog;

public class JListAdapter extends BaseAdapter {
    public static class ViewHolder {
        public TextView textView;
        public EditText editText;
    }

    public static class Section {
        public int start_row;
        public Section(Integer v) { start_row = v; }
    }

    public LayoutInflater inflater = null;
    public ArrayList<JModelItem> data = new ArrayList<JModelItem>();
    public ArrayList<Section> sections = new ArrayList<Section>();
    public int selected_section = 0, selected_row = 0;
    public int editable_section = -1, editable_start_row = -1;
    public long delete_row_cb = 0;
    public native void close();

    public JListAdapter(final MainActivity activity, final ArrayList<JModelItem> v) {
        inflater = (LayoutInflater)activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        data = v;
        beginUpdates();
        data.add(0, new JModelItem("", "", JModelItem.TYPE_SEPARATOR, 0, 0, 0, 0, 0, false));
        for (int i = 0, l = data.size(); i != l; ++i)
            if (data.get(i).type == JModelItem.TYPE_SEPARATOR) sections.add(new Section(i));
        endUpdates();
    }
    
    @Override
    protected void finalize() throws Throwable { try { close(); } finally { super.finalize(); } }

    @Override
    public int getItemViewType(int position) { return data.get(position).type; }

    @Override
    public int getViewTypeCount() { return JModelItem.TYPE_COUNT; }

    @Override
    public int getCount() { return data.size(); }

    @Override
    public String getItem(int position) { return data.get(position).key; }

    @Override
    public long getItemId(int position) { return position; }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        int type = getItemViewType(position);
        ViewHolder holder = null;
        if (convertView == null) {
            holder = new ViewHolder();
            switch (type) {
                case JModelItem.TYPE_TEXTINPUT:
                case JModelItem.TYPE_NUMBERINPUT:
                case JModelItem.TYPE_PASSWORDINPUT:
                    convertView = inflater.inflate(R.layout.listview_cell_textinput, null);
                    holder.textView = (TextView)convertView.findViewById(R.id.listview_cell_title);
                    holder.editText = (EditText)convertView.findViewById(R.id.listview_cell_textinput);
                    break;
                default:
                    convertView = inflater.inflate(R.layout.listview_cell, null);
                    holder.textView = (TextView)convertView.findViewById(R.id.listview_cell_title);
                    holder.editText = null;
                    break;
            }
            convertView.setTag(holder);
        } else {
            holder = (ViewHolder)convertView.getTag();
        }
        holder.textView.setText(type == JModelItem.TYPE_SEPARATOR ? "" : data.get(position).key);
        return convertView;
    }

    public View getViewByPosition(ListView listview, int pos) {
        final int firstListItemPosition = listview.getFirstVisiblePosition();
        final int lastListItemPosition = firstListItemPosition + listview.getChildCount() - 1;
   
        if (pos < firstListItemPosition || pos > lastListItemPosition ) {
            return listview.getAdapter().getView(pos, null, listview);
        } else {
            final int childIndex = pos - firstListItemPosition;
            return listview.getChildAt(childIndex);
        }
    }

    public int getSectionBeginRowId(final int section) { 
      return sections.get(section).start_row;
    }

    public int getSectionEndRowId(final int section) {
        return ((section+1) == sections.size() ? data.size() : getSectionBeginRowId(section+1)) - 1;
    }

    public int getSectionSize(final int section) {
        return getSectionEndRowId(section) - getSectionBeginRowId(section);
    }

    public int getCollapsedRowId(final int section, final int row) {
        return getSectionBeginRowId(section) + 1 + row;
    }

    public void beginUpdates() {}
    public void endUpdates() { notifyDataSetChanged(); }

    public void selectRow(final int s, final int r) {
        selected_section = s;
        selected_row = r;
    }

    public void setEditable(final int s, final int start_row, final long intint_cb) {
        editable_section = s;
        editable_start_row = start_row;
        delete_row_cb = intint_cb;
    }

    public void addSection() {
        sections.add(new Section(data.size()));
        data.add(new JModelItem("", "", JModelItem.TYPE_SEPARATOR, 0, 0, 0, 0, 0, false));
    }

    public void addRow(final int section, JModelItem row) {
        if (section == sections.size()) addSection();
        assert section < sections.size();
        data.add(getSectionEndRowId(section), row);
        moveSectionsAfterBy(section, 1);
    }

    public void replaceSection(final String h, final int image, final int flag, final int section, ArrayList<JModelItem> v) {
        if (section == sections.size()) addSection();
        assert section < sections.size();
        int start_row = getSectionBeginRowId(section), old_section_size = getSectionSize(section);
        data.subList(start_row + 1, start_row + 1 + old_section_size).clear();
        data.addAll(start_row + 1, v);
        moveSectionsAfterBy(section, v.size() - getSectionSize(section));
    }

    public void setSectionValues(final int section, ArrayList<String> v) {
        if (section == sections.size()) addSection();
        assert section < sections.size();
        int section_size = getSectionSize(section), section_row = getSectionBeginRowId(section);
        assert section_size == v.size();
        for (int i = 0; i < section_size; ++i) {
            data.get(section_row + i).val = v.get(i);
        }
    }

    public void moveSectionsAfterBy(final int section, final int delta) {
        for (int i = section + 1, l = sections.size(); i < l; i++) sections.get(i).start_row += delta;
    }

    public String getKey(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).val; }
    public int    getTag(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).tag; }

    public void setTag   (final int s, final int r, final int     v) { data.get(getCollapsedRowId(s, r)).tag = v; } 
    public void setKey   (final int s, final int r, final String  v) { data.get(getCollapsedRowId(s, r)).key = v; }
    public void setValue (final int s, final int r, final String  v) { data.get(getCollapsedRowId(s, r)).val = v; }
    public void setHidden(final int s, final int r, final boolean v) { data.get(getCollapsedRowId(s, r)).hidden = v; }

    public ArrayList<Pair<String, String>> getSectionText(ListView listview, final int section) {
        ArrayList<Pair<String, String>> ret = new ArrayList<Pair<String, String>>();
        final int start_section = section > 0               ? sections.get(section-1).start_row+1 : 0;
        final int   end_section = section < sections.size() ? sections.get(section).start_row-1   : data.size();
        for (int i = start_section; i < end_section; i++) {
            String valtext = "";
            View rowview = getViewByPosition(listview, i);
            ViewHolder holder = (ViewHolder)rowview.getTag();
            if (holder.editText != null)
                valtext = holder.editText.getText().toString();
            ret.add(new Pair<String, String>(data.get(i).key, (valtext.length() > 0) ? valtext : data.get(i).val));
        }
        return ret;
    }
}
