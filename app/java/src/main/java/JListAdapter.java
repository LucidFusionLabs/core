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
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.NumberPicker;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.res.ColorStateList;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.graphics.Paint;
import android.app.ActionBar;
import android.app.AlertDialog;
import android.text.InputType;

public class JListAdapter extends BaseAdapter {
    public static class ViewHolder {
        public TextView textView, label;
        public ColorStateList textViewTextColors, textViewLinkColors;
        public EditText editText;
        public ImageView leftIcon, rightIcon;
        public Button leftNav, rightNav;
        public Switch toggle;
        public NumberPicker picker;
        public RadioGroup radio;

        public void setTextView(TextView v) {
            textView = v;
            textViewTextColors = textView.getTextColors();
            textViewLinkColors = textView.getLinkTextColors();
        }
    }

    public static class Section {
        public int start_row;
        public Section(Integer v) { start_row = v; }
    }

    public LayoutInflater inflater = null;
    public ArrayList<JModelItem> data = new ArrayList<JModelItem>();
    public ArrayList<Section> sections = new ArrayList<Section>();
    public JModelItem nav_left = null, nav_right = null;
    public int selected_section = 0, selected_row = 0;
    public int editable_section = -1, editable_start_row = -1;
    public LIntIntCB delete_row_cb = null;

    public JListAdapter(final MainActivity activity, final ArrayList<JModelItem> v) {
        inflater = (LayoutInflater)activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        data = v;
        beginUpdates();
        data.add(0, new JModelItem("", "", "", "", JModelItem.TYPE_SEPARATOR, 0, 0, 0, null, null, null, false, null, null));
        data.add(new JModelItem("", "", "", "", JModelItem.TYPE_SEPARATOR, 0, 0, 0, null, null, null, false, null, null));
        for (int i = 0, l = data.size(); i != l; ++i) {
            JModelItem item = data.get(i);
            if (item.type == JModelItem.TYPE_SEPARATOR) sections.add(new Section(i));
        }
        endUpdates();
    }
    
    @Override
    public int getItemViewType(int position) {
        JModelItem item = data.get(position);
        return item.hidden ? JModelItem.TYPE_HIDDEN : item.type;
    }

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
                case JModelItem.TYPE_HIDDEN:
                    convertView = inflater.inflate(R.layout.listview_cell_hidden, null);
                    holder.textView = null;
                    holder.label = null;
                    holder.editText = null;
                    holder.leftIcon = null;
                    holder.rightIcon = null;
                    holder.toggle = null;
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = null;
                    holder.radio = null;
                    break;

                case JModelItem.TYPE_SEPARATOR:
                    convertView = inflater.inflate(R.layout.listview_cell_separator, null);
                    holder.setTextView((TextView)convertView.findViewById(R.id.listview_cell_title));
                    holder.label = null;
                    holder.editText = null;
                    holder.leftIcon = (ImageView)convertView.findViewById(R.id.listview_cell_left_icon);
                    holder.rightIcon = null;
                    holder.toggle = null;
                    holder.leftNav = (Button)convertView.findViewById(R.id.listview_cell_nav_left);
                    holder.rightNav = (Button)convertView.findViewById(R.id.listview_cell_nav_right);
                    holder.picker = null;
                    holder.radio = null;
                    break;

                case JModelItem.TYPE_TOGGLE:
                    convertView = inflater.inflate(R.layout.listview_cell_toggle, null);
                    holder.setTextView((TextView)convertView.findViewById(R.id.listview_cell_title));
                    holder.label = null;
                    holder.editText = null;
                    holder.leftIcon = (ImageView)convertView.findViewById(R.id.listview_cell_left_icon);
                    holder.rightIcon = null;
                    holder.toggle = (Switch)convertView.findViewById(R.id.listview_cell_toggle);
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = null;
                    holder.radio = null;
                    break;

                case JModelItem.TYPE_PICKER:
                    convertView = inflater.inflate(R.layout.listview_cell_picker, null);
                    holder.textView = null;
                    holder.label = null;
                    holder.editText = null;
                    holder.leftIcon = null;
                    holder.rightIcon = null;
                    holder.toggle = null;
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = (NumberPicker)convertView.findViewById(R.id.listview_cell_picker);
                    holder.radio = null;
                    break;

                case JModelItem.TYPE_TEXTINPUT:
                case JModelItem.TYPE_NUMBERINPUT:
                case JModelItem.TYPE_PASSWORDINPUT:
                    convertView = inflater.inflate(R.layout.listview_cell_textinput, null);
                    holder.setTextView((TextView)convertView.findViewById(R.id.listview_cell_title));
                    holder.label = null;
                    holder.editText = (EditText)convertView.findViewById(R.id.listview_cell_textinput);
                    holder.leftIcon = (ImageView)convertView.findViewById(R.id.listview_cell_left_icon);
                    holder.rightIcon = null;
                    holder.toggle = null;
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = null;
                    holder.radio = null;
                    break;

                case JModelItem.TYPE_SELECTOR:
                    convertView = inflater.inflate(R.layout.listview_cell_radio, null);
                    holder.textView = null;
                    holder.label = null;
                    holder.editText = null;
                    holder.leftIcon = null;
                    holder.rightIcon = null;
                    holder.toggle = null;
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = null;
                    holder.radio = (RadioGroup)convertView.findViewById(R.id.listview_cell_radio);
                    break;

                default:
                    convertView = inflater.inflate(R.layout.listview_cell, null);
                    holder.setTextView((TextView)convertView.findViewById(R.id.listview_cell_title));
                    holder.label = (TextView)convertView.findViewById(R.id.listview_cell_value);
                    holder.editText = null;
                    holder.leftIcon = (ImageView)convertView.findViewById(R.id.listview_cell_left_icon);
                    holder.rightIcon = (ImageView)convertView.findViewById(R.id.listview_cell_right_icon);
                    holder.toggle = null;
                    holder.leftNav = null;
                    holder.rightNav = null;
                    holder.picker = null;
                    holder.radio = null;
                    break;
            }
            convertView.setTag(holder);
        } else {
            holder = (ViewHolder)convertView.getTag();
        }

        final JModelItem item = data.get(position);
        if (holder.textView != null) {
            if (item.dropdown_key.length() > 0 && item.key.length() > 0 && item.cb != null &&
                (item.flags & JModelItem.TABLE_FLAG_FIXDROPDOWN) == 0) {
                holder.textView.setText(item.key + " \u02C5");
                holder.textView.setTextColor(holder.textViewLinkColors);
                holder.textView.setPaintFlags(holder.textView.getPaintFlags() | Paint.UNDERLINE_TEXT_FLAG);
                holder.textView.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { item.cb.run(); }});
            } else {
                holder.textView.setText(item.key);
                holder.textView.setTextColor(holder.textViewTextColors);
                holder.textView.setPaintFlags(holder.textView.getPaintFlags() & (~Paint.UNDERLINE_TEXT_FLAG));
                holder.textView.setOnClickListener(null);
            }
        }
        if (holder.leftIcon  != null) holder.leftIcon.setImageResource(item.left_icon);
        if (holder.rightIcon != null) {
            holder.rightIcon.setImageResource(item.right_icon);
            if (item.right_cb == null) holder.rightIcon.setClickable(false);
            else {
                holder.rightIcon.setClickable(true);
                holder.rightIcon.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { item.right_cb.run(); }});
            }
        }
        if (holder.label     != null) holder.label.setText(item.right_text.length() > 0 ? item.right_text : item.val);
        if (holder.editText  != null) {
            if (item.val.length() > 0 && item.val.charAt(0) == 1) { holder.editText.setText(""); holder.editText.setHint(item.val.substring(1)); }
            else                                                  { holder.editText.setText(item.val); holder.editText.setHint(""); }
            holder.editText.setInputType(InputType.TYPE_CLASS_TEXT | (type == JModelItem.TYPE_PASSWORDINPUT ? InputType.TYPE_TEXT_VARIATION_PASSWORD : 0));
        }

        switch (type) {
            case JModelItem.TYPE_SEPARATOR:
                holder.leftNav .setVisibility(item.val       .length() > 0 ? View.VISIBLE : View.GONE);
                holder.rightNav.setVisibility(item.right_text.length() > 0 ? View.VISIBLE : View.GONE);
                break;

            case JModelItem.TYPE_TOGGLE:
                holder.toggle.setChecked(item.val.equals("1"));
                break;
        }

        if (holder.picker != null && item.picker != null && item.picker.data.size() > 0) {
            ArrayList<String> picker_items = item.picker.data.get(0);
            if (picker_items.size() > 0) {
                holder.picker.setMinValue(0);
                holder.picker.setMaxValue(picker_items.size() - 1);
                String[] arr = new String[picker_items.size()];
                arr = picker_items.toArray(arr);
                holder.picker.setDisplayedValues(arr);
            }
        }

        if (holder.radio != null) {
            final JListAdapter self = this;
            holder.radio.removeAllViews();
            String[] v = item.val.split(",");
            for(int i = 0; i < v.length; i++) {
                RadioGroup.LayoutParams layout = new RadioGroup.LayoutParams
                    (RadioGroup.LayoutParams.MATCH_PARENT, RadioGroup.LayoutParams.WRAP_CONTENT);
                layout.weight = 1;
                RadioButton button = new RadioButton(parent.getContext());
                button.setLayoutParams(layout);
                button.setId(i);
                button.setText(v[i]);  
                button.setChecked(item.checkedId == i);
                button.setBackgroundResource(R.drawable.listview_radio_selector);
                holder.radio.addView(button);
            }
            if (item.depends != null)
                holder.radio.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
                    @Override
                    public void onCheckedChanged(RadioGroup group, int checkedId) {
                        RadioButton button = (RadioButton)group.findViewById(checkedId);
                        if (button != null) {
                            self.beginUpdates();
                            item.checkedId = checkedId;
                            JDependencyItem.applyList(item.depends, self, button.getText().toString());
                            self.endUpdates();
                        }
                    }});
        }

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

    public void setEditable(final int s, final int start_row, final LIntIntCB intint_cb) {
        editable_section = s;
        editable_start_row = start_row;
        delete_row_cb = intint_cb;
    }

    public void addNavButton(final int halign, final JModelItem row) {
        switch (halign) {
            case JModelItem.HALIGN_LEFT:
                nav_left = row;
                break;

            case JModelItem.HALIGN_RIGHT:
                nav_right = row;
                break;
        }
    }
    
    public void delNavButton(final int halign) {
        switch (halign) {
            case JModelItem.HALIGN_LEFT:
                nav_left = null;
                break;

            case JModelItem.HALIGN_RIGHT:
                nav_right = null;
                break;
        }
    }

    public void addSection() {
        sections.add(new Section(data.size()));
        data.add(new JModelItem("", "", "", "", JModelItem.TYPE_SEPARATOR, 0, 0, 0, null, null, null, false, null, null));
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
        JModelItem separator = data.get(start_row);
        separator.key = h;
        separator.left_icon = image;
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
            data.get(section_row + 1 + i).val = v.get(i);
        }
    }

    public void moveSectionsAfterBy(final int section, final int delta) {
        for (int i = section + 1, l = sections.size(); i < l; i++) sections.get(i).start_row += delta;
    }

    public String getKey(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).key; }
    public int    getTag(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).tag; }

    public void setTag   (final int s, final int r, final int     v) { data.get(getCollapsedRowId(s, r)).tag = v; } 
    public void setValue (final int s, final int r, final String  v) { data.get(getCollapsedRowId(s, r)).val = v; }
    public void setHidden(final int s, final int r, final boolean v) { data.get(getCollapsedRowId(s, r)).hidden = v; }

    public void setKey(final int s, final int r, final String  v) {
        JModelItem item = data.get(getCollapsedRowId(s, r));
        item.key = v; 
        if (item.depends != null) JDependencyItem.applyList(item.depends, this, v);
    }

    public ArrayList<Pair<String, String>> getSectionText(ListView listview, final int section) {
        assert section < sections.size();
        ArrayList<Pair<String, String>> ret = new ArrayList<Pair<String, String>>();
        int section_size = getSectionSize(section), section_row = getSectionBeginRowId(section);
        for (int i = 0; i < section_size; ++i) {
            String val = "";
            JModelItem item = data.get(section_row + 1 + i);
            if (listview != null) {
                View itemview = getViewByPosition(listview, section_row + 1 + i);
                ViewHolder holder = (ViewHolder)itemview.getTag();
                if (holder.editText != null) val = holder.editText.getText().toString();
            }
            if (item.dropdown_key.length() > 0) ret.add(new Pair<String, String>(item.dropdown_key, item.key)); 
            if (val.length() == 0 && item.val.length() > 0 && item.val.charAt(0) != 1) val = item.val;
            ret.add(new Pair<String, String>(item.key, val));
        }
        return ret;
    }
}
