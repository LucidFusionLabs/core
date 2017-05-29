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
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.NumberPicker;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.widget.CompoundButton;
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
import android.graphics.Color;
import android.app.AlertDialog;
import android.text.InputType;
import android.text.TextWatcher;
import android.support.v7.widget.RecyclerView;
import com.lucidfusionlabs.core.ModelItem;
import com.lucidfusionlabs.core.PickerItem;
import com.lucidfusionlabs.core.NativeIntIntCB;
import com.lucidfusionlabs.core.ModelItemChange;
import com.lucidfusionlabs.core.ModelItemLinearLayout;

public class ModelItemRecyclerViewAdapter
    extends RecyclerView.Adapter<ModelItemRecyclerViewAdapter.ViewHolder>
    implements com.lucidfusionlabs.core.ModelItemList {

    public static class ViewHolder extends RecyclerView.ViewHolder {
        public ModelItemLinearLayout root;
        public TextView textView, label, subtext;
        public ColorStateList textViewTextColors, textViewLinkColors;
        public EditText editText;
        public ImageView leftIcon, rightIcon;
        public Button leftNav, rightNav;
        public Switch toggle;
        public NumberPicker picker;
        public RadioGroup radio;

        public ViewHolder(View itemView) {
            super(itemView);
        }

        public void setTextView(TextView v) {
            textView = v;
            textViewTextColors = textView.getTextColors();
            textViewLinkColors = textView.getLinkTextColors();
        }
    }

    public static class Section implements Comparable<Integer> {
        public Integer start_row;
        public Section(Integer v) { start_row = v; }
        @Override public int compareTo(Integer x) { return start_row.compareTo(x); }
    }

    public final TableScreen parent_screen;
    public ArrayList<ModelItem> data = new ArrayList<ModelItem>();
    public ArrayList<Section> sections = new ArrayList<Section>();
    public ModelItem nav_left, nav_right;
    public int selected_section = 0, selected_row = 0;
    public int editable_section = -1, editable_start_row = -1;
    public NativeIntIntCB delete_row_cb;
    public RecyclerView recyclerview;
    public com.lucidfusionlabs.core.ViewOwner toolbar, advertising;

    CompoundButton.OnCheckedChangeListener checked_listener = new CompoundButton.OnCheckedChangeListener() {
        public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
            parent_screen.changed = true;
        }
    };

    TextWatcher text_listener = new TextWatcher() {
        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {
            parent_screen.changed = true;
        }
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
        @Override
        public void afterTextChanged(android.text.Editable s) {}
    };

    public ModelItemRecyclerViewAdapter(final TableScreen p, final ArrayList<ModelItem> d) {
        parent_screen = p;
        data = d;
        beginUpdates();
        data.add(0, new ModelItem("", "", "", "", ModelItem.TYPE_SEPARATOR, 0, 0, 0, 0, 0, 0, null, null, null, false, 0, 0));
        data.add(new ModelItem("", "", "", "", ModelItem.TYPE_SEPARATOR, 0, 0, 0, 0, 0, 0, null, null, null, false, 0, 0));
        for (int i = 0, l = data.size(); i != l; ++i) {
            ModelItem item = data.get(i);
            if (item.type == ModelItem.TYPE_SEPARATOR) sections.add(new Section(i));
        }
        notifyDataSetChanged(); 
    }

    @Override
    public int getItemCount() { return data.size() - 1; }

    @Override
    public int getItemViewType(int position) {
        ModelItem item = data.get(position);
        if (item.hidden) return ModelItem.TYPE_HIDDEN;
        else if (item.type == ModelItem.TYPE_SEPARATOR && (position+1) < data.size() &&
            data.get(position+1).type == ModelItem.TYPE_SEPARATOR) return ModelItem.TYPE_HIDDEN;
        else if ((item.flags & ModelItem.TABLE_FLAG_HIDEKEY) != 0) {
            if (item.type == ModelItem.TYPE_SELECTOR) return ModelItem.TYPE_SELECTOR_HIDEKEY;
        }
        return item.type;
    }

    @Override
    public void onAttachedToRecyclerView(RecyclerView recyclerView) {
        super.onAttachedToRecyclerView(recyclerView);
        recyclerview = recyclerView;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        switch (viewType) {
            case ModelItem.TYPE_HIDDEN: {
                View itemView = inflater.inflate(R.layout.listview_cell_hidden, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                return holder;
            }

            case ModelItem.TYPE_SEPARATOR: {
                View itemView = inflater.inflate(R.layout.listview_cell_separator, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.setTextView((TextView) itemView.findViewById(R.id.listview_cell_title));
                holder.leftIcon = (ImageView) itemView.findViewById(R.id.listview_cell_left_icon);
                holder.leftNav = (Button) itemView.findViewById(R.id.listview_cell_nav_left);
                holder.rightNav = (Button) itemView.findViewById(R.id.listview_cell_nav_right);
                return holder;
            }

            case ModelItem.TYPE_TOGGLE: {
                View itemView = inflater.inflate(R.layout.listview_cell_toggle, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.setTextView((TextView) itemView.findViewById(R.id.listview_cell_title));
                holder.leftIcon = (ImageView) itemView.findViewById(R.id.listview_cell_left_icon);
                holder.toggle = (Switch) itemView.findViewById(R.id.listview_cell_toggle);
                return holder;
            }

            case ModelItem.TYPE_PICKER: {
                View itemView = inflater.inflate(R.layout.listview_cell_picker, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.picker = (NumberPicker) itemView.findViewById(R.id.listview_cell_picker);
                return holder;
            }

            case ModelItem.TYPE_TEXTINPUT:
            case ModelItem.TYPE_NUMBERINPUT:
            case ModelItem.TYPE_PASSWORDINPUT: {
                View itemView = inflater.inflate(R.layout.listview_cell_textinput, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                itemView.setMinimumHeight(parent.getMeasuredHeight() * 2);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.setTextView((TextView) itemView.findViewById(R.id.listview_cell_title));
                holder.editText = (EditText) itemView.findViewById(R.id.listview_cell_textinput);
                holder.leftIcon = (ImageView) itemView.findViewById(R.id.listview_cell_left_icon);
                return holder;
            }

            case ModelItem.TYPE_SELECTOR: {
                View itemView = inflater.inflate(R.layout.listview_cell_radio, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.setTextView((TextView) itemView.findViewById(R.id.listview_cell_title));
                holder.radio = (RadioGroup) itemView.findViewById(R.id.listview_cell_radio);
                holder.leftIcon = (ImageView) itemView.findViewById(R.id.listview_cell_left_icon);
                return holder;
            }

            case ModelItem.TYPE_SELECTOR_HIDEKEY: {
                View itemView = inflater.inflate(R.layout.listview_cell_radio_hidekey, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.radio = (RadioGroup) itemView.findViewById(R.id.listview_cell_radio);
                return holder;
            }

            default: {
                View itemView = inflater.inflate(R.layout.listview_cell, parent, false);
                ViewHolder holder = new ViewHolder(itemView);
                itemView.setTag(holder);
                holder.root = (ModelItemLinearLayout)itemView.findViewById(R.id.listview_cell_root);
                holder.setTextView((TextView) itemView.findViewById(R.id.listview_cell_title));
                holder.label = (TextView) itemView.findViewById(R.id.listview_cell_value);
                holder.subtext = (TextView) itemView.findViewById(R.id.listview_cell_subtext);
                holder.leftIcon = (ImageView) itemView.findViewById(R.id.listview_cell_left_icon);
                holder.rightIcon = (ImageView) itemView.findViewById(R.id.listview_cell_right_icon);
                return holder;
            }
        }
    }

    @Override
    public void onBindViewHolder(final ViewHolder holder, final int position) {
        final ModelItem item = data.get(position);
        int type = getItemViewType(position);

        if (holder.itemView != null) {
            if (item.cb != null) holder.itemView.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { item.cb.run(); }});
            else                 holder.itemView.setOnClickListener(null);
        }

        if (holder.root != null) {
            holder.root.item = item;
            holder.root.section = data.get(getSectionBeginRowIdFromPosition(position));
        }

        if (holder.textView != null) {
            if (item.dropdown_key.length() > 0 && item.key.length() > 0 && item.cb != null &&
                (item.flags & ModelItem.TABLE_FLAG_FIXDROPDOWN) == 0) {
                holder.textView.setText(item.key + " \u02C5");
                holder.textView.setTextColor(holder.textViewLinkColors);
                holder.textView.setPaintFlags(holder.textView.getPaintFlags() | Paint.UNDERLINE_TEXT_FLAG);
            } else {
                holder.textView.setText(item.key);
                holder.textView.setTextColor(holder.textViewTextColors);
                holder.textView.setPaintFlags(holder.textView.getPaintFlags() & (~Paint.UNDERLINE_TEXT_FLAG));
            }
        }

        if (holder.leftIcon != null) {
            if (item.left_icon < 0) {
                holder.leftIcon.setImageBitmap(MainActivity.bitmaps.get(-item.left_icon-1));
                if (getCollapsedRowId(selected_section, selected_row) == position) holder.root.setBackgroundColor(Color.GRAY);
                else                                                               holder.root.setBackgroundColor(Color.TRANSPARENT);
            } else {
                holder.leftIcon.setImageResource(item.left_icon);
            }
            holder.leftIcon.setVisibility(item.left_icon == 0 ? View.GONE : View.VISIBLE);
        }

        if (holder.rightIcon != null) {
            holder.rightIcon.setImageResource(item.right_icon);
            if (item.right_cb == null) holder.rightIcon.setClickable(false);
            else {
                holder.rightIcon.setClickable(true);
                holder.rightIcon.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { item.right_cb.run(""); }});
            }
        }

        boolean subtext = (item.flags & ModelItem.TABLE_FLAG_SUBTEXT) != 0;
        if (holder.subtext != null) {
            holder.subtext.setText(subtext ? item.val : "");
        }

        if (holder.label != null) {
            holder.label.setText(item.right_text.length() > 0 ? item.right_text : (subtext ? "" : item.val));
            if (holder.textView != null) {
                if (type == ModelItem.TYPE_BUTTON || type == ModelItem.TYPE_COMMAND ||
                    type == ModelItem.TYPE_NONE) {
                  holder.label.setTextColor(holder.textViewLinkColors);
                } else {
                  holder.label.setTextColor(holder.textViewTextColors);
                }
            }
            holder.label.setVisibility(item.right_icon != 0 ? View.GONE : View.VISIBLE);
        }

        if (holder.editText != null) {
            holder.editText.removeTextChangedListener(text_listener);
            if (item.val.length() > 0 && item.val.charAt(0) == 1) { holder.editText.setText(""); holder.editText.setHint(item.val.substring(1)); }
            else                                                  { holder.editText.setText(item.val); holder.editText.setHint(""); }
            holder.editText.setInputType(InputType.TYPE_CLASS_TEXT | (type == ModelItem.TYPE_PASSWORDINPUT ? InputType.TYPE_TEXT_VARIATION_PASSWORD : 0));
            holder.editText.addTextChangedListener(text_listener);
        }

        switch (type) {
            case ModelItem.TYPE_SEPARATOR:
                holder.leftNav .setVisibility(item.val       .length() > 0 ? View.VISIBLE : View.GONE);
                holder.rightNav.setVisibility(item.right_text.length() > 0 ? View.VISIBLE : View.GONE);
                break;

            case ModelItem.TYPE_TOGGLE:
                holder.toggle.setOnCheckedChangeListener(null);
                holder.toggle.setChecked(item.val.equals("1"));
                holder.toggle.setOnCheckedChangeListener(checked_listener);
                break;
        }

        if (holder.picker != null && item.picker != null && item.picker.data.size() > 0) {
            ArrayList<String> picker_items = item.picker.data.get(0);
            if (picker_items.size() > 0) {
                holder.picker.setOnValueChangedListener(null);
                holder.picker.setMinValue(0);
                holder.picker.setMaxValue(picker_items.size() - 1);
                String[] arr = new String[picker_items.size()];
                arr = picker_items.toArray(arr);
                holder.picker.setDisplayedValues(arr);
                if (position > 1 && data.get(position-1).type == ModelItem.TYPE_LABEL) {
                    String v = data.get(position-1).val;
                    for (int i = 0; i < arr.length; i++)
                        if (v.equals(arr[i])) { holder.picker.setValue(i); break; }

                    final RecyclerView par = recyclerview;
                    final int pos = position;
                    holder.picker.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
                        @Override
                        public void onValueChange(NumberPicker numberPicker, int i, int i2) {
                            parent_screen.changed = true;
                            String v = numberPicker.getDisplayedValues()[numberPicker.getValue()];
                            ViewHolder h = (ViewHolder)par.findViewHolderForAdapterPosition(pos-1);
                            h.label.setText(v);
                            data.get(pos-1).val = v;
                        }});
                } else holder.picker.setValue(0);
            }
        }

        if (holder.radio != null) {
            final ModelItemRecyclerViewAdapter self = this;
            holder.radio.setOnCheckedChangeListener(null);
            holder.radio.removeAllViews();

            String[] v = item.val.split(",");
            for(int i = 0; i < v.length; i++) {
                RadioGroup.LayoutParams layout = new RadioGroup.LayoutParams
                    (RadioGroup.LayoutParams.MATCH_PARENT, RadioGroup.LayoutParams.WRAP_CONTENT);
                layout.weight = 1;
                RadioButton button = new RadioButton(holder.radio.getContext());
                button.setLayoutParams(layout);
                button.setId(i);
                button.setText(v[i]);
                button.setChecked(item.selected == i);
                if (type == ModelItem.TYPE_SELECTOR_HIDEKEY)
                    button.setBackgroundResource(R.drawable.listview_radio_selector);
                holder.radio.addView(button);
            }

            holder.radio.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(RadioGroup group, int checkedId) {
                    parent_screen.changed = true;
                    RadioButton button = (RadioButton)group.findViewById(checkedId);
                    if (button != null && item.right_cb != null)
                        item.right_cb.run(button.getText().toString());
                }});
        }
    }

    @Override public ArrayList<ModelItem> getData() { return data; }

    @Override public int getSectionBeginRowId(final int section) { 
      return sections.get(section).start_row;
    }

    @Override public int getSectionEndRowId(final int section) {
        return ((section+1) == sections.size() ? data.size() : getSectionBeginRowId(section+1)) - 1;
    }

    @Override public int getSectionSize(final int section) {
        return getSectionEndRowId(section) - getSectionBeginRowId(section);
    }

    @Override public int getCollapsedRowId(final int section, final int row) {
        return getSectionBeginRowId(section) + 1 + row;
    }

    public int getSectionBeginRowIdFromPosition(final int position) {
       return (sections.size() == 0) ? 0 : 
           sections.get(com.lucidfusionlabs.core.Util.lesserBound(sections, position)).start_row;
    }

    public void beginUpdates() {}
    public void endUpdates() {}

    public void selectRow(final int s, final int r) {
        selected_section = s;
        selected_row = r;
    }

    public void setEditable(final int s, final int start_row, final NativeIntIntCB intint_cb) {
        editable_section = s;
        editable_start_row = start_row;
        delete_row_cb = intint_cb;
    }

    public void addNavButton(final int halign, final ModelItem row) {
        switch (halign) {
            case ModelItem.HALIGN_LEFT:
                nav_left = row;
                break;

            case ModelItem.HALIGN_RIGHT:
                nav_right = row;
                break;
        }
    }
    
    public void delNavButton(final int halign) {
        switch (halign) {
            case ModelItem.HALIGN_LEFT:
                nav_left = null;
                break;

            case ModelItem.HALIGN_RIGHT:
                nav_right = null;
                break;
        }
    }

    public void addSection() {
        sections.add(new Section(data.size()));
        data.add(new ModelItem("", "", "", "", ModelItem.TYPE_SEPARATOR, 0, 0, 0, 0, 0, 0, null, null, null, false, 0, 0));
        notifyItemInserted(data.size()-1);
    }

    public void setHeader(final int section, ModelItem row) {
        int row_id = getSectionBeginRowId(section);
        data.set(row_id, row);
        notifyItemChanged(row_id);
    }

    public void addRow(final int section, ModelItem row) {
        if (section == (sections.size()-1)) addSection();
        if (section >= (sections.size()-1)) throw new java.lang.IllegalArgumentException();
        int insert_at = getSectionEndRowId(section);
        data.add(insert_at, row);
        moveSectionsAfterBy(section, 1);
        notifyItemInserted(insert_at);
    }
    
    public void replaceRow(final int s, final int r, ModelItem row) {
        int row_id = getCollapsedRowId(s, r); 
        data.set(row_id, row);
        notifyItemChanged(row_id);
    }

    public void replaceSection(final int section, final ModelItem h, final int flags, ArrayList<ModelItem> v) {
        if (section == (sections.size()-1)) addSection();
        if (section >= (sections.size()-1)) throw new java.lang.IllegalArgumentException();
        int start_row = getSectionBeginRowId(section), old_section_size = getSectionSize(section);
        h.flags = flags;
        data.set(start_row, h);
        notifyItemChanged(start_row);

        data.subList(start_row + 1, start_row + 1 + old_section_size).clear();
        moveSectionsAfterBy(section, -old_section_size);
        notifyItemRangeRemoved(start_row + 1, old_section_size);

        data.addAll(start_row + 1, v);
        moveSectionsAfterBy(section, v.size());
        notifyItemRangeInserted(start_row + 1, v.size());
    }

    public void setSectionValues(final int section, ArrayList<String> v) {
        if (section == sections.size()) addSection();
        if (section >= sections.size()) throw new java.lang.IllegalArgumentException();
        int section_size = getSectionSize(section), section_row = getSectionBeginRowId(section);
        if (section_size != v.size()) throw new java.lang.IllegalArgumentException();
        for (int i = 0; i < section_size; ++i) {
            data.get(section_row + 1 + i).val = v.get(i);
        }
        notifyItemRangeChanged(section_row + 1, section_size);
    }

    public void applyChangeList(final ArrayList<ModelItemChange> changes) {
        ModelItemChange.applyChangeList(changes, this, new ModelItemChange.ChangeWatcher() { @Override public void onChanged(int row) {
            notifyItemChanged(row);
        }});
    }

    public void moveSectionsAfterBy(final int section, final int delta) {
        for (int i = section + 1, l = sections.size(); i < l; i++) sections.get(i).start_row += delta;
    }

    public String getKey(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).key; }
    public String getVal(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).val; }
    public int    getTag(final int s, final int r) { return data.get(getCollapsedRowId(s, r)).tag; }

    public void setKey   (final int s, final int r, final String  v) { data.get(getCollapsedRowId(s, r)).key = v; }
    public void setTag   (final int s, final int r, final int     v) { data.get(getCollapsedRowId(s, r)).tag = v; } 
    public void setValue (final int s, final int r, final String  v) { data.get(getCollapsedRowId(s, r)).val = v; }
    public void setSelected(final int s, final int r, final int   v) { data.get(getCollapsedRowId(s, r)).selected = v; }
    
    public void setHidden(final int s, final int r, final int v) {
        int row_id = getCollapsedRowId(s, r);
        if (v < 0) {
            ModelItem item = data.get(row_id);
            item.hidden = !item.hidden;
        } else data.get(row_id).hidden = v > 0;
        notifyItemChanged(row_id);
    }

    public Pair<Long, ArrayList<Integer>> getPicked(final int s, final int r) {
        PickerItem picker = data.get(getCollapsedRowId(s, r)).picker;
        if (picker == null) return null;
        return new Pair<Long, ArrayList<Integer>>(picker.nativeParent, picker.picked);
    }

    public ArrayList<Pair<String, String>> getSectionText(RecyclerView recyclerview, final int section) {
        if (section >= sections.size()) throw new java.lang.IllegalArgumentException();
        ArrayList<Pair<String, String>> ret = new ArrayList<Pair<String, String>>();
        int section_size = getSectionSize(section), section_row = getSectionBeginRowId(section);
        for (int i = 0; i < section_size; ++i) {
            String val = "";
            ModelItem item = data.get(section_row + 1 + i);
            if (recyclerview != null) {
                ViewHolder holder = (ViewHolder)recyclerview.findViewHolderForAdapterPosition(section_row + 1 + i);
                if (holder.toggle != null) val = holder.toggle.isChecked() ? "1" : "0";
                else if (holder.radio != null) {
                    int checked = holder.radio.getCheckedRadioButtonId();
                    RadioButton button = (RadioButton)holder.radio.findViewById(checked);
                    val = button.getText().toString();
                } else if (holder.editText != null) val = holder.editText.getText().toString();
            }
            if (item.dropdown_key.length() > 0) ret.add(new Pair<String, String>(item.dropdown_key, item.key)); 
            if (val.length() == 0 && item.val.length() > 0 && item.val.charAt(0) != 1) val = item.val;
            ret.add(new Pair<String, String>(item.key, val));
        }
        return ret;
    }
}
