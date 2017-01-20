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

public class ListAdapter extends BaseAdapter {
    public static class ViewHolder {
        public TextView textView;
        public EditText editText;
    }

    public static final int TYPE_LABEL     = 0;
    public static final int TYPE_COMMAND   = 1;
    public static final int TYPE_BUTTON    = 2;
    public static final int TYPE_TEXTINPUT = 3;
    public static final int TYPE_NUMINPUT  = 4;
    public static final int TYPE_PWINPUT   = 5;
    public static final int TYPE_SEPARATOR = 6;
    public static final int TYPE_MAX       = 7;

    public LayoutInflater     inflater = null;
    public ArrayList<String>  keys     = new ArrayList<String>();
    public ArrayList<Integer> types    = new ArrayList<Integer>();
    public ArrayList<String>  vals     = new ArrayList<String>();
    public ArrayList<Integer> sections = new ArrayList<Integer>();

    public ListAdapter(final MainActivity activity, final ArrayList<JModelItem> item) {
        for (JModelItem r : item) addItem(r.key, "", r.val);
        inflater = (LayoutInflater)activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
    }

    public void addItem(String k, String t, String v) {
        if (k.indexOf(',') >= 0) {
            String[] kv = k.split(",");
            String[] tv = t.split(",");
            k = kv[0];
            t = tv[0];
        }
        keys.add(k);
        vals.add(v);

        int type = TYPE_LABEL;
        if      (k.equals("<separator>")) type = TYPE_SEPARATOR;
        else if (t.equals("command"))     type = TYPE_COMMAND;
        else if (t.equals("button"))      type = TYPE_BUTTON;
        else if (t.equals("textinput"))   type = TYPE_TEXTINPUT;
        else if (t.equals("numinput"))    type = TYPE_NUMINPUT;
        else if (t.equals("pwinput"))     type = TYPE_PWINPUT;
        types.add(type);

        if (type == TYPE_SEPARATOR) sections.add(keys.size());

        if (inflater != null) notifyDataSetChanged();
    }

    @Override
    public int getItemViewType(int position) { return types.get(position); }

    @Override
    public int getViewTypeCount() { return TYPE_MAX; }

    @Override
    public int getCount() { return keys.size(); }

    @Override
    public String getItem(int position) { return keys.get(position); }

    @Override
    public long getItemId(int position) { return position; }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        int type = getItemViewType(position);
        ViewHolder holder = null;
        if (convertView == null) {
            holder = new ViewHolder();
            switch (type) {
                case TYPE_TEXTINPUT:
                case TYPE_NUMINPUT:
                case TYPE_PWINPUT:
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
        holder.textView.setText(type == TYPE_SEPARATOR ? "" : keys.get(position));
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

    public ArrayList<Pair<String, String>> getSectionText(ListView listview, final int section) {
        ArrayList<Pair<String, String>> ret = new ArrayList<Pair<String, String>>();
        final int start_section = section > 0               ? sections.get(section-1)+1 : 0;
        final int   end_section = section < sections.size() ? sections.get(section)-1   : keys.size();
        for (int i = start_section; i < end_section; i++) {
            String valtext = "";
            View rowview = getViewByPosition(listview, i);
            ViewHolder holder = (ViewHolder)rowview.getTag();
            if (holder.editText != null)
                valtext = holder.editText.getText().toString();
            ret.add(new Pair<String, String>(keys.get(i), (valtext.length() > 0) ? valtext : vals.get(i)));
        }
        return ret;
    }
}
