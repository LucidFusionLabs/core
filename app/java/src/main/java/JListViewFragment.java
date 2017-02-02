package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.List;

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.LinearLayout;
import android.util.Log;

public class JListViewFragment extends Fragment implements OnItemClickListener {
    public MainActivity main_activity;
    public JWidget parent_widget;
    public JListAdapter data;
    public ListView listview;
    public View toolbar;
    public long lfl_self;

    JListViewFragment(final MainActivity activity, final JWidget p, final JListAdapter d, final View tb, final long lsp) {
        main_activity = activity;
        parent_widget = p;
        data = d;
        toolbar = tb;
        lfl_self = lsp;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        LinearLayout layout = (LinearLayout)inflater.inflate(R.layout.listview_main, container, false);
        if (toolbar != null) {
            ViewGroup parent = (ViewGroup)toolbar.getParent();
            if (parent != null) parent.removeView(toolbar);
            layout.addView(toolbar, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                                                                  LinearLayout.LayoutParams.WRAP_CONTENT,
                                                                  android.view.Gravity.BOTTOM));
        }
        listview = (ListView)layout.findViewById(R.id.list);
        listview.setAdapter(data);
        listview.setOnItemClickListener(this);
        return layout;
    }

    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        JModelItem i = data.data.get(position);
        if (i.type == JModelItem.TYPE_COMMAND || i.type == JModelItem.TYPE_BUTTON) {
            if (i.cb != null) i.cb.run();
        } else if (i.type == JModelItem.TYPE_LABEL && position+1 < data.data.size()) {
            JModelItem ni = data.data.get(position+1);
            if (ni.type == JModelItem.TYPE_PICKER || ni.type == JModelItem.TYPE_FONTPICKER) {
                data.beginUpdates();
                ni.hidden = !ni.hidden;
                data.endUpdates();
            }
        }
    }
}
