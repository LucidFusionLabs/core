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

public class ListViewFragment extends Fragment implements OnItemClickListener {
    public MainActivity main_activity;
    public String title;
    public ListAdapter data;
    public ListView listview;
    public View toolbar;

    ListViewFragment(final MainActivity activity,
                     final String t, final String[] k, final String[] v, final String[] w, View tb) {
        main_activity = activity;
        title = t;
        toolbar = tb;
        data = new ListAdapter(activity, k, v, w);
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
        int t = data.types.get(position);
        if (t == ListAdapter.TYPE_COMMAND || t == ListAdapter.TYPE_BUTTON) {
            main_activity.AppShellRun(data.vals.get(position));
        }
    }
}