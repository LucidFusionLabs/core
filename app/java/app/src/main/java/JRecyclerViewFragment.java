package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.List;

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.DefaultItemAnimator;

public class JRecyclerViewFragment extends JFragment {
    public JRecyclerViewAdapter data;
    public View toolbar;
    public RecyclerView recyclerview = null;

    public JRecyclerViewFragment() {
       super(null, null);
       data = null;
       toolbar = null;
    }

    public JRecyclerViewFragment(final MainActivity activity, final JWidget p, final JRecyclerViewAdapter d, final View tb) {
        super(activity, p);
        data = d;
        toolbar = tb;
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
        recyclerview = (RecyclerView)layout.findViewById(R.id.list);
        recyclerview.setHasFixedSize(true);
        recyclerview.setAdapter(data);
        recyclerview.setLayoutManager(new LinearLayoutManager(main_activity));
        recyclerview.setItemAnimator(new DefaultItemAnimator());
        return layout;
    }
}
