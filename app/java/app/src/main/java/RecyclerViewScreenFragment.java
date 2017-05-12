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
import android.support.v7.widget.DividerItemDecoration;

public class RecyclerViewScreenFragment extends ScreenFragment {
    public ModelItemRecyclerViewAdapter data;
    public View toolbar;
    public RecyclerView recyclerview = null;

    public RecyclerViewScreenFragment() {
       super(null, null);
       data = null;
       toolbar = null;
    }

    public RecyclerViewScreenFragment(final MainActivity activity, final Screen p, final ModelItemRecyclerViewAdapter d, final View tb) {
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
        LinearLayoutManager layoutmanager = new LinearLayoutManager(main_activity);
        recyclerview = (RecyclerView)layout.findViewById(R.id.list);
        recyclerview.setHasFixedSize(true);
        recyclerview.setAdapter(data);
        recyclerview.setLayoutManager(layoutmanager);
        recyclerview.setItemAnimator(new DefaultItemAnimator());
        recyclerview.addItemDecoration
            (new DividerItemDecoration(recyclerview.getContext(), layoutmanager.getOrientation()));
        return layout;
    }
}
