package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.List;

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.DefaultItemAnimator;
import android.support.v7.widget.DividerItemDecoration;
import com.lucidfusionlabs.core.LifecycleActivity;

public class RecyclerViewScreenFragment extends ScreenFragment {
    public RecyclerView recyclerview;
    public ModelItemRecyclerViewAdapter adapter;
    public int box_x, box_y, box_w, box_h;
    public static String PARENT_SCREEN_ID = "PARENT_SCREEN_ID";
    public static String BOX_X="BOX_X", BOX_Y="BOX_Y", BOX_W="BOX_W", BOX_H="BOX_H";

    public static final RecyclerViewScreenFragment newInstance(TableScreen parent, int x, int y, int w, int h) {
        RecyclerViewScreenFragment ret = new RecyclerViewScreenFragment();
        Bundle bundle = new Bundle(1);
        bundle.putInt(PARENT_SCREEN_ID, parent.screenId);
        bundle.putInt(BOX_X, x);
        bundle.putInt(BOX_Y, y);
        bundle.putInt(BOX_W, w);
        bundle.putInt(BOX_H, h);
        ret.setArguments(bundle);
        ret.adapter = parent.model;
        ret.parent_screen = parent;
        ret.box_x = x;
        ret.box_y = y;
        ret.box_w = w;
        ret.box_h = h;
        return ret;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (MainActivity.screens.next_screen_id == 1) return;
        int parent_screen_id = getArguments().getInt(PARENT_SCREEN_ID);
        if (parent_screen_id < MainActivity.screens.screens.size()) {
            if ((parent_screen = MainActivity.screens.screens.get(parent_screen_id)) instanceof TableScreen)
                adapter = ((TableScreen)parent_screen).model;
        } else NativeAPI.ERROR("invalid screen_id=" + parent_screen_id + " total=" + MainActivity.screens.screens.size());
        box_x = getArguments().getInt(BOX_X);
        box_y = getArguments().getInt(BOX_Y);
        box_w = getArguments().getInt(BOX_W);
        box_h = getArguments().getInt(BOX_H);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        LinearLayout layout = (LinearLayout)inflater.inflate(R.layout.listview_main, container, false);
        if (box_w > 0 && box_h > 0) {
            ViewGroup.MarginLayoutParams marginparams = new ViewGroup.MarginLayoutParams(box_w, box_h);
            marginparams.setMargins(box_x, container.getHeight() - box_y - box_h,
                                    box_y, container.getWidth()  - box_x - box_w);
            layout.setLayoutParams(new FrameLayout.LayoutParams(marginparams));
        }
        if (adapter.toolbar != null) {
            View toolbar = adapter.toolbar.getView(getContext());
            LifecycleActivity.replaceViewParent(toolbar, null);
            layout.addView(toolbar,
                           new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                                                         LinearLayout.LayoutParams.WRAP_CONTENT));
            adapter.toolbar.onViewAttached();
        }
        if (adapter.advertising != null) {
            View ads = adapter.advertising.getView(getContext());
            LifecycleActivity.replaceViewParent(ads, null);
            layout.addView(ads,
                           new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                                                         LinearLayout.LayoutParams.WRAP_CONTENT));
            adapter.advertising.onViewAttached();
        }
        LinearLayoutManager layoutmanager = new LinearLayoutManager(getContext());
        DefaultItemAnimator animator = new DefaultItemAnimator();
        animator.setSupportsChangeAnimations(false);
        recyclerview = (RecyclerView)layout.findViewById(R.id.list);
        recyclerview.setHasFixedSize(true);
        recyclerview.setAdapter(adapter);
        recyclerview.setLayoutManager(layoutmanager);
        recyclerview.setItemAnimator(animator);
        recyclerview.addItemDecoration
            (new DividerItemDecoration(recyclerview.getContext(), layoutmanager.getOrientation()));
        return layout;
    }
}
