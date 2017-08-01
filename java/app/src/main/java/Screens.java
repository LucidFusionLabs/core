package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.RelativeLayout;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.EditText;
import android.widget.TextView;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.app.AlertDialog;
import android.app.Activity;

public class Screens {
    public boolean                                 destroyed      = false;
    public int                                     next_screen_id = 1, fullscreen_height = 0;
    public HashMap<Integer, Screen>                screens        = new HashMap<Integer, Screen>();
    public ArrayList<ScreenFragmentNavigator>      navigators     = new ArrayList<ScreenFragmentNavigator>();
    public ArrayList<Toolbar>                      toolbar_bottom = new ArrayList<Toolbar>();
    public ViewTreeObserver.OnGlobalLayoutListener global_layout_listener;

    public void onDestroy() {
        NativeAPI.INFO("Screens.onDestroy()");
        for (Screen w : screens.values()) w.clear();
        for (Toolbar w : toolbar_bottom) w.clearView();
        destroyed = true;
    }

    public void onResume(MainActivity activity) {
        if (!destroyed) return;
        destroyed = false;
        NativeAPI.INFO("Screens.onResume(Activity) toolbars=" + toolbar_bottom.size());

        ArrayList<Toolbar> tb_bottom = new ArrayList<Toolbar>();
        tb_bottom.addAll(toolbar_bottom);
        toolbar_bottom.clear();
        for (Toolbar t : tb_bottom) t.show(activity, true);

        if (navigators.size() > 0) {
            activity.getSupportActionBar().show();
            navigators.get(navigators.size()-1).showBackFragment(activity, false, true);
            activity.onBackStackChanged();
        }
    }

    public void onStartRenderThread(Activity activity, View view) {
        if (navigators.size() > 0) {
            view.setVisibility(View.GONE);
        }
    }

    public void onFullScreenChanged(final MainActivity activity) {
        if (global_layout_listener != null) {
            activity.root_view.getViewTreeObserver().removeOnGlobalLayoutListener(global_layout_listener);
            global_layout_listener = null;
        }

        for (Toolbar t : toolbar_bottom) {
            if (t.view == null) continue;
            ViewParent parent = t.view.getParent();
            if (parent != null && parent instanceof ViewGroup) ((ViewGroup)parent).removeView(t.view);
            if (activity.fullscreen) activity.main_layout.addView(t.view, t.getFrameLayoutParams(activity));
            else                     activity.gl_layout  .addView(t.view, t.getLinearLayoutParams(activity));
        }

        if (activity.fullscreen) {
            global_layout_listener = new ViewTreeObserver.OnGlobalLayoutListener() { public void onGlobalLayout() {
                Rect r = new Rect();
                View view = activity.root_window.getDecorView();
                view.getWindowVisibleDisplayFrame(r);
                int h = r.bottom - r.top;
                NativeAPI.INFO("OnGlobalLayoutListener: " + r.toString() + " surface_height = " + activity.gl_view.surface_height);
                for (Toolbar toolbar : toolbar_bottom) {
                    View tb = toolbar.view;
                    if (tb == null || toolbar.shown_index == -1) continue;
                    FrameLayout.LayoutParams params = (FrameLayout.LayoutParams)tb.getLayoutParams();
                    if (params.bottomMargin != activity.gl_view.surface_height - h) {
                        params.bottomMargin = activity.gl_view.surface_height - h;
                        NativeAPI.INFO("set tb bottomMargin = " + (activity.gl_view.surface_height - h));
                        tb.setLayoutParams(params);
                    }
                    h -= tb.getHeight();
                }
                if (fullscreen_height != h) {
                    fullscreen_height = h;
                    NativeAPI.reshaped(r.left, Math.max(0, activity.gl_view.surface_height - h), r.right - r.left, h);
                }
            }};
            activity.root_view.getViewTreeObserver().addOnGlobalLayoutListener(global_layout_listener);
        }
    }
}
