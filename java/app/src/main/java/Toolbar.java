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
import android.widget.ImageButton;
import android.widget.EditText;
import android.widget.TextView;
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
import android.app.Activity;
import android.app.AlertDialog;
import com.lucidfusionlabs.core.ModelItem;

public class Toolbar implements com.lucidfusionlabs.core.ViewOwner {
    public ArrayList<ModelItem> model;
    public View view;
    public String theme;
    public int shown_index = -1;
    public boolean borderless_buttons;

    public Toolbar(String t, ArrayList<ModelItem> m, int flag) {
        model = m;
        theme = t;
        borderless_buttons = (flag & ModelItem.TOOLBAR_FLAG_BORDERLESS_BUTTONS) != 0;
    }

    @Override public void clearView() {
        view = null;
        shown_index = -1;
    }

    @Override public View getView(final Context context) {
        if (view == null) view = createView(context);
        return view;
    }

    @Override public void onViewAttached() {}

    private View createView(final Context context) {
        boolean light = theme.equals("Light");
        LayoutInflater inflater = LayoutInflater.from(context);
        LinearLayout toolbar = (LinearLayout)inflater.inflate
            (light ? R.layout.toolbar_light : R.layout.toolbar_dark, null, false);
        int defStyleAttr = light ? android.support.v7.appcompat.R.style.Theme_AppCompat_Light :
            android.support.v7.appcompat.R.style.Theme_AppCompat;
        View.OnTouchListener listener = new View.OnTouchListener() {
            @Override public boolean onTouch(View bt, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) return true;
                if (event.getAction() != MotionEvent.ACTION_UP) return false;
                ModelItem r = model.get(bt.getId());
                if (r == null) return true;
                if (r.cb != null) r.cb.run();
                if (r.val.equals("toggle")) bt.setPressed(!bt.isPressed());
                return true;
            }};
        
        for (int i = 0, l = model.size(); i != l; ++i) {
            ModelItem r = model.get(i);
            View bt = null;
            if (r.left_icon != 0) {
                ImageButton b = !borderless_buttons ? new ImageButton(context) : (ImageButton)inflater.inflate
                    (light ? R.layout.borderless_imagebutton_light : R.layout.borderless_imagebutton_dark, toolbar, false);
                b.setImageResource(r.left_icon);
                bt = b;
            } else if (r.key.equals("\u2699")) {
                ImageButton b = !borderless_buttons ? new ImageButton(context) : (ImageButton)inflater.inflate
                    (light ? R.layout.borderless_imagebutton_light : R.layout.borderless_imagebutton_dark, toolbar, false);
                b.setImageResource(android.R.drawable.ic_menu_preferences);
                bt = b;
            } else {
                Button b = !borderless_buttons ? new Button(context) : (Button)inflater.inflate
                    (light ? R.layout.borderless_button_light : R.layout.borderless_button_dark, toolbar, false);
                b.setSingleLine(true);
                b.setText(r.key);
                bt = b;
            }
            bt.setId(i);
            bt.setTag("toolbar_button:" + r.key);
            bt.setOnTouchListener(listener);
            bt.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.0f));
            toolbar.addView(bt);
        }

        return toolbar;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            View toolbar = getView(activity);
            if (show_or_hide) {
                if (shown_index >= 0) Log.e("lfl", "Show already shown toolbar");
                else {
                    activity.screens.toolbar_bottom.add(Toolbar.this);
                    if (activity.fullscreen) activity.main_layout.addView(toolbar, getFrameLayoutParams (activity));
                    else                     activity.gl_layout  .addView(toolbar, getLinearLayoutParams(activity));
                    shown_index = activity.screens.toolbar_bottom.size()-1;
                }
            } else {
                if (shown_index < 0) Log.e("lfl", "Hide unshown toolbar");
                else {
                    activity.screens.toolbar_bottom.remove(Toolbar.this);
                    ViewParent parent = toolbar.getParent();
                    if (parent != null && parent instanceof ViewGroup) ((ViewGroup)parent).removeView(toolbar);
                    else Log.e("lfl", "Hide toolbar without parent shown_index=" + shown_index);
                    shown_index = -1;
                }
            }
        }});
    }

    public LinearLayout.LayoutParams getLinearLayoutParams(final MainActivity activity) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams
            (LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.gravity = Gravity.BOTTOM;
        return lp;
    }

    public FrameLayout.LayoutParams getFrameLayoutParams(final MainActivity activity) {
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams
            (FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT);
        lp.gravity = Gravity.BOTTOM;
        return lp;
    }

    public void setTheme(final MainActivity activity, final String name) {
        activity.runOnUiThread(new Runnable() { public void run() { theme = name; }});
    }

    public void toggleButton(final MainActivity activity, final String name) {
        activity.runOnUiThread(new Runnable() { public void run() {
            View toolbar = getView(activity);
            View button = (View)toolbar.findViewWithTag("toolbar_button:" + name);
            if (button != null) button.setPressed(!button.isPressed());
        }});
    }
}
