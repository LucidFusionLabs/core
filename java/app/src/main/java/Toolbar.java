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
    public int shown_index = -1;
    public boolean borderless_buttons;

    public Toolbar(String theme, ArrayList<ModelItem> m, int flag) {
        model = m;
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
        LayoutInflater inflater = LayoutInflater.from(context);
        LinearLayout toolbar = (LinearLayout)inflater.inflate(R.layout.toolbar, null, false);
        View.OnClickListener listener = new View.OnClickListener() { public void onClick(View bt) {
            ModelItem r = model.get(bt.getId());
            if (r != null && r.cb != null) r.cb.run();
        }};
        
        for (int i = 0, l = model.size(); i != l; ++i) {
            ModelItem r = model.get(i);
            View bt = null;
            if (r.left_icon != 0) {
                ImageButton b = !borderless_buttons ? new ImageButton(context) :
                    (ImageButton)inflater.inflate(R.layout.borderless_imagebutton, toolbar, false);
                b.setImageResource(r.left_icon);
                bt = b;
            } else if (r.key.equals("\u2699")) {
                ImageButton b = !borderless_buttons ? new ImageButton(context) :
                    (ImageButton)inflater.inflate(R.layout.borderless_imagebutton, toolbar, false);
                b.setImageResource(android.R.drawable.ic_menu_preferences);
                bt = b;
            } else {
                Button b = !borderless_buttons ? new Button(context) :
                    (Button)inflater.inflate(R.layout.borderless_button, toolbar, false);
                b.setSingleLine(true);
                b.setText(r.key);
                bt = b;
            }
            bt.setId(i);
            bt.setTag(r.val);
            bt.setOnClickListener(listener);
            bt.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.0f));
            toolbar.addView(bt);
        }

        return toolbar;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        final Toolbar self = this;
        activity.runOnUiThread(new Runnable() { public void run() {
            View toolbar = getView(activity);
            if (show_or_hide) {
                if (shown_index >= 0) Log.e("lfl", "Show already shown toolbar");
                else {
                    activity.screens.toolbar_bottom.add(self);
                    activity.frame_layout.addView(toolbar, new LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,
                                                                            FrameLayout.LayoutParams.WRAP_CONTENT,
                                                                            Gravity.BOTTOM));
                    shown_index = activity.screens.toolbar_bottom.size()-1;
                }
            } else {
                if (shown_index < 0) Log.e("lfl", "Hide unshown toolbar");
                else {
                    int size = activity.screens.toolbar_bottom.size();
                    if (shown_index != size-1) Log.e("lfl", "Hide shown_index=" + shown_index + " size=" + size);
                    if (size > 0) activity.screens.toolbar_bottom.remove(size-1);
                    activity.frame_layout.removeView(toolbar);
                    shown_index = -1;
                }
            }
        }});
    }
}
