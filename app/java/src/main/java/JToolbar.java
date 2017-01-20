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
import android.app.ActionBar;
import android.app.AlertDialog;

public class JToolbar extends JWidget {
    public ArrayList<JModelItem> model;
    public View view;

    public JToolbar(final MainActivity activity, ArrayList<JModelItem> m) {
        super(activity);
        model = m;
    }

    public void clear() { view = null; }

    public View get(final MainActivity activity) {
        if (view == null) {
            LinearLayout toolbar = new LinearLayout(activity);
            View.OnClickListener listener = new View.OnClickListener() { public void onClick(View bt) {
                JModelItem r = model.get(bt.getId());
                if (r != null && r.cb != 0) activity.AppRunCallbackInMainThread(r.cb);
            }};
          
            for (int i = 0, l = model.size(); i != l; ++i) {
                JModelItem r = model.get(i);
                View bt = null;
                if (r.key.equals("\u2699")) {
                    ImageButton b = new ImageButton(activity);
                    b.setImageResource(android.R.drawable.ic_menu_preferences);
                    bt = b;
                } else {
                    Button b = new Button(activity);
                    b.setText(r.key);
                    bt = b;
                }
                bt.setId(i);
                bt.setTag(r.val);
                bt.setOnClickListener(listener);
                bt.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f)); 
                toolbar.addView(bt);
            }
            view = toolbar;
        }
        return view;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        final JToolbar self = this;
        activity.runOnUiThread(new Runnable() { public void run() {
            View toolbar = get(activity);
            activity.jwidgets.toolbar_bottom.add(self);
            activity.frame_layout.addView(toolbar, new LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,
                                                                    FrameLayout.LayoutParams.WRAP_CONTENT,
                                                                    Gravity.BOTTOM));
        }});
    }
}
