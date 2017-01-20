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

public class JMenu extends JWidget {
    public ArrayList<JModelItem> model;
    public ListViewFragment view;

    public JMenu(final MainActivity activity, String t, ArrayList<JModelItem> m) {
        super(activity);
        title = t;
        model = m;
    }

    public void clear() { view = null; }

    public ListViewFragment get(final MainActivity activity) {
        if (view == null) {
            view = new ListViewFragment(activity, title, new ListAdapter(activity, model), null);
        }
        return view;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
        }});
    }
}
