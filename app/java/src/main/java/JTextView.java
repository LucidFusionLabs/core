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

public class JTextView extends JWidget {
    public String text;
    public JTextViewFragment view;
    public long lfl_self = 0;

    public JTextView(final MainActivity activity, String t, String v) {
        super(JWidget.TYPE_TEXTVIEW, activity, "");
        title = t;
        text = v;
    }

    public void clear() { view = null; }

    public JTextViewFragment get(final MainActivity activity) {
        if (view == null) {
            view = new JTextViewFragment(activity, this, text, lfl_self);
        }
        return view;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            JTextViewFragment frag = get(activity);
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getFragmentManager().beginTransaction().replace(R.id.content_frame, frag).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                activity.getFragmentManager().beginTransaction().remove(frag).commit();
            }
        }});
    }
}
