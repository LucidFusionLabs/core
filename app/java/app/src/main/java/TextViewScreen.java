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
import android.app.AlertDialog;

public class TextViewScreen extends Screen {
    public String text;
    public TextViewScreenFragment view;

    public TextViewScreen(final MainActivity activity, String t, String v) {
        super(Screen.TYPE_TEXTVIEW, activity, "", 0);
        title = t;
        text = v;
    }

    public void clear() { view = null; }
    public ScreenFragment get(final MainActivity activity) { return getView(activity); }

    public TextViewScreenFragment getView(final MainActivity activity) {
        if (view == null) view = createView(activity);
        return view;
    }

    private TextViewScreenFragment createView(final MainActivity activity) {
        return new TextViewScreenFragment(activity, this, text);
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            TextViewScreenFragment frag = getView(activity);
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, frag).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                activity.getSupportFragmentManager().beginTransaction().remove(frag).commit();
            }
        }});
    }
}
