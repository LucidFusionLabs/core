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
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.app.AlertDialog;
import android.support.v4.app.Fragment;

public class TextScreen extends Screen {
    public String text;

    public TextScreen(String t, String v) {
        super(Screen.TYPE_TEXTVIEW, t, 0);
        text = v;
    }

    @Override
    public ScreenFragment createFragment() {
        return TextViewScreenFragment.newInstance(this);
    }

    @Override
    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, createFragment()).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                Fragment frag = activity.getSupportFragmentManager().findFragmentById(R.id.content_frame);
                activity.getSupportFragmentManager().beginTransaction().remove(frag).commit();
            }
        }});
    }
}
