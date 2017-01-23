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

public class JNavigation extends JWidget {
    public ArrayList<JWidget> stack = new ArrayList<JWidget>();

    public JNavigation(final MainActivity activity) {
        super(activity);
    }

    public void clear() {}

    public ListViewFragment get(final MainActivity activity) {
        return null;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        final JNavigation self = this;
        activity.runOnUiThread(new Runnable() { public void run() {
           if (show_or_hide) activity.jwidgets.navigations.add(self);
           else {
               int size = activity.jwidgets.navigations.size();
               assert size != 0 && activity.jwidgets.navigations.get(size-1) == self;
               activity.jwidgets.navigations.remove(size-1);
           }
           if (stack.size() != 0) stack.get(stack.size()-1).show(activity, show_or_hide);
        }});
    }

    public void pushTable(final MainActivity activity, final JTable x) {
        activity.runOnUiThread(new Runnable() { public void run() {
            stack.add(x);
            ListViewFragment table = x.get(activity);
            activity.getFragmentManager().beginTransaction()
                .replace(R.id.content_frame, table).addToBackStack(table.title).commit();
        }});
    }
    
    public void popView(final MainActivity activity) {
        activity.runOnUiThread(new Runnable() { public void run() {
            stack.remove(stack.size()-1);
            activity.getFragmentManager().popBackStackImmediate(null, android.app.FragmentManager.POP_BACK_STACK_INCLUSIVE);
        }});
    }
}
