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

public class AppWidgets {
    class Spec {
        String title;
        String[] k, v, w;
        Spec(String[] K, String[] V) {
            k = K;
            v = V;
        }
        Spec(String T, String[] K, String[] V, String[] W) {
            title = T;
            k = K;
            v = V;
            w = W;
        }
    }

    public boolean destroyed;

    public ArrayList<Spec> toolbar_specs = new ArrayList<Spec>();
    public ArrayList<Spec> table_specs = new ArrayList<Spec>();
    public ArrayList<Spec> alert_specs = new ArrayList<Spec>();
    public ArrayList<Spec> navigation_specs = new ArrayList<Spec>();
    public ArrayList<ArrayList<Integer>> table_toolbars = new ArrayList<ArrayList<Integer>>();

    public ArrayList<Integer> toolbar_top = new ArrayList<Integer>();
    public ArrayList<Integer> toolbar_bottom = new ArrayList<Integer>();
    public ArrayList<Integer> navigation_stack = new ArrayList<Integer>();

    public ArrayList<View> toolbars = new ArrayList<View>();
    public ArrayList<ListViewFragment> tables = new ArrayList<ListViewFragment>();
    public ArrayList<Pair<AlertDialog, EditText>> alerts = new ArrayList<Pair<AlertDialog, EditText>>();
    public ArrayList<Integer> navigations = new ArrayList<Integer>();

    void onDestroy() {
        Log.i("lfl", "AppWidgets.onDestroy()");
        for (int i = 0; i < toolbars   .size(); i++) toolbars   .set(i, null);
        for (int i = 0; i < tables     .size(); i++) tables     .set(i, null);
        for (int i = 0; i < alerts     .size(); i++) alerts     .set(i, null);
        destroyed = true;
    }

    void onResume(MainActivity activity) {
        if (!destroyed) return;
        destroyed = false;
        Log.i("lfl", "AppWidgets.onResume(Activity)");

        ArrayList<Integer> tb_bottom = new ArrayList<Integer>();
        tb_bottom.addAll(toolbar_bottom);
        toolbar_bottom.clear();
        for (Integer id : tb_bottom) activity.showToolbar(id);

        ArrayList<Integer> nav_stack = new ArrayList<Integer>();
        nav_stack.addAll(navigation_stack);
        navigation_stack.clear();
        for (Integer id : nav_stack) activity.showNavigation(id, true);
    }

    Pair<AlertDialog, EditText> getAlert(final MainActivity activity, int id) {
        if (alerts.get(id) == null) {
            final Spec spec = alert_specs.get(id);
            AlertDialog.Builder alert = new AlertDialog.Builder(activity);
            alert.setTitle(spec.k[1]);
            alert.setMessage(spec.v[1]);
            
            final EditText input = spec.v[0].equals("textinput") ? new EditText(activity) : null;
            if (input != null) {
                input.setInputType(android.text.InputType.TYPE_CLASS_TEXT);
                alert.setView(input);
            }
            
            alert.setPositiveButton(spec.k[2], new DialogInterface.OnClickListener() {
                                               public void onClick(DialogInterface dialog, int which) {
                activity.AppShellRun((input == null) ? spec.v[2] : (spec.v[2] + " " + input.getText().toString()));
            }});
            alert.setNegativeButton(spec.k[3], new DialogInterface.OnClickListener() {
                                               public void onClick(DialogInterface dialog, int which) {
                activity.AppShellRun((input == null) ? spec.v[3] : (spec.v[3] + " " + input.getText().toString()));
            }});
            
            alerts.set(id, new Pair<AlertDialog, EditText>(alert.create(), input));
        }
        return alerts.get(id);
    }

    View getToolbar(final MainActivity activity, int id) {
        if (toolbars.get(id) == null) {
            Spec spec = toolbar_specs.get(id);
            LinearLayout toolbar = new LinearLayout(activity);
            View.OnClickListener listener = new View.OnClickListener() { public void onClick(View bt) {
                activity.AppShellRun((String)bt.getTag());
            }};
            
            for (int i = 0; i < spec.k.length; i++) {
                View bt = null;
                if (spec.k[i].equals("\u2699")) {
                  ImageButton b = new ImageButton(activity);
                  b.setImageResource(android.R.drawable.ic_menu_preferences);
                  bt = b;
                } else {
                  Button b = new Button(activity);
                  b.setText(spec.k[i]);
                  bt = b;
                }
                bt.setId(i);
                bt.setTag(spec.v[i]);
                bt.setOnClickListener(listener);
                bt.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f)); 
                toolbar.addView(bt);
            }
            toolbars.set(id, toolbar);
        }
        return toolbars.get(id);
    }

    ListViewFragment getTable(final MainActivity activity, int id) {
        if (tables.get(id) == null) {
            Spec spec = table_specs.get(id);
            ArrayList<Integer> toolbars = table_toolbars.get(id);
            ListViewFragment frag = new ListViewFragment(activity, spec.title, spec.k, spec.v, spec.w,
                                                         toolbars.size() > 0 ? getToolbar(activity, toolbars.get(0)) : null);
            tables.set(id, frag);
        }
        return tables.get(id);
    }
}
