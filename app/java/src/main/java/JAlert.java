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

public class JAlert extends JWidget {
    public ArrayList<JModelItem> model;
    public Pair<AlertDialog, EditText> view;

    public JAlert(final MainActivity activity, ArrayList<JModelItem> m) {
        super(activity);
        model = m;
    }

    public void clear() { view = null; }

    public Pair<AlertDialog, EditText> get(final MainActivity activity) {
        if (view == null) {
            if (model.size() < 3) return null;
            AlertDialog.Builder alert = new AlertDialog.Builder(activity);
            alert.setTitle(model.get(1).key);
            alert.setMessage(model.get(1).val);
            
            final EditText input = model.get(0).val.equals("textinput") ? new EditText(activity) : null;
            if (input != null) {
                input.setInputType(android.text.InputType.TYPE_CLASS_TEXT);
                alert.setView(input);
            }
            
            alert.setPositiveButton(model.get(2).key, new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                JModelItem confirm = model.get(2);
                if (confirm.string_cb != 0) {
                    activity.AppRunStringCBInMainThread
                        (confirm.string_cb, (input == null) ? confirm.val : (confirm.val + " " + input.getText().toString()));
                }
            }});

            alert.setNegativeButton(model.get(3).key, new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                JModelItem cancel = model.get(3);
                if (cancel.string_cb != 0) {
                    activity.AppRunStringCBInMainThread
                        (cancel.string_cb, (input == null) ? cancel.val : (cancel.val + " " + input.getText().toString()));
                }
            }});

            view = new Pair<AlertDialog, EditText>(alert.create(), input);
        }
        return view;
    }
    
    public void show(final MainActivity activity, final boolean show_or_hide) {
        if (show_or_hide) { showText(activity, ""); }
    } 

    public void showText(final MainActivity activity, final String arg) {
        activity.runOnUiThread(new Runnable() { public void run() {
            Pair<AlertDialog, EditText> alert = get(activity);
            if (alert.second != null) { alert.second.setText(arg); }
            alert.first.show();
        }});
    }
}
