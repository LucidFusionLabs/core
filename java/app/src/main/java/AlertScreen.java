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
import com.lucidfusionlabs.core.NativeStringCB;

public class AlertScreen extends Screen {
    public ArrayList<ModelItem> model;
    public Pair<AlertDialog, EditText> view;

    public AlertScreen(ArrayList<ModelItem> m) {
        super(Screen.TYPE_ALERT, "", 0);
        model = m;
    }

    @Override
    public void clear() { view = null; }

    private Pair<AlertDialog, EditText> getView(final Activity activity) {
        if (view == null) view = createView(activity);
        return view;
    }

    private Pair<AlertDialog, EditText> createView(final Activity activity) {
        if (model.size() < 3) return null;
        AlertDialog.Builder alert = new AlertDialog.Builder(activity);
        alert.setTitle(model.get(1).key);
        alert.setMessage(model.get(1).val);

        ModelItem style = model.get(0);
        boolean pw = style.val.equals("pwinput");
        final EditText input = (pw || style.val.equals("textinput")) ? new EditText(activity) : null;
        if (input != null) {
            input.setInputType(android.text.InputType.TYPE_CLASS_TEXT |
                               (pw ? android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD : 0));
            alert.setView(input);
        }
        
        alert.setPositiveButton(model.get(3).key, new DialogInterface.OnClickListener() {
                                public void onClick(DialogInterface dialog, int which) {
            ModelItem confirm = model.get(3);
            if (confirm.right_cb != null)
                confirm.right_cb.run((input == null) ? confirm.val :
                                     (confirm.val + (confirm.val.length() > 0 ? " " : "") + input.getText().toString()));
        }});

        alert.setNegativeButton(model.get(2).key, new DialogInterface.OnClickListener() {
                                public void onClick(DialogInterface dialog, int which) {
            ModelItem cancel = model.get(2);
            if (cancel.right_cb != null)
                cancel.right_cb.run((input == null) ? cancel.val :
                                    (cancel.val + (cancel.val.length() > 0 ? " " : "") + input.getText().toString()));
        }});

        return new Pair<AlertDialog, EditText>(alert.create(), input);
    }
    
    @Override
    public void show(final MainActivity activity, final boolean show_or_hide) {
        if (show_or_hide) { showText(activity, ""); }
    } 

    public void showText(final Activity activity, final String arg) {
        activity.runOnUiThread(new Runnable() { public void run() {
            Pair<AlertDialog, EditText> alert = getView(activity);
            if (alert.second != null) { alert.second.setText(arg); }
            alert.first.show();
        }});
    }

    public void showTextCB(final Activity activity, final String tstr, final String msg,
                           final String arg, final NativeStringCB confirm_cb) {
        activity.runOnUiThread(new Runnable() { public void run() {
            ModelItem title = model.get(1), confirm = model.get(3);
            title.key = tstr;
            title.val = msg;
            confirm.right_cb = confirm_cb;
            clear();

            Pair<AlertDialog, EditText> alert = getView(activity);
            if (alert.second != null) { alert.second.setText(arg); }
            alert.first.show();
        }});
    }
}
