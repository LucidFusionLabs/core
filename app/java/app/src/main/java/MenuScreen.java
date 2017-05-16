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

public class MenuScreen extends Screen {
    public ArrayList<ModelItem> model;
    public AlertDialog view;

    public MenuScreen(String t, ArrayList<ModelItem> m) {
        super(Screen.TYPE_MENU, t, 0);
        model = m;
    }

    @Override
    public void clear() { view = null; }

    public AlertDialog getView(final MainActivity activity) {
        if (view == null) {
            AlertDialog.Builder builder = new AlertDialog.Builder(activity);
            builder.setTitle(title);
            if (model.size() > 0) { 
                String[] items = new String[model.size()];
                for (int i = 0; i < items.length; i++) items[i] = model.get(i).val;
                builder.setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (which >= model.size()) throw new java.lang.IllegalArgumentException();
                        ModelItem item = model.get(which);
                        if (item.cb != null) item.cb.run();
                    }});
            }
            view = builder.create();
        }
        return view;
    }

    @Override
    public void show(final MainActivity activity, final boolean show_or_hide) {
        if (!show_or_hide) return;
        activity.runOnUiThread(new Runnable() { public void run() {
            AlertDialog alert = getView(activity);
            alert.show();
        }});
    }
}
