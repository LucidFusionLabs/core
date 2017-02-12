package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;

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

public class JWidgets {
    public boolean                destroyed;
    public HashSet<JWidget>       widgets        = new HashSet<JWidget>();
    public ArrayList<JNavigation> navigations    = new ArrayList<JNavigation>();
    public ArrayList<JToolbar>    toolbar_bottom = new ArrayList<JToolbar>();

    public void onDestroy() {
        Log.i("lfl", "JWidgets.onDestroy()");
				for (JWidget w : widgets) w.clear();
        destroyed = true;
    }

    public void onResume(MainActivity activity) {
        if (!destroyed) return;
        destroyed = false;
        Log.i("lfl", "JWidgets.onResume(Activity)");

        ArrayList<JToolbar> tb_bottom = new ArrayList<JToolbar>();
        tb_bottom.addAll(toolbar_bottom);
        toolbar_bottom.clear();
        for (JToolbar t : tb_bottom) t.show(activity, true);

        ArrayList<JNavigation> navs = new ArrayList<JNavigation>();
        navs.addAll(navigations);
        navigations.clear();
        for (JNavigation n : navs) n.show(activity, true);
    }
}
