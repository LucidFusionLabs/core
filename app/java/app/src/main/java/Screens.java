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
import android.app.AlertDialog;

public class Screens {
    public boolean                            destroyed      = false;
    public int                                next_screen_id = 1;
    public HashMap<Integer, Screen>           screens        = new HashMap<Integer, Screen>();
    public ArrayList<ScreenFragmentNavigator> navigators     = new ArrayList<ScreenFragmentNavigator>();
    public ArrayList<Toolbar>                 toolbar_bottom = new ArrayList<Toolbar>();

    public void onDestroy() {
        Log.i("lfl", "Screens.onDestroy()");
        for (Screen w : screens.values()) w.clear();
        for (Toolbar w : toolbar_bottom) w.clear();
        destroyed = true;
    }

    public void onResume(MainActivity activity) {
        if (!destroyed) return;
        destroyed = false;
        Log.i("lfl", "Screens.onResume(Activity) toolbars=" + toolbar_bottom.size());

        ArrayList<Toolbar> tb_bottom = new ArrayList<Toolbar>();
        tb_bottom.addAll(toolbar_bottom);
        toolbar_bottom.clear();
        for (Toolbar t : tb_bottom) t.show(activity, true);
    }
}
