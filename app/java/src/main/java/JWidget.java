package com.lucidfusionlabs.app;

import java.util.HashSet;
import android.util.Log;

public class JWidget {
    public static final int TYPE_NONE       = 0;
    public static final int TYPE_ALERT      = 1;
    public static final int TYPE_TOOLBAR    = 2;
    public static final int TYPE_MENU       = 3;
    public static final int TYPE_TABLE      = 4;
    public static final int TYPE_TEXTVIEW   = 5;
    public static final int TYPE_NAVIGATION = 6;
    public static final int TYPE_COUNT      = 7;

    public int widgetType;
    public String title = "";
    public HashSet<JWidget> parent;

    public JWidget(int wt, final MainActivity activity, String t) {
        widgetType = wt;
        title = t;
        parent = activity.jwidgets.widgets;
        parent.add(this);
    }

    protected void finalize() throws Throwable {
        try { parent.remove(this); } finally { super.finalize(); }
    }

    public void clear() {}
    public void show(final MainActivity activity, final boolean show_or_hide) {}
}
