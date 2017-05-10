package com.lucidfusionlabs.app;

import java.util.HashSet;
import android.util.Log;

public class Screen {
    public static final int TYPE_NONE       = 0;
    public static final int TYPE_ALERT      = 1;
    public static final int TYPE_MENU       = 2;
    public static final int TYPE_TABLE      = 3;
    public static final int TYPE_TEXTVIEW   = 4;
    public static final int TYPE_COUNT      = 5;

    public int screenType;
    public long nativeParent;
    public String title = "";
    public boolean changed = false;
    public HashSet<Screen> parent;

    public Screen(final int wt, final MainActivity activity, final String t, final long np) {
        screenType = wt;
        title = t;
        nativeParent = np;
        parent = activity.screens.widgets;
        parent.add(this);
    }

    protected void finalize() throws Throwable {
        try { parent.remove(this); }
        finally { super.finalize(); }
    }

    public void clear() {}
    public void show(final MainActivity activity, final boolean show_or_hide) {}
}
