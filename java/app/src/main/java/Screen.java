package com.lucidfusionlabs.app;

import java.util.HashMap;
import java.util.ArrayList;

public class Screen {
    public static final int TYPE_NONE       = 0;
    public static final int TYPE_ALERT      = 1;
    public static final int TYPE_MENU       = 2;
    public static final int TYPE_TABLE      = 3;
    public static final int TYPE_TEXTVIEW   = 4;
    public static final int TYPE_COUNT      = 5;

    public int screenId, screenType;
    public long nativeParent;
    public String title = "";
    public boolean changed = false;

    public Screen(final int st, final String t, final long np) {
        screenId = MainActivity.screens.next_screen_id++;
        screenType = st;
        title = t;
        nativeParent = np;
        MainActivity.screens.screens.put(screenId, this);
    }

    protected void finalize() throws Throwable {
        try { MainActivity.screens.screens.remove(screenId); }
        finally { super.finalize(); }
    }

    public void clear() {}
    public ScreenFragment createFragment() { return null; }
    public void show(final MainActivity activity, final boolean show_or_hide) {}
}
