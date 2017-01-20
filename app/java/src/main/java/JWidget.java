package com.lucidfusionlabs.app;

import java.util.HashSet;
import android.util.Log;

public class JWidget {
    public String title = "";
    public HashSet<JWidget> parent;

    public JWidget(final MainActivity activity) {
        parent = activity.jwidgets.widgets;
        parent.add(this);
    }

    protected void finalize() throws Throwable {
        try { parent.remove(this); } finally { super.finalize(); }
    }

    public void clear() {}
    public void show(final MainActivity activity, final boolean show_or_hide) {}
}
