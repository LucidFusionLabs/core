package com.lucidfusionlabs.app;

import android.util.Log;

public final class JModelItem {
    public String key, val;
    public int type, left_icon, right_icon;
    public long cb, right_cb, string_cb;
    public boolean hidden;
    public native void close();

    JModelItem(final String k, final String v, final int t, final int li, final int ri,
                final long lcb, final long rcb, final long scb, final boolean h) {
        key = k;
        val = v;
        type = t;
        left_icon = li;
        right_icon = ri;
        cb = lcb;
        right_cb = rcb;
        string_cb = scb;
        hidden = h;
    }
}
