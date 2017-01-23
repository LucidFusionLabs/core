package com.lucidfusionlabs.app;

import android.util.Log;

public final class JModelItem {
    public static final int TYPE_NONE          = 0;
    public static final int TYPE_LABEL         = 1;
    public static final int TYPE_SEPARATOR     = 2;
    public static final int TYPE_COMMAND       = 3;
    public static final int TYPE_BUTTON        = 4;
    public static final int TYPE_TOGGLE        = 5;
    public static final int TYPE_SELECTOR      = 6;
    public static final int TYPE_PICKER        = 7;
    public static final int TYPE_TEXTINPUT     = 8;
    public static final int TYPE_NUMBERINPUT   = 9;
    public static final int TYPE_PASSWORDINPUT = 10;
    public static final int TYPE_FONTPICKER    = 11;
    public static final int TYPE_COUNT         = 12;

    public String key, val;
    public int type, left_icon, right_icon, tag=0;
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

    protected void finalize() throws Throwable { try { close(); } finally { super.finalize(); } }
}
