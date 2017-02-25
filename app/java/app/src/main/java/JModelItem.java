package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

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
    public static final int TYPE_HIDDEN        = 12;
    public static final int TYPE_COUNT         = 13;

    public static final int HALIGN_LEFT   = 1;
    public static final int HALIGN_CENTER = 2;
    public static final int HALIGN_RIGHT  = 3;

    public static final int TABLE_FLAG_FIXDROPDOWN = 1;

    public String key, val, right_text, dropdown_key;
    public int type, tag, flags, left_icon, right_icon, selected, height;
    public LCallback cb;
    public LStringCB right_cb;
    public JPickerItem picker;
    public boolean hidden;

    JModelItem(final String k, final String v, final String rt, final String ddk,
               final int t, final int ta, final int f, final int li, final int ri, final int s, final int hi,
               final LCallback lcb, final LStringCB rcb, final JPickerItem p,
               final boolean h, final int fg, final int bg) {
        key = k;
        val = v;
        right_text = rt;
        dropdown_key = ddk;
        type = t;
        tag = ta;
        flags = f;
        left_icon = li;
        right_icon = ri;
        selected = s;
        height = hi;
        cb = lcb;
        right_cb = rcb;
        picker = p;
        hidden = h;
    }
}
