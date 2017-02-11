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
    public int type, flags, left_icon, right_icon, tag=0, checkedId=0;
    public LCallback cb, right_cb;
    public LStringCB string_cb;
    public boolean hidden;
    public JPickerItem picker;
    public HashMap<String, ArrayList<JDependencyItem>> depends;

    JModelItem(final String k, final String v, final String rt, final String ddk,
               final int t, final int f, final int li, final int ri,
               final LCallback lcb, final LCallback rcb, final LStringCB scb, final boolean h,
               final JPickerItem p, final HashMap<String, ArrayList<JDependencyItem>> d) {
        key = k;
        val = v;
        right_text = rt;
        dropdown_key = ddk;
        type = t;
        flags = f;
        left_icon = li;
        right_icon = ri;
        cb = lcb;
        right_cb = rcb;
        string_cb = scb;
        hidden = h;
        picker = p;
        depends = d;
    }
}
