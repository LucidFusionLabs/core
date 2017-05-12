package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class ModelItem {
    public static final int TYPE_NONE          = 0;
    public static final int TYPE_LABEL         = 1;
    public static final int TYPE_SEPARATOR     = 2;
    public static final int TYPE_COMMAND       = 3;
    public static final int TYPE_BUTTON        = 4;
    public static final int TYPE_TOGGLE        = 5;
    public static final int TYPE_SELECTOR      = 6;
    public static final int TYPE_PICKER        = 7;
    public static final int TYPE_SLIDER        = 8;
    public static final int TYPE_TEXTINPUT     = 9;
    public static final int TYPE_NUMBERINPUT   = 10;
    public static final int TYPE_PASSWORDINPUT = 11;
    public static final int TYPE_FONTPICKER    = 12;
    public static final int TYPE_HIDDEN        = 13;
    public static final int TYPE_COUNT         = 14;

    public static final int HALIGN_LEFT   = 1;
    public static final int HALIGN_CENTER = 2;
    public static final int HALIGN_RIGHT  = 3;

    public static final int TABLE_FLAG_LEFTTEXT       = 1;
    public static final int TABLE_FLAG_SUBTEXT        = 2;
    public static final int TABLE_FLAG_FIXDROPDOWN    = 4;
    public static final int TABLE_FLAG_HIDEKEY        = 8;
    public static final int TABLE_FLAG_PLACEHOLDERVAL = 16;
    public static final int TABLE_FLAG_COLORSUBTEXT   = 32;
    public static final int TABLE_FLAG_COLORRIGHTTEXT = 64;

    public static final int TABLE_SECTION_FLAG_EDIT_BUTTON                 = 1;
    public static final int TABLE_SECTION_FLAG_EDITABLE_IF_HAS_TAG         = 2;
    public static final int TABLE_SECTION_FLAG_MOVABLE_ROWS                = 4;
    public static final int TABLE_SECTION_FLAG_DOUBLE_ROW_HEIGHT           = 8;
    public static final int TABLE_SECTION_FLAG_HIGHlIGHT_SELECTED_ROW      = 16;
    public static final int TABLE_SECTION_FLAG_DELETE_ROWS_WHEN_ALL_HIDDEN = 32;
    public static final int TABLE_SECTION_FLAG_CLEAR_LEFT_NAV_WHEN_EMPTY   = 64;

    public String key, val, right_text, dropdown_key;
    public int type, tag, flags, left_icon, right_icon, selected, height;
    public NativeCallback cb;
    public NativeStringCB right_cb;
    public PickerItem picker;
    public boolean hidden;

    ModelItem(final String k, final String v, final String rt, final String ddk,
              final int t, final int ta, final int f, final int li, final int ri,
              final int s, final int hi, final NativeCallback lcb, final NativeStringCB rcb,
              final PickerItem p, final boolean h, final int fg, final int bg) {
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