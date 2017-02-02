package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;

public final class JPickerItem {
    public ArrayList<ArrayList<String>> data;
    public LPickerItemCB cb;
    public long lfl_self;

    JPickerItem(final ArrayList<ArrayList<String>> d, final LPickerItemCB c, final long self) {
        data = d;
        cb = c;
        lfl_self = self;
    }
}
