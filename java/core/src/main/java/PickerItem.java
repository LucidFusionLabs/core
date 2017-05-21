package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;

public final class PickerItem {
    public ArrayList<ArrayList<String>> data;
    public ArrayList<Integer> picked = new ArrayList<Integer>();
    public NativePickerItemCB cb;
    public long nativeParent;

    PickerItem(final ArrayList<ArrayList<String>> d, final NativePickerItemCB c, final long np) {
        data = d;
        cb = c;
        nativeParent = np;
    }
}
