package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;

public final class PickerItem {
    public ArrayList<ArrayList<String>> data;
    public ArrayList<Integer> picked = new ArrayList<Integer>();
    public NativePickerItemCB cb;
    public long nativeParent;

    public PickerItem(final ArrayList<ArrayList<String>> d, final NativePickerItemCB c, final long np) {
        data = d;
        cb = c;
        nativeParent = np;
        for (int i=0, l=data.size(); i < l; i++) picked.add(0);
    }

    public String getPicked(int i) {
        return data.get(i).get(picked.get(i));
    }

    public String getPickedString() {
        String v = "";
        for (int i=0, l=data.size(); i < l; i++) v += (i != 0 ? " " : "") + getPicked(i);
        return v;
    }

    public static native PickerItem getFontPickerItem();
}
