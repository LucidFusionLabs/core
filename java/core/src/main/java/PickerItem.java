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

    public int updatePicked(int i, String[] arr, String v) {
        for (int j = 0; j < arr.length; j++)
            if (v.equals(arr[j])) {
                picked.set(i, j);
                return j;
            }
        return -1;
    }

    public String getPicked(int i) {
        return data.get(i).get(picked.get(i));
    }

    public String getPickedString() {
        String v = "";
        for (int i=0, l=data.size(); i < l; i++) v += (i != 0 ? " " : "") + getPicked(i);
        return v;
    }

    public String[] getData(int i) {
        ArrayList<String> picker_items = data.get(i);
        if (picker_items.size() == 0) return null;
        String[] arr = new String[picker_items.size()];
        return picker_items.toArray(arr);
    }

    public static native PickerItem getFontPickerItem();
}
