package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class ModelItemChange {
    public interface ChangeWatcher {
        void onChanged(int row);
    }

    public int section, row, type, left_icon, right_icon, flags;
    public String key, val;
    public boolean hidden;
    public NativeCallback cb;

    public ModelItemChange(final int s, final int r, final int t, final String k,
                           final String v, final int li, final int ri, final int f,
                           final boolean h, final NativeCallback c) {
        section = s;
        row = r;
        type = t;
        key = k;
        val = v;
        left_icon = li;
        right_icon = ri;
        flags = f;
        hidden = h;
        cb = c;
    }

    public void apply(ModelItem item) {
        item.val    = val;
        item.flags  = flags;
        item.hidden = hidden;
        if (left_icon  != 0)                 item.left_icon  = left_icon  == -1 ? 0 : left_icon;
        if (right_icon != 0)                 item.right_icon = right_icon == -1 ? 0 : right_icon;
        if (key != null && key.length() > 1) item.key        = key;
        if (cb != null)                      item.cb         = cb;
        if (type != 0)                       item.type       = type;
    }

    public static void applyChangeList(ArrayList<ModelItemChange> changes, ModelItemList target, ChangeWatcher watcher) {
        for (ModelItemChange i : changes) {
            int row_id = target.getCollapsedRowId(i.section, i.row);
            if (row_id >= target.getData().size()) throw new java.lang.IllegalArgumentException();
            i.apply(target.getData().get(row_id));
            watcher.onChanged(row_id);
        }
    }
}
