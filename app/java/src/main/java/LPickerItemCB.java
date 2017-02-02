package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class LPickerItemCB {
    public final long cb;

    LPickerItemCB(final long c) {
        cb = c;
        assert cb != 0;
    }

    protected void finalize() throws Throwable {
        try { FreePickerItemCB(cb); }
        finally { super.finalize(); }
    }

    public static native void RunPickerItemCBInMainThread(long cb, long picker, ArrayList<Integer> picked);
    public static native void FreePickerItemCB(long cb);
}
