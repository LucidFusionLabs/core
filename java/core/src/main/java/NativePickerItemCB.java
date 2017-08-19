package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativePickerItemCB {
    public final long cb;

    NativePickerItemCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreePickerItemCB(cb); }
        finally { super.finalize(); }
    }

    public static native void RunPickerItemCBInMainThread(long cb, long picker, ArrayList<Integer> picked, Runnable done_cb);
    public static native void FreePickerItemCB(long cb);
}
