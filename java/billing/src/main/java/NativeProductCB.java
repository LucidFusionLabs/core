package com.lucidfusionlabs.billing;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeProductCB {
    public final long cb;

    NativeProductCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeProductCB(cb); }
        finally { super.finalize(); }
    }

    public void run(final String id, final String name, final String desc, final String price) {
        RunProductCBInMainThread(cb, id, name, desc, price);
    }

    public static native void RunProductCBInMainThread(long cb, String id, String name, String desc, String price);
    public static native void FreeProductCB(long cb);
}
