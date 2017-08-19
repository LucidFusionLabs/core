package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeIntCB {
    public final long cb;

    NativeIntCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeIntCB(cb); }
        finally { super.finalize(); }
    }

    public void run(final int x) { run(x, null); }
    public void run(final int x, Runnable done_cb) { RunIntCBInMainThread(cb, x, done_cb); }

    public static native void RunIntCBInMainThread(long cb, int x, Runnable done_cb);
    public static native void FreeIntCB(long cb);
}
