package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeIntIntCB {
    public final long cb;

    NativeIntIntCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeIntIntCB(cb); }
        finally { super.finalize(); }
    }

    public void run(int x, int y) { run(x, y, null); }
    public void run(int x, int y, Runnable done_cb) { RunIntIntCBInMainThread(cb, x, y, done_cb); }

    public static native void RunIntIntCBInMainThread(long cb, int x, int y, Runnable done_cb);
    public static native void FreeIntIntCB(long cb);
}
