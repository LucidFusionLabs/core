package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeCallback implements Runnable {
    public final long cb;

    NativeCallback(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeCallback(cb); }
        finally { super.finalize(); }
    }

    @Override public void run() { run(null); }
    public void run(Runnable done_cb) { RunCallbackInMainThread(cb, done_cb); }

    public static native void RunCallbackInMainThread(long cb, Runnable done_cb);
    public static native void FreeCallback(long cb);
}
