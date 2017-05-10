package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeCallback {
    public final long cb;

    NativeCallback(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeCallback(cb); }
        finally { super.finalize(); }
    }

    public void run() { RunCallbackInMainThread(cb); }

    public static native void RunCallbackInMainThread(long cb);
    public static native void FreeCallback(long cb);
}
