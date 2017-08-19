package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class NativeStringCB {
    public final long cb;

    NativeStringCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeStringCB(cb); }
        finally { super.finalize(); }
    }

    public void run(final String s) { run(s, null); }
    public void run(final String s, Runnable done_cb) { RunStringCBInMainThread(cb, s, done_cb); }

    public static native void RunStringCBInMainThread(long cb, String text, Runnable done_cb);
    public static native void FreeStringCB(long cb);
}
