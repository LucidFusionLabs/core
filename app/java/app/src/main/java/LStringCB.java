package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class LStringCB {
    public final long cb;

    LStringCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeStringCB(cb); }
        finally { super.finalize(); }
    }

    public void run(final String s) { RunStringCBInMainThread(cb, s); }

    public static native void RunStringCBInMainThread(long cb, String text);
    public static native void FreeStringCB(long cb);
}
