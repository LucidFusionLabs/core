package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class LCallback {
    public final long cb;

    LCallback(final long c) {
        cb = c;
        assert cb != 0;
    }

    protected void finalize() throws Throwable {
        try { FreeCallback(cb); }
        finally { super.finalize(); }
    }

    public void run() { RunCallbackInMainThread(cb); }

    public static native void RunCallbackInMainThread(long cb);
    public static native void FreeCallback(long cb);
}
