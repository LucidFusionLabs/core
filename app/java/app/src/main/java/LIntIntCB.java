package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class LIntIntCB {
    public final long cb;

    LIntIntCB(final long c) {
        cb = c;
        if (cb == 0) throw new java.lang.IllegalArgumentException();
    }

    protected void finalize() throws Throwable {
        try { FreeIntIntCB(cb); }
        finally { super.finalize(); }
    }

    public static native void RunIntIntCBInMainThread(long cb, int x, int y);
    public static native void FreeIntIntCB(long cb);
}
