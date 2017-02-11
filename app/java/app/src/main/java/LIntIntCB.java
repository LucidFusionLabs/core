package com.lucidfusionlabs.app;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public final class LIntIntCB {
    public final long cb;

    LIntIntCB(final long c) {
        cb = c;
        assert cb != 0;
    }

    protected void finalize() throws Throwable {
        try { FreeIntIntCB(cb); }
        finally { super.finalize(); }
    }

    public static native void RunIntIntCBInMainThread(long cb, int x, int y);
    public static native void FreeIntIntCB(long cb);
}
