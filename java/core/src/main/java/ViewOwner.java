package com.lucidfusionlabs.core;

import android.content.Context;
import android.view.View;

public interface ViewOwner {
    public void clearView();
    public void onViewAttached();
    public View getView(final Context context);
}
