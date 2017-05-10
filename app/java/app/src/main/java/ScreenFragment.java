package com.lucidfusionlabs.app;

import android.util.Log;
import android.support.v4.app.Fragment;
import java.util.ArrayList;
import java.util.HashMap;

public class ScreenFragment extends Fragment {
    public MainActivity main_activity;
    public Screen parent_screen;

    ScreenFragment(final MainActivity activity, final Screen p) {
        main_activity = activity;
        parent_screen = p;
    }
}
