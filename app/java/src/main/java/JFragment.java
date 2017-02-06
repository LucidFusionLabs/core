package com.lucidfusionlabs.app;

import android.util.Log;
import android.app.Fragment;
import java.util.ArrayList;
import java.util.HashMap;

public class JFragment extends Fragment {
    public MainActivity main_activity;
    public JWidget parent_widget;

    JFragment(final MainActivity activity, final JWidget p) {
        main_activity = activity;
        parent_widget = p;
    }
}
