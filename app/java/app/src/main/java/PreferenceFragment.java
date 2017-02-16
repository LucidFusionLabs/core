package com.lucidfusionlabs.app;

import android.os.*;

public class PreferenceFragment extends android.preference.PreferenceFragment {
    public int layout_id;

    public PreferenceFragment() { layout_id = 0; }
    public PreferenceFragment(int id) { layout_id = id; }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(layout_id);
    }
}
