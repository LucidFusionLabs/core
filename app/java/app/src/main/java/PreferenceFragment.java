package com.lucidfusionlabs.app;

import android.os.*;

public class PreferenceFragment extends android.support.v7.preference.PreferenceFragmentCompat {
    public int layout_id;

    public PreferenceFragment() { layout_id = 0; }
    public PreferenceFragment(int id) { layout_id = id; }

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(layout_id);
    }
}
