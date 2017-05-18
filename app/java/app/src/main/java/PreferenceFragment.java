package com.lucidfusionlabs.app;

import android.os.*;

public class PreferenceFragment extends android.support.v7.preference.PreferenceFragmentCompat {
    public static String LAYOUT_ID = "LAYOUT_ID";
    public int layout_id;
    
    public static final PreferenceFragment newInstance(int lid) {
        PreferenceFragment ret = new PreferenceFragment();
        Bundle bundle = new Bundle(1);
        bundle.putInt(LAYOUT_ID, lid);
        ret.setArguments(bundle);
        ret.layout_id = lid;
        return ret;
    }

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        super.onCreate(savedInstanceState);
        layout_id = getArguments().getInt(LAYOUT_ID);
        addPreferencesFromResource(layout_id);
    }
}
