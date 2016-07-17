package com.lucidfusionlabs.app;
import com.lucidfusionlabs.app.MainActivity;

import java.util.ArrayList;
import java.util.List;

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.util.Log;

public class ListViewFragment extends Fragment implements OnItemClickListener {
    public MainActivity main_activity;
    public String title;
    public ListAdapter data;
    public ListView listview;

    ListViewFragment(final MainActivity activity,
                     final String t, final String[] k, final String[] v, final String[] w) {
        main_activity = activity;
        title = t;
        data = new ListAdapter(activity, k, v, w);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.listview_main, container, false);
        listview = (ListView)view.findViewById(R.id.list);
        listview.setAdapter(data);
        listview.setOnItemClickListener(this);
        return view;
    }

    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        int t = data.types.get(position);
        if (t == ListAdapter.TYPE_COMMAND || t == ListAdapter.TYPE_BUTTON) {
            main_activity.AppShellRun(data.vals.get(position));
        }
    }
}
