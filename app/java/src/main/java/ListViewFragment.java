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
    public ArrayList<String> list, type, val;

    ListViewFragment(MainActivity activity,
                     final String t, final String[] k, final String[] v, final String[] w) {
        main_activity = activity;
        title = t;
        list = new ArrayList<String>();
        type = new ArrayList<String>();
        val  = new ArrayList<String>();
        for (int i = 0; i < k.length; i++) {
            list.add(k[i].equals("<separator>") ? "" : k[i]);
            type.add(v[i]);
            val.add(w[i]);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.listview_main, container, false);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>
            (getActivity(), R.layout.listview_cell, R.id.listview_cell_title, list);
        ListView listView = (ListView)view.findViewById(R.id.list);
        listView.setAdapter(adapter);
        listView.setOnItemClickListener(this);
        return view;
    }

    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        String t = type.get(position);
        if (t.equals("command") || t.equals("button")) {
            main_activity.AppShellRun(val.get(position));
        }
    }
}
