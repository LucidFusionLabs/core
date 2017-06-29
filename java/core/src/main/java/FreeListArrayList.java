package com.lucidfusionlabs.core;

import android.util.Log;
import java.util.ArrayList;

public final class FreeListArrayList<T> {
    public ArrayList<T>       data = new ArrayList<T>();
    public ArrayList<Integer> free = new ArrayList<Integer>();

    public int size() { return data.size(); }

    public void clear() {
        data.clear();
        free.clear();
    }

    public int add(T x) {
        if (free.size() == 0) {
            data.add(x);
            return data.size() - 1;
        } else {
            int free_data_ind = free.get(free.size() - 1);
            free.remove(free.size() - 1);
            data.set(free_data_ind, x);
            return free_data_ind;
        }
    }

    public T remove(int ind) {
        T ret = data.get(ind);
        data.set(ind, null);
        free.add(ind);
        return ret;
    }

    public T get(int ind)      { return data.get(ind); }
    public T set(int ind, T x) { return data.set(ind, x); }
}
