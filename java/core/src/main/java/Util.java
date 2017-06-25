package com.lucidfusionlabs.core;

import java.util.List;
import java.util.ArrayList;
import java.util.RandomAccess;
import java.lang.Comparable;

public class Util {
    public static <T, L extends List<? extends Comparable<? super T>> & RandomAccess>
    int lowerBound(L arr, T target) {
      int low = 0, high = arr.size() - 1;
      while (low < high) {
          int mid = low + (high - low) / 2;
          if (arr.get(mid).compareTo(target) < 0) low  = mid + 1;
          else                                    high = mid;
      }
      return low;
    }

    public static <T, L extends List<? extends Comparable<? super T>> & RandomAccess>
    int lesserBound(L arr, T key) {
        int ind = lowerBound(arr, key);
        if (ind < 0) ind = arr.size();
        if (ind < arr.size() && arr.get(ind).compareTo(key) <= 0) return ind;
        return Math.max(0, ind - 1);
    }

    public static ArrayList<String> splitStringWords(final String x) {
        ArrayList<String> ret = new ArrayList<String>();
        if (x != null) ret.addAll(java.util.Arrays.asList(x.trim().split("\\s+")));
        return ret;
    }

}
