/*
 * $Id: nlp.h 1309 2014-10-10 19:20:55Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __LFL_NLP_LM_H__
#define __LFL_NLP_LM_H__
namespace LFL {

/* LanguageModel */
struct LanguageModel {
    Matrix *prior, *transit, *map;
    vector<string> *names;
    double total;

    ~LanguageModel() { reset(); }
    LanguageModel() : prior(0), transit(0), map(0), total(0) {}
    void reset() { delete prior; prior=0; delete transit; transit=0; delete map; map=0; delete names; names=0; total=0; }

    static const int map_buckets=5, map_values=3, col_prior=1, col_transit=2;
    double *getHashEntry(unsigned hash) { return HashMatrix::Get(map, hash, map_values); }

    bool get(unsigned hash, int *priorInd, int *transitInd=0) {
        double *he = getHashEntry(hash);
        if (!he) return 0;
        if (priorInd) *priorInd = he[col_prior];
        if (transitInd) *transitInd = he[col_transit];
        return 1;
    }
    bool get(const char *word, int *priorInd, int *transitInd=0) { return get(fnv32(word), priorInd, transitInd); }

    int read(const char *name, const char *dir) {
        reset();

        string flags;
        int lastiter = MatrixFile::ReadVersioned(dir, name, "transition", &transit, &flags);
        if (!transit) { ERROR("no language model: ", name); return -1; }
        if (flags.size()) DEBUG("loading ", name, " ", lastiter, " : ", flags);

        if (MatrixFile::ReadVersioned(dir, name, "prior", &prior, 0, lastiter) < 0) { ERROR(name, ".", lastiter, ".prior"); return -1; }
        if (MatrixFile::ReadVersioned(dir, name, "map",   &map,   0, lastiter) < 0) { ERROR(name, ".", lastiter, ".map"  ); return -1; }
        if (StringFile::ReadVersioned(dir, name, "name",  &names, 0, lastiter) < 0) { ERROR(name, ".", lastiter, ".name" ); return -1; }

        MatrixRowIter(prior) total += prior->row(i)[0];
        return lastiter;
    }

    int transits(int transitInd, unsigned hash) {
        int transits = 0;
        while (transitInd + transits < transit->M && transit->row(transitInd + transits)[TC_Self] == hash) transits++;
        return transits;
    }

    int occurences(int priorInd) { return prior->row(priorInd)[0]; }
    const char *name(int priorInd) { return (*names)[priorInd].c_str(); }
};

}; // namespace LFL
#endif // __LFL_NLP_LM_H__
