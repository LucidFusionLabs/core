/*
 * $Id$
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

struct LanguageModel {
    static const int map_values=3, col_prior=1, col_transit=2;
    HashMatrix map;
    Matrix *prior=0, *transit=0;
    vector<string> *names=0;
    double total=0;

    ~LanguageModel() { Reset(); }
    LanguageModel() : map(0, map_values) {}

    void Reset() { delete map.map; delete prior; delete transit; delete names; map.map=0; prior=0; transit=0; names=0; total=0; }
    int Open(const char *name, const char *dir) {
        string flags;
        int lastiter = MatrixFile::ReadVersioned(dir, name, "transition", &transit, &flags);
        if (!transit) { ERROR("no language model: ", name); return -1; }
        if (MatrixFile::ReadVersioned(dir, name, "prior", &prior,   0, lastiter) < 0) { ERROR(name, ".", lastiter, ".prior"); return -1; }
        if (MatrixFile::ReadVersioned(dir, name, "map",   &map.map, 0, lastiter) < 0) { ERROR(name, ".", lastiter, ".map"  ); return -1; }
        if (StringFile::ReadVersioned(dir, name, "name",  &names,   0, lastiter) < 0) { ERROR(name, ".", lastiter, ".name" ); return -1; }
        MatrixRowIter(prior) total += prior->row(i)[0];
        return lastiter;
    }

    string     Name(int prior_ind) const { return (*names)[prior_ind]; }
    int Occurrences(int prior_ind) const { return prior->row(prior_ind)[0]; }
    int Transits(int transit_ind, unsigned hash) const {
        int n = 0;
        while (transit_ind + n < transit->M && transit->row(transit_ind + n)[TC_Self] == hash) n++;
        return n;
    }

    bool Get(const char *word, int *prior_ind_out, int *transit_ind_out=0) const {
        return Get(fnv32(word), prior_ind_out, transit_ind_out);
    }
    bool Get(unsigned hash, int *prior_ind_out, int *transit_ind_out=0) const {
        const double *he = map.Get(hash); 
        if (!he) return 0;
        if (prior_ind_out)   *prior_ind_out   = he[col_prior];
        if (transit_ind_out) *transit_ind_out = he[col_transit];
        return 1;
    }

    string DebugString(const char *word) {
        int prior_ind, transit_ind;
        if (!Get(fnv32(word), &prior_ind, &transit_ind)) return "";
        return StrCat(word, " ", DebugString(fnv32(word), prior_ind, transit_ind));
    }
    string DebugString(int hash, int prior_ind, int transit_ind) {
        int occurrences = Occurrences(prior_ind), transits = Transits(transit_ind, hash);
        string ret = StrCat(" : count=", occurrences, ", transits=", transits, " : {");
        for (int i=0; i<transits; i++) {
            double *tr = transit->row(transit_ind + i);
            unsigned child = tr[TC_Edge];
            int child_prior_ind, child_transit_ind;
            if (Get(child, &child_prior_ind, &child_transit_ind))
                StrAppend(&ret, "\n", i, " : ", (*names)[child_prior_ind], " = ", exp(tr[TC_Cost]));
        }
        return ret + "\n}";
    }
};

struct BigramLanguageModelBuilder {
    typedef map<string, CounterS> Words;
    Words words;
    string dir, name;
    int iteration, min_samples;
    BigramLanguageModelBuilder(const string &d, const string &n, int iter, int min_samps)
        : dir(d), name(n), iteration(iter), min_samples(min_samps) {}

    void Input(const string &fn, SentenceCorpus::Sentence *s) {
        for (auto i = s->begin(), e = s->end(); i != e; /**/) {
            auto j = FindOrInsert(words, tolower(i->text));
            if (1)        j->second.seen++;
            if (++i != e) j->second.incr(tolower(i->text));
        }
    }
    void Done() {
        if (min_samples) {
            for (auto i = words.begin(), e = words.end(); i != e; /**/) {
                CounterS *word = &i->second;
                if (word->seen < min_samples) words.erase(i++);
                else                                      i++;
            }
        }

        int total = 0, transits = 0, wordcount = 0;
        for (auto i = words.begin(), ie = words.end(); i != ie; ++i) {
            CounterS *word = &i->second;
            for (auto j = word->count.begin(), je = word->count.end(); j != je; /**/) {
                if (words.find(j->first) == words.end()) word->count.erase(j++);
                else { j++; transits++; }
            }
        }

        INFO("writing LM of ", words.size(), " words");
        const char *flagtext = "LM";
        Matrix *map_data = new Matrix(NextPrime(words.size()*4), 5*LanguageModel::map_values);
        HashMatrix map(map_data, LanguageModel::map_values);

        LocalFile names  (string(dir) + MatrixFile::Filename(name, "name",       "string", iteration), "w");
        LocalFile prior  (string(dir) + MatrixFile::Filename(name, "prior",      "matrix", iteration), "w");
        LocalFile transit(string(dir) + MatrixFile::Filename(name, "transition", "matrix", iteration), "w");

        MatrixFile::WriteHeader(&names,   basename(names.Filename(),0,0),   flagtext, words.size(), 1);
        MatrixFile::WriteHeader(&prior,   basename(prior.Filename(),0,0),   flagtext, words.size(), 1);
        MatrixFile::WriteHeader(&transit, basename(transit.Filename(),0,0), flagtext, transits,     TransitCols);

        wordcount = transits = 0;
        for (auto i = words.begin(); i != words.end(); ++i) {
            CounterS *word = &i->second;
            unsigned self = fnv32(i->first.c_str());
            double pr[] = { (double)i->second.seen /* log(i->second.seen/(float)total) */ }, *he;

            StringFile::WriteRow(&names, i->first);
            MatrixFile::WriteRow(&prior, pr, 1);

            if (!(he = map.Set(self))) FATAL("Matrix hash collision: ", (*i).first);
            he[LanguageModel::col_prior] = wordcount++;
            he[LanguageModel::col_transit] = transits;

            double tx[TransitCols];
            for (auto j = word->count.begin(); j != word->count.end(); j++, transits++) {
                tx[TC_Self] = self;
                tx[TC_Edge] = fnv32((*j).first.c_str());
                tx[TC_Cost] = log((double)(*j).second / word->incrs);
                MatrixFile::WriteRow(&transit, tx, TransitCols);
            }
        }

        MatrixFile out(map.map, flagtext);
        if (out.WriteVersioned(VersionedFileName(dir.c_str(), name.c_str(), "map"), iteration) < 0)
            ERROR(name, " write map");
    }
};

}; // namespace LFL
#endif // __LFL_NLP_LM_H__
