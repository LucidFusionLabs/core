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

#ifndef LFL_NLP_CORPUS_H__
#define LFL_NLP_CORPUS_H__
namespace LFL {

struct SentenceCorpus : public Corpus {
  struct Word {
    string text;
    vector<string> tag;
    Word(const string &t) : text(t) {}
  };

  struct Sentence : public vector<Word> {
    Sentence(const char *s) {
      StringWordIter words(s);
      for (string word = words.NextString(); !words.Done(); word = words.NextString()) push_back(Word(word));
    }
    string DebugString() const {
      int ind = 0;
      string ret = "Sentence:\n";
      for (auto w : *this) {
        StrAppend(&ret, "w[", ind++, "] = '", w.text, "'");
        for (auto j = w.tag.rbegin(), je = w.tag.rend(); j != je; ++j) StrAppend(&ret, ", ", j);
        ret += "\n";
      }
      return ret;
    }
  };

  typedef function<void(const string&, Sentence*)> SentenceCB;
  SentenceCB sentence_cb;
  SentenceCorpus(const SentenceCB &cb) : sentence_cb(cb) {}
};

struct QueryCorpus : public SentenceCorpus {
  string query;
  QueryCorpus(const SentenceCB &cb, const string &q) : SentenceCorpus(cb) { query = q + "\r\n"; }
  void Run(const string &filename) {
    StringLineIter iter(query);
    for (const char *line = iter.Next(); line; line = iter.Next()) {
      Sentence s(line);
      sentence_cb(filename, &s);
    }
  }
};

struct TextCorpus : public SentenceCorpus {
  TextCorpus(const SentenceCB &cb) : SentenceCorpus(cb) {}
  void RunFile(const string &filename) {
    LocalFile file(filename, "r");
    if (!file.Opened()) return;
    NextRecordReader nr(&file);
    for (const char *line = nr.NextLine(); line; line = nr.NextLine()) {
      Sentence s(line);
      sentence_cb(filename, &s);
    }
  }
};

}; // namespace LFL
#endif // LFL_NLP_CORPUS_H__
