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

#include "core/app/gui.h"
#include "core/app/types/trie.h"
#include "core/ml/corpus.h"
#include "core/ml/counter.h"
#include "core/ml/hmm.h"
#include "core/speech/speech.h"
#include "corpus.h"
#include "lm.h"

namespace LFL {
DEFINE_string(query,           "",                          "Query corpus");
DEFINE_string(target,          "printer",                   "Handler [printer,lmbuilder,triebuilder]");
DEFINE_string(modeldir,        "model/",                    "Model directory");
DEFINE_int   (min_occurrences, 0,                           "Minimum occurrences");
DEFINE_string(corpus,          "text",                      "Corpus [query,text,treebank,propbank,nombank]");
DEFINE_string(corpuspath,      "",                          "Corpus path");
DEFINE_string(treecorpuspath,  "corpus/penn/combined/wsj/", "Treebank path");
DEFINE_string(propcorpuspath,  "corpus/propbank/frames/",   "Propbank path");
DEFINE_string(nomcorpuspath,   "corpus/nombank/frames/",    "Nombank path");
}; // namespace LFL
using namespace LFL;

extern "C" void MyAppCreate(int argc, const char* const* argv) {
  FLAGS_open_console = 1;
  app = new Application(argc, argv);
  app->focused = new Window();
  app->name = "trainer";
}

extern "C" int MyAppMain() {
  if (app->Create(__FILE__)) return -1;
  if (app->Init()) return -1;

  Callback finish_cb;
  SentenceCorpus::SentenceCB input_cb;
  if (FLAGS_target == "printer") {
    input_cb = [=](const string &fn, SentenceCorpus::Sentence *s) {
      printf("%s\n", s->DebugString().c_str());
    };
  } else if (FLAGS_target == "lmbuilder") {
    BigramLanguageModelBuilder *target =
      new BigramLanguageModelBuilder(FLAGS_modeldir, "LanguageModel", 0, FLAGS_min_occurrences);
    input_cb  = bind(&BigramLanguageModelBuilder::Input, target, _1, _2);
    finish_cb = bind(&BigramLanguageModelBuilder::Done,  target);
  } else if (FLAGS_target == "lmquery") {
    LanguageModel *lm = new LanguageModel();
    lm->Open("LanguageModel", FLAGS_modeldir.c_str());
    input_cb = [=](const string &fn, SentenceCorpus::Sentence *s) {
      for (auto w : *s) printf("%s\n", lm->DebugString(tolower(w.text).c_str()).c_str());
    };
  } else if (FLAGS_target == "triebuilder") {
    CounterS *words = new CounterS();
    input_cb = [=](const string &fn, SentenceCorpus::Sentence *s) { for (auto w : *s) words->Incr(tolower(w.text)); };
    finish_cb = [=]() {
      PatriciaCompleter<char, int> trie(words->count.begin(), words->count.end());
      for (auto w : words->count) {
        auto n = trie.Query(w.first);
        CHECK(n);
        CHECK(n->val_ind);
        CHECK_EQ(w.second, trie.val[n->val_ind-1].val);
      }
    };
  } else { ERROR("unknown target ", FLAGS_target); return -1; }

  SentenceCorpus *corpus = 0;
  if      (!FLAGS_query.empty())   corpus = new QueryCorpus(input_cb, FLAGS_query);
  else if (FLAGS_corpus == "text") corpus = new TextCorpus(input_cb);
  else { ERROR("unknown corpus ", FLAGS_corpus); return -1; }

  corpus->finish_cb = finish_cb;
  corpus->Run(FLAGS_corpuspath);
  return 0;
};
