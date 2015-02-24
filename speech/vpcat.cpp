/*
 * $Id: vpcat.cpp 1306 2014-09-04 07:13:16Z justin $
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

#include "lfapp/lfapp.h"
#include "ml/hmm.h"
#include "speech.h"
#include "ml/corpus.h"
#include "corpus.h"

namespace LFL {
DEFINE_string(homedir,     "/corpus",   "Home directory");
DEFINE_string(WavDir,      "wav",       "WAV directory");
DEFINE_string(ModelDir,    "model",     "Model directory");
DEFINE_string(DTreeDir,    "model",     "Decision tree directory");
DEFINE_string(FeatDir,     "features",  "Feature directory");
DEFINE_int   (WantIter,    -1,          "Model iteration");
DEFINE_bool  (vp,          false,       "Print viterbi paths");
DEFINE_bool  (amtx,        false,       "Print acoustic model transit");

void path(AcousticModel::Compiled *, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) {}

}; // namespace LFL
using namespace LFL;

extern "C" {
int main(int argc, const char *argv[]) {
    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }

    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_camera = FLAGS_lfapp_network = 0;
#ifdef _WIN32
    open_console = 1;
#endif

    if (app->Init()) { app->Free(); return -1; }

    string modeldir = StrCat(FLAGS_homedir, "/", FLAGS_ModelDir, "/");
    string featdir  = StrCat(FLAGS_homedir, "/", FLAGS_FeatDir,  "/");

#define LOAD_ACOUSTIC_MODEL(model, lastiter) AcousticModelFile model; int lastiter; \
    if ((lastiter = model.Open("AcousticModel", modeldir.c_str(), FLAGS_WantIter)) < 0) FATAL("LOAD_ACOUSTIC_MODEL ", modeldir, " ", lastiter); \
    INFO("loaded acoustic model iter ", lastiter, ", ", model.getStateCount(), " states");

    if (FLAGS_vp) {
        FLAGS_lfapp_debug = 1;
        MatrixArchiveIn ViterbiPathsIn;
        ViterbiPathsIn.Open(argv[1]);

        int count = PathCorpus::path_iter(featdir.c_str(), &ViterbiPathsIn, path, 0);
        INFO(count, " paths");
    }

    if (FLAGS_amtx) {
        LOAD_ACOUSTIC_MODEL(model, FLAGS_lastiter);
        if (FLAGS_lastiter < 0) FATAL("no acoustic model: ", FLAGS_lastiter);

        for (int i=0; i<model.states; i++) {
            HMM::ActiveStateIndex active(1,1,1);
            HMM::ActiveState::Iterator src(i);
            AcousticModel::State *s1 = &model.state[i];

            AcousticHMM::TransitMap tm(&model, true);
            AcousticHMM::TransitMap::Iterator iter;
            for (tm.begin(&iter, &active, &src); !iter.done; tm.next(&iter)) {
                unsigned hash = iter.state;
                AcousticModel::State *s2 = model.getState(hash);
                INFO(s1->name, " -> ", s2->name, " @ ", iter.cost);
            }
        }
    }

    return 0;
}
}

