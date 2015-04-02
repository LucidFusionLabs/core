/*
 * $Id: trainer.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "ml/hmm.h"
#include "speech.h"

#include "ml/corpus.h"
#include "ml/counter.h"
#include "ml/kmeans.h"
#include "ml/sample.h"
#include "ml/gmmem.h"

#include "corpus.h"
#include "ml/viterbitrain.h"
#include "ml/baumwelch.h"
#include "ml/cart.h"
#include "ml/cluster.h"
#include "nlp/corpus.h"
#include "nlp/lm.h"

#include "wfst.h"
#include "recognition.h"

namespace LFL {
DEFINE_string(homedir,               "/corpus",   "Home directory");
DEFINE_string(WavDir,                "wav",       "WAV directory");
DEFINE_string(ModelDir,              "model",     "Model directory");
DEFINE_string(DTreeDir,              "model",     "Decision tree directory");
DEFINE_string(FeatDir,               "features",  "Feature directory");
DEFINE_string(decodefile,            "",          "Decode audio file");
DEFINE_string(recognizefile,         "",          "Recognize audio file");
DEFINE_string(uttmodel,              "",          "Utterance transcript");
DEFINE_string(vocab,                 "",          "Vocab file path");

DEFINE_int   (Initialize,            0,           "Initialize models");
DEFINE_int   (WantIter,              -1,          "Model iteration");
DEFINE_bool  (fullyConnected,        false,       "Decode with fully connected HMM");
DEFINE_bool  (visualize,             false,       "Visualize features");
DEFINE_bool  (interactive,           false,       "Interactive visualization");
DEFINE_bool  (wav2feats,             false,       "Create features from transcribed audio corpus");
DEFINE_bool  (feats2cluster,         false,       "Cluster features");
DEFINE_bool  (samplemean,            false,       "Initialize components to sample mean and variance");
DEFINE_bool  (UniformViterbi,        false,       "Uniform viterbi paths on iteration 0");
DEFINE_bool  (FullVariance,          false,       "Make second pass for variance");
DEFINE_bool  (AcceptIncomplete,      false,       "Accept paths that don't reach final state");
DEFINE_bool  (viterbitrain,          false,       "Viterbi Training");
DEFINE_bool  (baumwelch,             false,       "Baum Welch Training");
DEFINE_bool  (feats2pronounce,       false,       "Check corpus pronunciations");
DEFINE_bool  (wav2segs,              false,       "Segment audio");
DEFINE_bool  (clonetriphone,         false,       "Initialize context dependent model");
DEFINE_string(clonetied,             "",          "Initialize tied context dependent model using tiedstates.matrix");
DEFINE_bool  (triphone2tree,         false,       "Build decision trees");
DEFINE_bool  (tiestates,             false,       "Tie model states");
DEFINE_bool  (tieindependent,        false,       "Tie model states to context independent phones");
DEFINE_bool  (compilerecognition,    false,       "Compile recognition network");
DEFINE_bool  (rewritemodel,          false,       "Rewrite acoustic model (restoring map after edits)");
DEFINE_int   (growcomponents,        0,           "Change components in model to");
DEFINE_bool  (recognizecorpus,       false,       "Measure decoding accuracy against training corpus");
DEFINE_string(recognizequery,        "",          "Query probability assigned to word");

DEFINE_double(BeamWidth,             256,         "Beam search width");
DEFINE_double(CovarFloor,            1e-6,        "Model covariance floor");
DEFINE_double(PriorFloor,            -16,         "Model prior and mixture weight floor");
DEFINE_double(LanguageModelWeight,    2,          "Language model weight");
DEFINE_double(WordInsertionPenalty,   0,          "Word insertion penalty");
DEFINE_int   (MaxIterations,         20,          "Training iterations");
DEFINE_int   (InnerIterations,       1,           "Inner training iterations");
DEFINE_int   (MeansIterations,       1,           "Means training iterations");
DEFINE_int   (MixtureComponents,     32,          "Guassians per mixture");
DEFINE_int   (UsePrior,              0,           "Use prior probabilities in training");
DEFINE_int   (UseTransition,         1,           "Use transition probabilities in training");
DEFINE_string(UttPathsInFile,        "",          "Viterbi paths input file");
DEFINE_string(UttPathsOutFile,       "",          "Viterbi paths output file");

AssetMap asset;
SoundAssetMap soundasset;

void MyResynth(const vector<string> &args) { SoundAsset *sa=soundasset(args.size()?args[0]:"snap"); if (sa) { resynthesize(&app->audio, sa); } }

struct Wav2Features {
    enum Target { File, Archive };
    string dir;
    Target targ;
    MatrixArchiveOut out;

    Wav2Features(const string &Dir, Target Targ) : dir(Dir), targ(Targ) {
        if (targ == Archive) {
            string outfile = dir + StringPrintf("%lx.featlist", ::rand());
            INFO("selected output file: ", outfile);
            out.Open(outfile);
        }
    }

    void AddFeatures(const string &sd, SoundAsset *wav, const char *transcript) {
        Matrix *features = Features::fromAsset(wav, Features::Flag::Storable);
        MatrixFile feat(features, transcript);
        if (targ == File) {
            string outfile = dir + StringPrintf("%lx.feat", fnv32(BaseName(wav->filename)));
            feat.Write(outfile, wav->filename);
        } else if (targ == Archive) {
            out.Write(&feat, wav->filename);
        }
    }   
};

struct Features2Pronunciation {
    CounterS words, phones;
    AcousticModel::Compiled *model;
    bool UseTransition;

    Features2Pronunciation(AcousticModel::Compiled *M=0, bool useTransition=0) : model(M), UseTransition(useTransition) {}

    void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
        PronunciationDict *dict = PronunciationDict::instance();

        StringWordIter worditer(transcript);
        for (const char *word = worditer.Next(); word; word = worditer.Next()) {
            if (!dict->pronounce(word)) ERROR("no pronunciation dictionary for '", word, "'");
            words.incr(word);
        }
        if (!model) return;

        AcousticModel::Compiled *hmm = AcousticModel::fromUtterance(model, transcript, UseTransition);
        if (!hmm) { ERROR("compile utterance '", transcript, "' failed"); return; }

        for (int i=0; i<hmm->states; i++)
            phones.incr(hmm->state[i].name);

        delete hmm;
    }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *F, const char *T, void *a) { return ((Features2Pronunciation*)a)->add_features(fn, MFCC, F, T); }

    void add_path(AcousticModel::Compiled *hmm, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript) {
        PronunciationDict *dict = PronunciationDict::instance();

        StringWordIter worditer(transcript);
        for (const char *word = worditer.Next(); word; word = worditer.Next()) {
            if (!dict->pronounce(word)) ERROR("no pronunciation dictionary for '", word, "'");
            words.incr(word);
        }
        if (!model) return;

        if (hmm) ERROR("unexpected parameter ", hmm);
        hmm = AcousticModel::fromUtterance(model, transcript, UseTransition);
        if (!hmm) { ERROR("compile utterance '", transcript, "' failed"); return; }

        if (viterbi->M != features->M) { ERROR("viterbi length mismatch ", viterbi->M, " != ", features->M); }
        MatrixRowIter(viterbi) {
            int vind = viterbi->row(i)[0];
            if (vind < 0 || vind >= hmm->states) FATAL("oob vind ", vind);
            phones.incr(hmm->state[vind].name);
        }

        delete hmm;
    }
    static void add_path(AcousticModel::Compiled *hmm, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { return ((Features2Pronunciation*)arg)->add_path(hmm, viterbi, vprob, vtime, MFCC, features, transcript); }

    void iter(const char *featdir) { FeatCorpus::feat_iter(featdir, add_features, this); }

    void print() {
        INFO("words: ", words.incrs);
        for (CounterS::Count::iterator i = words.count.begin(); i != words.count.end(); i++)
            INFO((*i).first, " \t ", (*i).second);

        INFO("phones: ", phones.incrs);
        for (CounterS::Count::iterator i = phones.count.begin(); i != phones.count.end(); i++)
            INFO((*i).first, " \t ", (*i).second);
    }
};

int model1init(const char *modeldir, int K, int D, Matrix *mean=0, Matrix *covar=0) {
    AcousticModelBuilder model;
    double mw = log(1.0/K), tp = log(.5);
    int phone_id = 0;

    PhonemeIter() { /* create phone models */
        int states = phone == Phoneme::SIL ? 1 : AcousticModel::StatesPerPhone; 

        for (int j=0; j<states; j++) {
            AcousticModel::State *s = new AcousticModel::State();
            s->name = AcousticModel::name(phone, j);
            
            s->emission.prior.Open(K, 1, mw);

            if (phone == Phoneme::SIL) {
                s->transition.Open(LFL_PHONES, TransitCols, log(1.0/LFL_PHONES));
                for (int k=0; k<LFL_PHONES; k++) {
                    s->transition.row(k)[TC_Self] = s->id();
                    s->transition.row(k)[TC_Edge] = fnv32(AcousticModel::name(k, 0).c_str());
                }
            }
            else {
                s->transition.Open(2, TransitCols, tp);
                s->transition.row(0)[TC_Self] = s->id();
                s->transition.row(0)[TC_Edge] = s->id(); /* to self */

                s->transition.row(1)[TC_Self] = s->id();
                s->transition.row(1)[TC_Edge] = fnv32(j == states-1 ? "Model_SIL_State_00" : AcousticModel::name(phone, j+1).c_str());
            }

            s->emission.mean.Open(K, D);
            if (mean) MatrixRowIter(&s->emission.mean) Vector::Assign(s->emission.mean.row(i), mean->row(0), D);

            s->emission.diagcov.Open(K, D);
            if (covar) MatrixRowIter(&s->emission.diagcov) Vector::Assign(s->emission.diagcov.row(i), covar->row(0), D);

            model.add(s, phone_id++);
        }
    }

    if (AcousticModel::write(&model, "AcousticModel", modeldir, 0, 0)) ERROR("write ", modeldir);
    INFOf("initialized %d phones, %d states each (%d)", LFL_PHONES, AcousticModel::StatesPerPhone, model.getStateCount());
    return 0;
}

int model3initwrite(vector<string> &list, Matrix *siltrans, const char *modeldir, AcousticModel::StateCollection *model1, int lastiter) {
    /* sort */ 
    sort(list.begin(), list.end(), AcousticModel::triphoneSort);

    /* compile model */
    int model3count=0;
    AcousticModel::Compiled model3(list.size());

    for (vector<string>::iterator it = list.begin(); it != list.end(); it++) {
        int prevPhone, nextPhone, phoneState, phone=AcousticModel::parseName((*it).c_str(), &phoneState, &prevPhone, &nextPhone);
        if (phone<0) phone = AcousticModel::parseName((*it).c_str(), &phoneState);
        if (phone<0) FATAL("unknown phone ", (*it));

        AcousticModel::State *s1 = model1->getState(fnv32(AcousticModel::name(phone, phoneState).c_str()));
        if (!s1) FATAL("missing model ", AcousticModel::name(phone, phoneState), " (", (*it), ")");

        AcousticModel::State *s3 = &model3.state[model3count++];

        s3->assignPtr(s1);

        s3->name = (*it).c_str();

        if (phone == Phoneme::SIL) s3->transition.AssignDataPtr(siltrans->M, siltrans->N, siltrans->m);
        else {
            s3->transition.Open(2, TransitCols, log(.5));
            s3->transition.row(0)[TC_Self] = s3->id();
            s3->transition.row(0)[TC_Edge] = s3->id(); /* to self */

            s3->transition.row(1)[TC_Self] = s3->id();
            s3->transition.row(1)[TC_Edge] = fnv32(phoneState == AcousticModel::StatesPerPhone-1 ? "Model_SIL_State_00" : AcousticModel::name(prevPhone, phone, nextPhone, phoneState+1).c_str());
        }
    }

    /* write model */
    FLAGS_TriphoneModel = true;
    if (AcousticModel::write(&model3, "AcousticModel", modeldir, lastiter+1, 0)) ERROR("write ", modeldir);
    INFO("initialized ", model3.getStateCount(), " states");
    return 0;
}

int model3init(const char *modeldir, AcousticModel::StateCollection *model1, int lastiter, const char *featdir, const char *vfn) {
    PronunciationDict *dict = PronunciationDict::instance();
    string silname = AcousticModel::name(Phoneme::SIL, 0);

    map<string, int> inventory;
    map<int, int> begPhone, endPhone;
    int begTriphoneCount = 1;

    /* populate inventory */
#define inventory_add_word(inventory, dict, word) { \
    const char *pronunciation = dict->pronounce(word); \
    if (!pronunciation) { ERROR("no pronunciation for '", word, "'"); continue; } \
    int prevPhone=0, len=strlen(pronunciation); \
    for (int i=0; i<len; i++) { \
        int phone = pronunciation[i]; \
        int nextPhone = i+1<len ? pronunciation[i+1] : 0; \
        inventory[AcousticModel::name(prevPhone, phone, nextPhone, 0)] = 1; \
        prevPhone = phone; \
    } }

    /* from utterance corpus */
    if (featdir) {
        Features2Pronunciation f2p;
        f2p.iter(featdir);
        if (!Running()) return 0;

        for (CounterS::Count::iterator i = f2p.words.count.begin(); i != f2p.words.count.end(); i++) {
            const char *word = (*i).first.c_str();
            inventory_add_word(inventory, dict, word);
        }
    }

    /* from vocab file */
    if (vfn) {
        LocalFileLineIter vocab(vfn);
        for (const char *word=vocab.Next(); word; word=vocab.Next()) {
            inventory_add_word(inventory, dict, word);
        }
    }

    /* determine all begin/end phonemes */
    for (map<string, int>::iterator it = inventory.begin(); it != inventory.end(); it++) {
        int prevPhone, nextPhone, phone=AcousticModel::parseName((*it).first.c_str(), 0, &prevPhone, &nextPhone);
        if (!prevPhone) begPhone[phone] = 1; 
        if (!nextPhone) endPhone[phone] = 1;
    }

    /* add all begin, end pairs */
    for (map<string, int>::iterator it = inventory.begin(); it != inventory.end(); it++) {
        int prevPhone, nextPhone, phone=AcousticModel::parseName((*it).first.c_str(), 0, &prevPhone, &nextPhone);
        if (!prevPhone) {
            for (map<int, int>::iterator j = endPhone.begin(); j != endPhone.end(); j++)
                inventory[AcousticModel::name((*j).first, phone, nextPhone, 0)] = 1;
        }
        if (!nextPhone) {
            for (map<int, int>::iterator j = begPhone.begin(); j != begPhone.end(); j++)
                inventory[AcousticModel::name(prevPhone, phone, (*j).first, 0)] = 1;
        }
    }

    /* linearize */
    vector<string> list;

    PhonemeIter() {
        int states = phone == Phoneme::SIL ? 1 : AcousticModel::StatesPerPhone; 
        for (int j=0; j<states; j++)
            list.push_back(AcousticModel::name(phone, j));
    }

    for (map<string, int>::iterator it = inventory.begin(); it != inventory.end(); it++) {
        int prevPhone, nextPhone, phone=AcousticModel::parseName((*it).first.c_str(), 0, &prevPhone, &nextPhone);
        if (!prevPhone && !nextPhone) continue;
        if (!prevPhone) begTriphoneCount++;

        /* add subsequent model states */
        for (int j=0; j<AcousticModel::StatesPerPhone; j++)
            list.push_back(AcousticModel::name(prevPhone, phone, nextPhone, j));
    }

    /* build silence transit map */
    int silxfers=LFL_PHONES+begTriphoneCount, siltranscount=0;
    Matrix siltrans(silxfers, 3, log(1.0/silxfers));
    for (int i=0; i<LFL_PHONES; i++) siltrans.row(siltranscount++)[1] = fnv32(AcousticModel::name(i, 0).c_str());
    MatrixRowIter(&siltrans) siltrans.row(i)[0] = fnv32(silname.c_str());

    for (map<string, int>::iterator it = inventory.begin(); it != inventory.end(); it++) {
        int prevPhone, nextPhone, phone=AcousticModel::parseName((*it).first.c_str(), 0, &prevPhone, &nextPhone);
        if (!prevPhone && !nextPhone) continue;
        if (prevPhone) continue;

        if (siltranscount >= siltrans.M) FATAL("overflow ", siltranscount, " >= ", siltrans.M);
        siltrans.row(siltranscount++)[1] = fnv32((*it).first.c_str());
    }

    /* write */
    INFO(list.size(), " states (", begPhone.size(), " beg phones, ", endPhone.size(), "end phones)");
    return model3initwrite(list, &siltrans, modeldir, model1, lastiter);
}

int model3tiedinit(const char *modeldir, Matrix *tiedstates, AcousticModel::StateCollection *model1, int lastiter) {
    AcousticModel::State *sil = model1->getState(fnv32(AcousticModel::name(Phoneme::SIL, 0).c_str()));
    if (!sil) { ERROR("no sil state ", sil); return -1; }

    vector<string> inventory;
    for (int i=0; i<LFL_PHONES; i++)
        for (int j=0; j<LFL_PHONES; j++)
            for (int k=0; k<LFL_PHONES; k++)
                for (int l=0; l<AcousticModel::StatesPerPhone; l++) {
                    string name = AcousticModel::name(i, j, k, l);
                    unsigned hash = fnv32(name.c_str());
                    unsigned thash = AcousticModel::State::tied(tiedstates, i, j, k, l);
                    if (hash == thash) inventory.push_back(name);
                }

    return model3initwrite(inventory, &sil->transition, modeldir, model1, lastiter);
}

int growDecisionTrees(const char *modeldir, AcousticModel::Compiled *model3, int lastiter, const char *featdir) {

    /* get triphone occcurence counts from UttPathsIn */
    Features2Pronunciation f2p(model3);
    if (!FLAGS_UttPathsInFile.size()) { ERROR("tie model states requires -UttPathsInFile: ", -1); return -1; }

    MatrixArchiveIn UttPathsIn;
    UttPathsIn.Open(modeldir + FLAGS_UttPathsInFile);
    PathCorpus::path_iter(featdir, &UttPathsIn, Features2Pronunciation::add_path, &f2p);
    if (!Running()) return 0;

    /* initialize a decision tree for each center phoneme state */
    const int trees = LFL_PHONES * AcousticModel::StatesPerPhone;
    PhoneticDecisionTree::FeatureSet feats(f2p.phones.count.size());
    vector<int> root[trees];
    CART::TreeBuilder tree[trees];
    int ind = 0;

    /* populate a global feature inventory and root node sets for each tree */
    for (CounterS::Count::iterator i = f2p.phones.count.begin(); i != f2p.phones.count.end(); i++) {
        const char *triphone = (*i).first.c_str();
        AcousticModel::State *s = model3->getState(fnv32(triphone));
        if (s->emission.mean.M != 1) FATAL("expects single gaussian mixture ", -1);

        PhoneticDecisionTree::Feature *f = (PhoneticDecisionTree::Feature*)feats.get(ind);
        f->ind = ind++;
        f->D = s->emission.mean.N;
        f->count = (*i).second;
        f->mean = s->emission.mean.row(0);
        f->covar = s->emission.diagcov.row(0);
        f->name = s->name;

        int phoneState, prevPhone, nextPhone, phone=AcousticModel::parseName(f->name, &phoneState, &prevPhone, &nextPhone);
        if (phone < 0) ERROR("AcousticModel::parseName failed for ", f->name);
        int tree_n = (LFL_PHONES * phoneState) + phone;
        if (tree_n >= trees) FATAL("oob tree ", tree_n);
        root[tree_n].push_back(f->ind);

        vector<int> terminal;
        terminal.push_back(f->ind);
        INFO(f->name, " count ", f->count, " prob ", PhoneticDecisionTree::SplitRanker::likelihood(&feats, terminal));
    }

    PhoneticDecisionTree::QuestionList *QL = PhoneticDecisionTree::QuestionList::create();
    PhoneticDecisionTree::SplitRanker ranker;

    /* grow trees */
    for (int i=0; i<trees; i++) {
        if (!root[i].size()) continue;
        int phone = i % LFL_PHONES, state = i / LFL_PHONES;
        INFO("growing tree for phone ", Phoneme::name(phone), " state ", state);

        CART::grow_tree(&feats, QL, &ranker, "t", root[i], 35, tree[i]);
    }
    
    int leaves = 0;
    for (int i=0; i<trees; i++) {
        if (!root[i].size()) continue;
        int phone = i % LFL_PHONES, state = i / LFL_PHONES;
        INFO("writing tree for phone ", Phoneme::name(phone), " state ", state);

        string n = StrCat("DecisionTree_", Phoneme::name(phone), "_", state);
        if (CART::write_tree(tree[i], QL, n.c_str(), modeldir, lastiter)) FATAL("CART::write_tree: ", strerror(errno));
        leaves += tree[i].leafname.size();
    }

    INFO("final tree has ", leaves, " leaves");
    return 0;
}

int tieModelStates(const char *modeldir, AcousticModel::StateCollection *model3, int lastiter, const char *DTdir) {

    int iteration;
    if ((iteration = MatrixFile::FindHighestIteration(VersionedFileName(DTdir, "DecisionTree_AH_0", "questions"), "matrix")) == -1) FATAL("can open DT ", -1);

    const int trees = LFL_PHONES * AcousticModel::StatesPerPhone;
    CART::Tree tree[trees];
    PhoneticDecisionTree::QuestionList *QL = PhoneticDecisionTree::QuestionList::create();
    vector<unsigned> leaves;

    for (int t=0; t<trees; t++) {
        int phone = t % LFL_PHONES, state = t / LFL_PHONES;
        if (phone == Phoneme::SIL && state) continue;
        string n = StrCat("DecisionTree_", Phoneme::name(phone), "_", state);

        if (CART::read(DTdir, n.c_str(), iteration, QL, &tree[t])) {
            ERROR("read tree ", Phoneme::name(phone), " ", state);
            CART::blank(QL, &tree[t], (double)fnv32(AcousticModel::name(phone, state).c_str()));
        }
        if (!tree[t].namemap.map->M) tree[t].namemap.map->Open(5, CART::Tree::map_buckets*CART::Tree::map_values);

        int N = tree[t].leafnamemap.map->N;
        MatrixRowIter(tree[t].leafnamemap.map) {
            for (int j=0; j<N; j+=CART::Tree::map_values) {
                double *he = &tree[t].leafnamemap.map->row(i)[j];
                if (he[0] || he[1]) leaves.push_back(he[1]);
            }
        }
    }

    INFO("loaded ", leaves.size(), " leaves");
    AcousticModel::Compiled modelTied(leaves.size());
    int modelTiedCount=0;

    for (int i=0; i<leaves.size(); i++) {
        AcousticModel::State *s = model3->getState(leaves[i]);
        if (!s) FATAL("missing model ", leaves[i]);

        AcousticModel::State *st = &modelTied.state[modelTiedCount++];
        st->assignPtr(s);
        st->val.emission_index = i;
    }
    
    if (AcousticModel::write(&modelTied, "AcousticModel", modeldir, lastiter+1, 0)) ERROR("write ", modeldir);
    INFO("initialized ", modelTied.getStateCount(), " states");

    /* open model just wrote */
    AcousticModelFile modelWrote; int wroteiter;
    if ((wroteiter = modelWrote.Open("AcousticModel", modeldir))<0 || wroteiter != lastiter+1) { ERROR("read ", modeldir, " ", lastiter+1); return -1; }

    LocalFile tiedstates(string(modeldir) + MatrixFile::Filename("AcousticModel", "tiedstates", "matrix", lastiter+1), "w");
    MatrixFile::WriteHeader(&tiedstates, BaseName(tiedstates.Filename()), "", powf(LFL_PHONES, 3)*AcousticModel::StatesPerPhone, 1);

    for (int i=0; i<LFL_PHONES; i++) {
        for (int j=0; j<LFL_PHONES; j++) {
            for (int k=0; k<LFL_PHONES; k++) {
                for (int l=0; l<AcousticModel::StatesPerPhone; l++) {
                    if (!j && (i || l)) { double row[] = { (double)fnv32("Model_SIL_State_00") }; MatrixFile::WriteRow(&tiedstates, row, 1); continue; }

                    string name = AcousticModel::name(i, j, k, l);
                    PhoneticDecisionTree::Feature f;
                    memset(&f, 0, sizeof(f));
                    f.name = name.c_str();
                    const double *he = CART::query(&tree[LFL_PHONES*l+j], &f);
                    unsigned tieto = he[1];

                    AcousticModel::State *s = modelWrote.getState(tieto);
                    if (!s) FATAL("cant tie ", name, " to ", tieto);

                    double row[] = { (double)tieto } ;
                    MatrixFile::WriteRow(&tiedstates, row, 1); 

                    DEBUG("tie ", name, " to ", s->name);
                }
            }
        }
    }
    return 0;
}

int tieIndependentStates(const char *modeldir, AcousticModel::StateCollection *model1, int lastiter) {
    LocalFile tiedstates(string(modeldir) + MatrixFile::Filename("AcousticModel", "tiedstates", "matrix", lastiter), "w");
    MatrixFile::WriteHeader(&tiedstates, BaseName(tiedstates.Filename()), "", powf(LFL_PHONES, 3)*AcousticModel::StatesPerPhone, 1);

    for (int i=0; i<LFL_PHONES; i++) {
        for (int j=0; j<LFL_PHONES; j++) {
            for (int k=0; k<LFL_PHONES; k++) {
                for (int l=0; l<AcousticModel::StatesPerPhone; l++) {
                    if (!j && (i || l)) { double row[] = { (double)fnv32("Model_SIL_State_00") }; MatrixFile::WriteRow(&tiedstates, row, 1); continue; }

                    string name = AcousticModel::name(i, j, k, l);
                    unsigned tieto = fnv32(AcousticModel::name(j, l).c_str());
                    AcousticModel::State *s = model1->getState(tieto);
                    if (!s) FATAL("cant tie ", name, " to ", tieto);

                    double row[] = { (double)tieto } ;
                    MatrixFile::WriteRow(&tiedstates, row, 1); 

                    DEBUG("tie ", name, " to ", s->name);
                }
            }
        }
    }
    return 0;
}

int growComponents(const char *modeldir, AcousticModel::StateCollection *model, int lastiter, int newComponents) {
    AcousticModel::Compiled modelNext(model->getStateCount());
    modelNext.tiedstates = model->tiedStates();
    AcousticModel::StateCollection::Iterator iter; int ind=0;

    for (model->beginState(&iter); !iter.done; model->nextState(&iter)) {
        AcousticModel::State *s = iter.v;
        AcousticModel::State *sn = &modelNext.state[ind];
        sn->assignPtr(s);
        sn->val.emission_index = ind++;
        sn->emission.mean.Open(newComponents, s->emission.mean.N);
        sn->emission.diagcov.Open(newComponents, s->emission.diagcov.N);
        sn->emission.prior.Open(newComponents, s->emission.prior.N);
    }
    
    if (AcousticModel::write(&modelNext, "AcousticModel", modeldir, lastiter+1, 0)) ERROR("write ", modeldir);
    INFO("initialized ", modelNext.getStateCount(), " states");
    return 0;
}

int viterbiTrain(const char *featdir, const char *modeldir) {
    AcousticModelFile model; int lastiter;
    if ((lastiter = model.Open("AcousticModel", modeldir))<0) { ERROR("read ", modeldir, " ", lastiter); return -1; }
    AcousticModel::toCUDA(&model);

    ViterbiTrain train(&model, FLAGS_BeamWidth, FLAGS_UsePrior, FLAGS_UseTransition, FLAGS_FullVariance, (!FLAGS_Initialize && !FLAGS_AcceptIncomplete), FLAGS_Initialize ? ViterbiTrain::KMeansAccum::create : ViterbiTrain::GMMAccum::create);
    INFO("ViterbiTrain iter ", lastiter, ", BW=", FLAGS_BeamWidth, ", ", model.getStateCount(), " states, Accum=", FLAGS_Initialize?"KMeans":"GMM");

    for (int i=0; i<FLAGS_MaxIterations; i++) {
        bool init = lastiter==0;
        INFO("ViterbiTrain begin (init=", init, ")");

        if (FLAGS_UniformViterbi && init) {
            INFO("Using uniform viterbi paths");
            train.viterbF = AcousticHMM::uniformViterbi;
        }
        else train.viterbF = AcousticHMM::viterbi;

        int innerIteration = 0;
        for (int j=0; j<FLAGS_InnerIterations; j++) {
            for (int k=0; k<FLAGS_MeansIterations; k++) {
                train.mode = ViterbiTrain::Mode::Means;
                train.run(modeldir, featdir, !j && !k, lastiter, innerIteration++, !i ? FLAGS_UttPathsInFile : "", !i && !j && !k ? FLAGS_UttPathsOutFile : "");
                if (Running()) train.complete(); else return 0;
                INFOf("ViterbiTrain means, totalprob=%f (PP=%f) accumprob=%f (PP=%f) (init=%d) iter=(%d,%d,%d)",
                      train.totalprob, train.totalprob/-train.totalcount, train.accumprob, train.accumprob/-train.accumcount, init, i, j, k);
            }

            train.mode = ViterbiTrain::Mode::Cov;
            if (FLAGS_FullVariance) train.run(modeldir, featdir, 0, lastiter, innerIteration++, !i ? FLAGS_UttPathsInFile : "", "");
            if (Running()) train.complete(); else return 0;

            train.mode = ViterbiTrain::Mode::UpdateModel;
            train.complete();

            if (AcousticModel::write(&model, "AcousticModel", modeldir, lastiter+1, 0)) ERROR("ViterbiTrain iteration ", lastiter);
            INFO("ViterbiTrain iter ", lastiter, " completed wrote model iter ", lastiter+1);
            AcousticModel::toCUDA(&model);
            train.reset();
        }
        lastiter++;
    }
    return 0;
}

int baumWelch(const char *featdir, const char *modeldir) {
    AcousticModelFile model; int lastiter;
    if ((lastiter = model.Open("AcousticModel", modeldir))<0) { ERROR("read ", modeldir, " ", lastiter); return -1; }
    model.phonetx = new Matrix(LFL_PHONES, LFL_PHONES);
    AcousticModel::toCUDA(&model);

    BaumWelch train(&model, FLAGS_BeamWidth, FLAGS_UsePrior, FLAGS_UseTransition, FLAGS_FullVariance);
    INFO("BaumWelch iter ", lastiter, ", BW=", FLAGS_BeamWidth, ", ", model.getStateCount(), " states");

    for (int i=0; i<FLAGS_MaxIterations; i++) {
        int innerIteration = 0;

        train.mode = BaumWelch::Mode::Means;
        train.run(modeldir, featdir, lastiter, innerIteration++);
        if (Running()) train.complete(); else return 0;

        train.mode = BaumWelch::Mode::Cov;
        if (FLAGS_FullVariance) train.run(modeldir, featdir, lastiter, innerIteration++);
        if (Running()) train.complete(); else return 0;

        if (AcousticModel::write(&model, "AcousticModel", modeldir, lastiter+1, 0)) ERROR("BaumWelch iteration ", lastiter);
        INFO("BaumWelch iter ", lastiter, " completed wrote model iter ", lastiter+1);
        AcousticModel::toCUDA(&model);
        lastiter++;
    }
    return 0;
}

int compileRecognitionNetwork(const char *modeldir, AcousticModel::Compiled *AM, LanguageModel *LM, int lastiter, const char *vfn) {
    Semiring *K = Singleton<LogSemiring>::Get();
    WFST::AlphabetBuilder words, aux, cd(&aux);
    WFST::PhonemeAlphabet phones(&aux);
    WFST::AcousticModelAlphabet ama(AM, &aux);

    /* L */
    LocalFileLineIter vocab(vfn);
    WFST *L = WFST::pronunciationLexicon(K, &phones, &words, &aux, PronunciationDict::instance(), &vocab);
    WFST::Union(L->K, (WFST::TransitMapBuilder*)L->E, L->I, L->F);
    WFST::determinize(L, Singleton<TropicalSemiring>::Get());
    WFST::removeNulls(L); /* only to remove 1 null */
    WFST::removeNonAccessible(L);
    WFST::closure(L->K, (WFST::TransitMapBuilder*)L->E, L->I, L->F);
    ((WFST::Statevec*)L->F)->push_back(0);
    ((WFST::Statevec*)L->F)->sort();
    L->writeGraphViz("aa1.gv");

    /* G */
    vocab.Reset();
    WFST *G = WFST::grammar(K, &words, LM, &vocab);
    WFST::write(G, "grammar", modeldir, lastiter, false, false);
    delete G;

    /* C */
    WFST *C = WFST::contextDependencyTransducer(K, &phones, &cd, &aux);
    WFST::determinize(C, Singleton<TropicalSemiring>::Get());
    WFST::addAuxiliarySymbols((WFST::TransitMapBuilder*)C->E, &aux, K->One());
    WFST::invert((WFST::TransitMapBuilder*)C->E, &C->A, &C->B);
    WFST::contextDependencyRebuildFinal(C);
    ((WFST::Statevec*)C->F)->push_back(0);
    ((WFST::Statevec*)C->F)->sort();

    /* C o L */
    WFST::State::LabelMap CoLL;
    WFST *CoL = WFST::compose(C, L, "label-reachability", 0, &CoLL, WFST::Composer::T2MatchNull | WFST::Composer::FreeT1 | WFST::Composer::FreeT2);
    WFST::removeNonCoAccessible(CoL);
    WFST::removeNulls(CoL);
    WFST::normalizeInput(CoL);

    /* H */
    WFST *H = WFST::hmmTransducer(K, &ama, &cd);
    WFST::Union(H->K, (WFST::TransitMapBuilder*)H->E, H->I, H->F);
    WFST::removeNulls(H); /* only to remove 1 null */
    WFST::removeNonAccessible(H);
    WFST::closure(H->K, (WFST::TransitMapBuilder*)H->E, H->I, H->F);
    WFST::addAuxiliarySymbols((WFST::TransitMapBuilder*)H->E, &aux, 0, K->One());
    ((WFST::TransitMapBuilder*)H->E)->sort();

    /* H o C o L */
    WFST::State::LabelMap HoCoLL;
    WFST::Composer::SingleSourceNullClosure HoCoLR(true);
    WFST *HoCoL = WFST::compose(H, CoL, "epsilon-matching", &HoCoLR, &HoCoLL);
    WFST::removeNonCoAccessible(HoCoL);
    WFST::determinize(HoCoL, Singleton<TropicalSemiring>::Get());
    WFST::removeNulls(HoCoL);
    WFST::normalizeInput(HoCoL);
    WFST::closure(HoCoL->K, (WFST::TransitMapBuilder*)HoCoL->E, HoCoL->I, HoCoL->F);

    /* HW */
    WFST *HW = WFST::hmmWeightTransducer(K, &ama); 
    WFST::addAuxiliarySymbols((WFST::TransitMapBuilder*)HW->E, &aux, K->One());
    ((WFST::TransitMapBuilder*)HW->E)->sort();

    /* HW o H o C o L */
    WFST *HWoHoCoL = WFST::compose(HW, HoCoL, "epsilon-matching", 0, 0, WFST::Composer::T2MatchNull | WFST::Composer::FreeT1 | WFST::Composer::FreeT2);
    WFST::removeSelfLoops(HWoHoCoL);
    WFST::removeNonCoAccessible(HWoHoCoL);
    HWoHoCoL->minimize();
    HWoHoCoL->writeGraphViz("aa92.gv");

    /* optimize */
    WFST *final = HWoHoCoL;
    WFST::replaceAuxiliarySymbols((WFST::TransitMapBuilder*)final->E); 
    WFST::shiftFinalTransitions(final, 0, WFST::IOAlphabet::auxiliary_symbol_id(1));
    final->writeGraphViz("aa93.gv");
    WFST::removeNulls(final); 
    WFST::normalizeInput(final);
    WFST::replaceAuxiliarySymbols((WFST::TransitMapBuilder*)final->E); 
    WFST::pushWeights(Singleton<LogSemiring>::Get(), final, final->I, true);
    final->minimize();
    final->writeGraphViz("aa99.gv");

    /* finish */
    WFST::write(final, "recognition", modeldir, lastiter);
    delete final;
    return 0;
}

int recognizeQuery(RecognitionModel *model, const char *input) {
    PronunciationDict *dict = PronunciationDict::instance();
    StringWordIter words(input);
    vector<int> query, query2;
    int pp = 0;

    for (string nextword, word = words.Next(); word.size(); word = nextword) {
        nextword = BlankNull(words.Next());
        const char *pronunciation = dict->pronounce(word.c_str());
        query2.push_back(model->recognitionNetworkOut.id(word.c_str()));

        for (const char *pi = pronunciation; *pi; pi++) {
            int p = *pi, pn = *(pi+1);
            if (!pn && nextword.size()) pn = dict->pronounce(nextword.c_str())[0];

            for (int j=0; j<AcousticModel::StatesPerPhone; j++) {
                unsigned hash = AcousticModel::State::tied(model->acousticModel.tiedStates(), pp, p, pn, j);
                AcousticModel::State *s = model->acousticModel.getState(hash);
                INFOf("in[%d] = %s (%d,%d,%d,%d)", query.size(), s->name.c_str(), pp, p, pn, j);
                query.push_back(s->val.emission_index);
            }
            pp = p;
        }
    }

    WFST::ShortestDistanceMap search(Singleton<TropicalSemiring>::Get());
    int flag = WFST::ShortestDistance::Transduce | WFST::ShortestDistance::SourceReset | WFST::ShortestDistance::Trace;
    WFST::shortestDistance(&search, Singleton<TropicalSemiring>::Get(), model->recognitionNetwork.E, model->recognitionNetwork.I, 0, 0, &query, flag);
    vector<int> successful;
    search.successful(model->recognitionNetwork.F, successful);
    INFO("recognition: ", successful.size(), " successful");

    if (1) {
        WFST::ShortestDistance::Path *path = search.get(0);
        INFO("to 0 = ", path->D);
        for (int i=0, l=path->traceback.size(); i<l; i++) INFO("traceback[", i, "] = ", path->traceback[i]);
    }

    for (int i=0, l=successful.size(); i<l; i++) {
        WFST::ShortestDistance::Path *path = search.get(successful[i]);
        string v;
        for (int k=0; k<path->out.size(); k++) StrAppend(&v, model->recognitionNetwork.B->name(path->out[k]), ", ");
        INFO(successful[i], " = ", path->D, " (", v, ")");
    }

    search.clear(); successful.clear();
    WFST::shortestDistance(&search, Singleton<TropicalSemiring>::Get(), model->grammar.E, model->grammar.I, 0, 0, &query2, flag);
    search.successful(model->grammar.F, successful);
    INFO("grammar: ", successful.size(), " successful");
    
    if (1) {
        WFST::ShortestDistance::Path *path = search.get(0);
        INFO("to 0 = ", path->D);
        for (int i=0, l=path->traceback.size(); i<l; i++) INFO("traceback[", i, "] = ", path->traceback[i]);
    }

    for (int i=0, l=successful.size(); i<l; i++) {
        WFST::ShortestDistance::Path *path = search.get(successful[i]);
        string v;
        for (int k=0; k<path->out.size(); k++) StrAppend(&v, model->recognitionNetwork.B->name(path->out[k]), ", ");
        INFO(successful[i], " = ", path->D, " (", v, ")");
    }

    return 0;
}

struct RecognizeCorpus {
    RecognitionModel *recognize; double WER; int total;
    RecognizeCorpus(RecognitionModel *model) : recognize(model), WER(0), total(0) {}
    void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
        INFO("IN = '", transcript, "'");
        Timer vtime; double vprob = 0;
        matrix<HMM::Token> *viterbi = Recognizer::decodeFeatures(recognize, features, FLAGS_BeamWidth, FLAGS_UseTransition, &vprob, FLAGS_lfapp_debug ? &recognize->nameCB : 0);
        string decodescript = Recognizer::transcript(recognize, viterbi);
        double time = vtime.GetTime(), wer = Recognizer::wordErrorRate(recognize, transcript, decodescript);
        WER += wer; total++;
        INFO("OUT = '", decodescript, "' WER=", wer, " (total ", WER/total, ")");
        if (FLAGS_lfapp_video) visualize(recognize, MFCC, viterbi, vprob, time);
        delete viterbi;
    }
    static void visualize(RecognitionModel *recognize, Matrix *MFCC, matrix<HMM::Token> *viterbi, double vprob, double time) {
        Matrix path(viterbi->M, 1);
        AcousticModel::Compiled *hmm = Recognizer::decodedAcousticModel(recognize, viterbi, &path);
        Decoder::visualizeFeatures(hmm, MFCC, &path, vprob, time, FLAGS_interactive);
        delete hmm;
    }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *F, const char *T, void *a) { return ((RecognizeCorpus*)a)->add_features(fn, MFCC, F, T); }
};

struct Wav2Segments {
    struct Out {
        LocalFile lf;
        WavWriter wav;
        MatrixArchiveOut index;
        int count, samples;
        Out(const char *basename) : lf(StrCat(basename, ".wav"), "w"), wav(&lf), index(StrCat(basename, ".mat").c_str()), count(0), samples(0) {}
    };

    AcousticModel::Compiled *model;
    Out **out;
    int HMMFlag;

    ~Wav2Segments() { for (int i=0; i<LFL_PHONES; i++) delete out[i]; delete [] out; }
    Wav2Segments(AcousticModel::Compiled *M, const char *dir) : model(M), out(new Out *[LFL_PHONES]), HMMFlag(0) {
        int len = 0;
        for (int i=0; i<LFL_PHONES; i++) out[len++] = new Out(StrCat(dir, "seg", Phoneme::name(i)).c_str());
    }

    void AddWAV(const string &sd, SoundAsset *wav, const char *transcript) {
        Matrix *MFCC = Features::fromAsset(wav, Features::Flag::Storable);
        Matrix *features = Features::fromFeat(MFCC->Clone(), Features::Flag::Full);

        AcousticModel::Compiled *hmm = AcousticModel::fromUtterance1(model, transcript, HMMFlag & AcousticHMM::Flag::UseTransit);
        if (!hmm) return DEBUG("utterance decode failed");
        if (!DimCheck("Wav2Segments", features->N, hmm->state[0].emission.mean.N)) return;

        Matrix viterbi(features->M, 1); Timer vtime;
        double vprob = AcousticHMM::viterbi(hmm, features, &viterbi, 2, FLAGS_BeamWidth, HMMFlag);
        if (FLAGS_lfapp_video) Decoder::visualizeFeatures(hmm, MFCC, &viterbi, vprob, vtime.GetTime(), FLAGS_interactive);

        int transitions=0, longrun=0;
        for (Decoder::PhoneIter iter(hmm, &viterbi); !iter.done(); iter.next()) {
            if (!iter.phone) continue;
            int len = iter.end - iter.beg;
            if (len > longrun) longrun = len;

            RingBuf::Handle B(wav->wav, iter.beg*FLAGS_feat_hop, len*FLAGS_feat_hop);
            Out *o = out[iter.phone];
            o->wav.Write(&B);

            Matrix seg(1,2);
            seg.row(0)[0] = o->samples;
            seg.row(0)[1] = o->samples + len*FLAGS_feat_hop;
            o->samples += len*FLAGS_feat_hop;

            MatrixFile f(&seg, "range");
            string fn = StringPrintf("%d:%s:%d:%d-%d:%s", o->count++, wav->filename.c_str(),
                transitions, iter.beg*FLAGS_feat_hop, (iter.beg+len)*FLAGS_feat_hop,
                hmm->state[(int)viterbi.row(iter.beg)[0]].name.c_str());

            o->index.Write(&f, fn);
            f.Clear();

            transitions++;  
        }

        INFO(wav->filename, " ", features->M, " features ", transitions, " transitions, longrun ", longrun);
        delete features;
        delete MFCC;
        delete hmm;
    }
};

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "trainer.txt");
    screen->caption = "trainer";
    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    INFO("LFL_PHONES=", LFL_PHONES);

    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_visualize;
    FLAGS_lfapp_camera = FLAGS_lfapp_network = 0;
#ifdef _WIN32
    open_console = 1;
#endif
    if (app->Init()) { app->Free(); return -1; }

    asset.Add(Asset("snap", 0, 0, 0, 0, 0, 0, 0, 0));
    asset.Load();
    app->shell.assets = &asset;

    soundasset.Add(SoundAsset("snap", 0, new RingBuf(FLAGS_sample_rate*FLAGS_sample_secs), 1, FLAGS_sample_rate, FLAGS_sample_secs));
    soundasset.Load();
    app->shell.soundassets = &soundasset;

    BindMap *binds = screen->binds = new BindMap();
    binds->Add(Bind(Key::Backquote, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit,   &app->shell, vector<string>()))));
    binds->Add(Bind(Key::F5,        Bind::CB(bind(&Shell::play,   &app->shell, vector<string>(1, "snap")))));
    binds->Add(Bind(Key::F6,        Bind::CB(bind(&Shell::snap,   &app->shell, vector<string>(1, "snap")))));
    binds->Add(Bind(Key::F7,        Bind::CB(bind(&MyResynth,                  vector<string>(1, "snap")))));
    binds->Add(Bind(Key::F8,        Bind::CB(bind(&Shell::sinth,  &app->shell, vector<string>(1, "440" )))));

    string wavdir=FLAGS_homedir, featdir=FLAGS_homedir, modeldir=FLAGS_homedir, dtdir=FLAGS_homedir;
    wavdir += "/" + FLAGS_WavDir + "/";
    modeldir += "/" + FLAGS_ModelDir + "/";
    featdir += "/" + FLAGS_FeatDir + "/";
    dtdir += "/" + FLAGS_DTreeDir + "/"; 

#define LOAD_ACOUSTIC_MODEL(model, lastiter) AcousticModelFile model; int lastiter; \
    if ((lastiter = model.Open("AcousticModel", modeldir.c_str(), FLAGS_WantIter)) < 0) FATAL("LOAD_ACOUSTIC_MODEL ", modeldir, " ", lastiter); \
    INFO("loaded acoustic model iter ", lastiter, ", ", model.getStateCount(), " states");

#define LOAD_LANGUAGE_MODEL(model, lastiter) LanguageModel model; int lastiter; \
    if ((lastiter = model.Open("LanguageModel", modeldir.c_str())) < 0) FATAL("LOAD_LANGUAGE_MODEL ", modeldir, " ", lastiter); \
    INFO("loaded language model iter ", lastiter, ", ", model.prior->M, " words, ", model.transit->M, " transits");

    if (FLAGS_wav2feats) {
        INFO("wav2features begin (wavdir='", wavdir, "', featdir='", featdir, "')");
        Wav2Features w2f(featdir, Wav2Features::Target::Archive);
        WavCorpus wav(bind(&Wav2Features::AddFeatures, &w2f, _1, _2, _3));
        wav.Run(wavdir);
        INFO("wav2features end");
    }
    if (FLAGS_feats2cluster) {
        INFO("cluster begin (featdir='", featdir, "', modeldir='", modeldir, "')");
        features2cluster(featdir.c_str(), modeldir.c_str(), ClusterAlgorithm::GMMEM);
        INFO("cluster end");
    }
    if (FLAGS_samplemean) {
    }
    if (FLAGS_viterbitrain) {
        if (FLAGS_Initialize) model1init(modeldir.c_str(), FLAGS_MixtureComponents, FLAGS_Initialize);
        INFO("train begin (featdir='", featdir, "', modeldir='", modeldir, "')");
        viterbiTrain(featdir.c_str(), modeldir.c_str());
        INFO("train end");
    }
    if (FLAGS_baumwelch) {
        INFO("train begin (featdir='", featdir, "', modeldir='", modeldir, "')");
        baumWelch(featdir.c_str(), modeldir.c_str());
        INFO("train end");
    }
    if (FLAGS_feats2pronounce) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        Features2Pronunciation f2p(&model);
        f2p.iter(featdir.c_str());
        f2p.print();
    }
    if (FLAGS_clonetriphone) {
        LOAD_ACOUSTIC_MODEL(model1, lastiter);
        INFO("model3init begin (vocab='", FLAGS_vocab, "')");
        model3init(modeldir.c_str(), &model1, lastiter, !FLAGS_vocab.size() ? featdir.c_str() : 0, FLAGS_vocab.size() ? FLAGS_vocab.c_str() : 0);
        INFO("model3init end (vocab='", FLAGS_vocab, "')");
    }
    if (FLAGS_clonetied.size()) {
        LOAD_ACOUSTIC_MODEL(model1, lastiter);
        MatrixFile tied;
        tied.Read(FLAGS_clonetied.c_str());
        if (!tied.F || !tied.F->m) FATAL("read ", FLAGS_clonetied, " failed");
        INFO("model3init begin (vocab='", FLAGS_vocab, "')");
        model3tiedinit(modeldir.c_str(), tied.F, &model1, lastiter);
        INFO("model3init end (vocab='", FLAGS_vocab, "')");
    }
    if (FLAGS_triphone2tree) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        INFO("growDecisionTrees begin ", lastiter);
        growDecisionTrees(modeldir.c_str(), &model, lastiter, featdir.c_str());
        INFO("growDecisionTrees end ", lastiter);
    }
    if (FLAGS_tiestates) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        INFO("tieModelStates begin ", lastiter);
        tieModelStates(modeldir.c_str(), &model, lastiter, dtdir.c_str());
        INFO("tieModelStates end ", lastiter);
    }
    if (FLAGS_tieindependent) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        INFO("tieIndependentStates begin ", lastiter);
        tieIndependentStates(modeldir.c_str(), &model, lastiter);
        INFO("tieIndependentStates end ", lastiter);
    }
    if (FLAGS_growcomponents) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        INFO("growComponents begin ", lastiter);
        growComponents(modeldir.c_str(), &model, lastiter, FLAGS_growcomponents);
        INFO("growComponents end ", lastiter);
    }
    if (FLAGS_wav2segs) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        INFO("segmentation begin (wavdir='", wavdir, "', modeldir='", modeldir, "')");
        Wav2Segments w2s(&model, modeldir.c_str());
        WavCorpus wav(bind(&Wav2Segments::AddWAV, &w2s, _1, _2, _3));
        wav.Run(wavdir);
        INFO("segmentation end (", wavdir, ")");
    }
    if (FLAGS_compilerecognition) {
        LOAD_ACOUSTIC_MODEL(am, amiter);
        LOAD_LANGUAGE_MODEL(lm, lmiter);

        /* compile */
        INFO("compile recognition network (modeldir='", modeldir, "')");
        compileRecognitionNetwork(modeldir.c_str(), &am, &lm, amiter, FLAGS_vocab.size() ? FLAGS_vocab.c_str() : 0);
        INFO("compile recognition network end (modeldir='", modeldir, "')");
    }
    if (FLAGS_rewritemodel) {
        INFO("rewrite model begin");
        AcousticModelFile model; int lastiter;
        if ((lastiter = model.Open("AcousticModel", modeldir.c_str(), FLAGS_WantIter, true)) < 0) FATAL("LOAD_ACOUSTIC_MODEL ", modeldir, " ", lastiter);
        INFO("loaded acoustic model iter ", lastiter, ", ", model.getStateCount(), " states");

        if (AcousticModel::write(&model, "AcousticModel", modeldir.c_str(), lastiter+1, 0)) ERROR("write ", modeldir);
        INFO("rewrite model end");
    }
    if (FLAGS_uttmodel.size()) {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        AcousticModel::Compiled *hmm = AcousticModel::fromUtterance(&model, FLAGS_uttmodel.c_str(), FLAGS_UseTransition);
        if (!hmm) FATAL("compiled utterance for transcript '", FLAGS_uttmodel, "' failed");
        AcousticHMM::printLattice(hmm, &model);
    }
    if (FLAGS_decodefile.size()) do {
        LOAD_ACOUSTIC_MODEL(model, lastiter);
        AcousticModel::Compiled *hmm = FLAGS_fullyConnected ? AcousticModel::fullyConnected(&model) : AcousticModel::fromModel1(&model, false);
        if (!hmm) { ERROR("hmm create failed ", hmm); break; }

        INFO("decode(", FLAGS_decodefile, ") begin");
        Matrix *viterbi = Decoder::decodeFile(hmm, FLAGS_decodefile.c_str(), FLAGS_BeamWidth);
        HMM::printViterbi(viterbi, &model.nameCB);
        string transcript = Decoder::transcript(hmm, viterbi);
        INFO("decode(", FLAGS_decodefile, ") end");
        delete viterbi;
        delete hmm;
        INFO("decode transcript: '", transcript, "'");
    } while(0);

    if (FLAGS_recognizequery.size()) {
        RecognitionModel recognize;
        if (recognize.read("RecognitionNetwork", modeldir.c_str(), FLAGS_WantIter)) FATAL("open RecognitionNetwork ", modeldir);
        AcousticModel::toCUDA(&recognize.acousticModel);
        recognizeQuery(&recognize, FLAGS_recognizequery.c_str());
    }
    
    if (FLAGS_recognizecorpus) {
        RecognitionModel recognize;
        if (recognize.read("RecognitionNetwork", modeldir.c_str(), FLAGS_WantIter)) FATAL("open RecognitionNetwork ", modeldir);
        AcousticModel::toCUDA(&recognize.acousticModel);

        RecognizeCorpus tester(&recognize);
        INFO("begin RecognizeCorpus ", 0);
        int count = FeatCorpus::feat_iter(featdir.c_str(), RecognizeCorpus::add_features, &tester);
        INFO("end RecognizeCorpus ", count);
    }

    if (FLAGS_recognizefile.size()) {
        RecognitionModel recognize;
        if (recognize.read("RecognitionNetwork", modeldir.c_str(), FLAGS_WantIter)) FATAL("open RecognitionNetwork ", modeldir);
        if (FLAGS_lfapp_debug) RecognitionHMM::printLattice(&recognize);
        AcousticModel::toCUDA(&recognize.acousticModel);
        do {
            INFO("recognize(", FLAGS_recognizefile, ") begin");
            Matrix *MFCC=0; double vprob=0; Timer vtimer;
            matrix<HMM::Token> *viterbi = Recognizer::decodeFile(&recognize, FLAGS_recognizefile.c_str(), FLAGS_BeamWidth,
                                                                 FLAGS_UseTransition, &vprob, &MFCC, FLAGS_lfapp_debug ? &recognize.nameCB : 0);
            string transcript = Recognizer::transcript(&recognize, viterbi);
            INFO("recognize(", FLAGS_recognizefile, ") end");
            HMM::Token::printViterbi(viterbi, &recognize.nameCB);
            if (!transcript.size()) { ERROR("decode failed ", transcript.size()); break; }
            INFO("vprob = ", vprob, " : '", transcript, "'");
            if (FLAGS_lfapp_video) RecognizeCorpus::visualize(&recognize, MFCC, viterbi, vprob, vtimer.GetTime());
            delete viterbi;
            delete MFCC;
        } while (Running() && FLAGS_visualize && FLAGS_interactive);
    }

    return app->Free();
}
