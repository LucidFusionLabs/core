/*
 * $Id: baumwelch.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_BAUMWELCH_H__
#define __LFL_ML_BAUMWELCH_H__
namespace LFL {

struct BaumWelch : public HMM::BaumWelchAccum {
    AcousticModel::Compiled *model;

    int K, D;
    double totalprob, accumprob, beamwidth;
    static const int minposterior = -16;

    struct Mode { enum { Means=1, Cov=2 }; }; int mode;

    struct Accum {
        GMM mixture;
        double total, prior;

        typedef map<int, double> XiMap;
        XiMap transit;
        double transtotal;
        int count;

        Accum(int K, int D) : mixture(K, D) {}
    } **accum;

    bool UsePrior, UseTransition, FullVariance;
    int HMMFlag;

    Matrix phonetx, phonetxtotal;

    BaumWelch(AcousticModel::Compiled *m, double beamWidth, bool usePrior=0, bool useTransition=0, bool fullVariance=0) :
        model(m), K(model->state[0].emission.mean.M), D(model->state[0].emission.mean.N),
        beamwidth(beamWidth), mode(0), accum(new Accum*[model->states]),
        UsePrior(usePrior), UseTransition(useTransition), FullVariance(fullVariance), HMMFlag(0),
        phonetx(LFL_PHONES, LFL_PHONES), phonetxtotal(LFL_PHONES, 1)
    {
        if (UsePrior) HMMFlag |= AcousticHMM::Flag::UsePrior;
        if (UseTransition) HMMFlag |= AcousticHMM::Flag::UseTransit;

        for (int i=0; i<model->states; i++) {
            AcousticModel::State *s = &model->state[i];
            if (s->val.emission_index != i) FATAL("BaumWelch emission_index mismatch ", s->val.emission_index, " != ", i);
            accum[i] = new Accum(K, D);
        }
        reset();
    }
    ~BaumWelch() { for(int i=0; i<model->states; i++) delete accum[i]; delete [] accum; }

    void reset() { 
        for (int i=0; i<model->states; i++) {
            Accum *a = accum[i];
            MatrixIter(&a->mixture.mean) a->mixture.mean.row(i)[j] = 0;
            MatrixIter(&a->mixture.diagcov) a->mixture.diagcov.row(i)[j] = 0;
            MatrixIter(&a->mixture.prior) a->mixture.prior.row(i)[j] = -INFINITY;
            a->total = -INFINITY;
            a->prior = -INFINITY;
            a->transit.clear();
            a->transtotal = -INFINITY;
            a->count = 0;
        }
        totalprob=accumprob=0;
        MatrixIter(&phonetx) phonetx.row(i)[j] = -INFINITY;
        MatrixIter(&phonetxtotal) phonetxtotal.row(i)[j] = -INFINITY;
    }

    void run(const char *modeldir, const char *featdir, int iterO, int iterI) {
        INFO("enter BaumWelch iteration ", iterO, "-", iterI);
        int count = FeatCorpus::feat_iter(featdir, BaumWelch::add_features, this);
        INFOf("exit BaumWelch iteration %d-%d, utterances = %d, totalprob = %f, accumprob = %f", iterO, iterI, count, totalprob, accumprob);
    }

    void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
        if (!dim_check("BW", features->N, D)) return;

        AcousticModel::Compiled *hmm = AcousticModel::fromUtterance(model, transcript, UseTransition);
        if (!hmm) return DEBUG("utterance construct failed: ", transcript);

        double prob = AcousticHMM::forwardBackward(hmm, features, 2, beamwidth, this);
        if (mode == Mode::Means) totalprob += prob;

        delete hmm;
    }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { return ((BaumWelch*)arg)->add_features(fn, MFCC, features, transcript); }

    void add_prior(int state, double pi) {
        if (state < 0 || state >= model->states) FATAL("OOB emission index ", state);
        Accum *a = accum[state];
        logadd(&a->prior, pi);
    }

    void add_emission(int state, const double *feature, const double *emissionPosterior, double emissionTotal, double gamma) {
        if (gamma < minposterior) return;
        if (state < 0 || state >= model->states) FATAL("OOB emission index ", state);
        Accum *a = accum[state];

        double *posterior = (double*)alloca(K*sizeof(double));
        for (int i=0; i<K; i++) posterior[i] = emissionPosterior[i] + gamma;
        emissionTotal += gamma;

        if (mode == Mode::Means) {
            double *featscaled = (double*)alloca(D*sizeof(double));
            for (int i=0; i<K; i++) {
                if (emissionPosterior[i] < minposterior) continue;
                Vector::mult(feature, exp(posterior[i]), featscaled, D);
                Vector::add(a->mixture.mean.row(i), featscaled, D);
                logadd(&a->mixture.prior.row(i)[0], posterior[i]);

                if (!FullVariance) {
                    Vector::mult(featscaled, feature, D);
                    Vector::add(a->mixture.diagcov.row(i), featscaled, D);
                }
            }
            a->count++;
            logadd(&a->total, emissionTotal); 
            accumprob += emissionTotal;
        }
        else if (mode == Mode::Cov) {
            double *diff = (double*)alloca(D*sizeof(double));
            for (int i=0; i<K; i++) {
                if (emissionPosterior[i] < minposterior) continue;
                Vector::sub(a->mixture.mean.row(i), feature, diff, D);
                Vector::mult(diff, diff, D);
                Vector::mult(diff, exp(posterior[i]), D);
                Vector::add(a->mixture.diagcov.row(i), diff, D);
            }
            if (!FullVariance) FATAL("incompatible modes ", -1);
        }
    }

    void add_transition(int Lstate, int Rstate, double xi) {
        if (mode != Mode::Means || !isfinite(xi)) return;
        if (Lstate < 0 || Lstate >= model->states) FATAL("OOB emission index ", Lstate);
        if (Rstate < 0 || Rstate >= model->states) FATAL("OOB emission index ", Rstate);
        Accum *a = accum[Lstate];

        Accum::XiMap::iterator i = a->transit.find(Rstate);
        if (i == a->transit.end()) {
            a->transit[Rstate] = -INFINITY;
            i = a->transit.find(Rstate);
        }

        logadd(&(*i).second, xi);
        logadd(&a->transtotal, xi);

        AcousticModel::State *s1 = &model->state[Lstate], *s2 = &model->state[Rstate];
        int ap, an, phone1 = AcousticModel::parseName(s1->name, 0, &ap, &an);
        int bp, bn, phone2 = AcousticModel::parseName(s2->name, 0, &bp, &bn);
        if (phone1 >= 0 && phone2 >= 0 && phone1 != phone2) {
            logadd(&phonetx.row(phone1)[phone2], xi);
            logadd(&phonetxtotal.row(phone1)[0], xi);
        }
    }

    void complete() {
        if (mode == Mode::Means) {
            for (int si=0; si<model->states; si++) {
                Accum *a = accum[si];
                for (int i=0; i<K; i++) {
                    double sum = a->mixture.prior.row(i)[0];
                    double dn = exp(sum);
                    if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;
                    Vector::div(a->mixture.mean.row(i), dn, D);
                }
            }
        }
        else if (mode == Mode::Cov) {
            double *x = (double*)alloca(D*sizeof(double));
            double totalprior = -INFINITY;
            for (int si=0; si<model->states; si++) {
                Accum *a = accum[si];
                logadd(&totalprior, a->prior);
            }

            if (model->phonetx && model->phonetx->M == LFL_PHONES && model->phonetx->N == LFL_PHONES) {
                MatrixIter(model->phonetx) model->phonetx->row(i)[j] = phonetx.row(i)[j] - phonetxtotal.row(i)[0];
            }

            for (int si=0; si<model->states; si++) {
                AcousticModel::State *s = &model->state[si];
                Accum *a = accum[si];
                double lc = log((float)a->count);

                for (int i=0; i<K; i++) {
                    double sum = a->mixture.prior.row(i)[0];
                    double dn = exp(sum);
                    if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;
                    Vector::div(a->mixture.diagcov.row(i), dn, D);

                    Vector::assign(s->emission.mean.row(i), a->mixture.mean.row(i), D);
                    Vector::assign(s->emission.diagcov.row(i), a->mixture.diagcov.row(i), D);

                    if (!FullVariance) {
                        Vector::mult(s->emission.mean.row(i), s->emission.mean.row(i), x, D);
                        Vector::sub(s->emission.diagcov.row(i), x, D);
                    }

                    MatrixColIter(&s->emission.diagcov)
                        if (s->emission.diagcov.row(i)[j] < FLAGS_CovarFloor) s->emission.diagcov.row(i)[j] = FLAGS_CovarFloor;

                    double prior = sum - lc;
                    s->emission.prior.row(i)[0] = prior < FLAGS_PriorFloor ? FLAGS_PriorFloor : prior;

                    s->emission.computeNorms();

                    s->prior = a->prior - totalprior;
                }

                int txind = 0;
                if (s->transition.M != a->transit.size() || s->transition.N != TransitCols) s->transition.open(a->transit.size(), TransitCols);
                for (Accum::XiMap::iterator i = a->transit.begin(); i != a->transit.end(); i++) {
                    AcousticModel::State *s2 = &model->state[(*i).first];
                    double *tr = s->transition.row(txind++);
                    tr[TC_Self] = fnv32(s->name.c_str());
                    tr[TC_Edge] = fnv32(s2->name.c_str());
                    tr[TC_Cost] = (*i).second - a->transtotal;
                }
                AcousticModel::State::sortTransitionMap(&s->transition);
            }
            reset();
        }
    }
};

}; // namespace LFL
#endif // __LFL_ML_BAUMWELCH_H__
