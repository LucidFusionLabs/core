/*
 * $Id: aed.h 1330 2014-11-06 03:04:15Z justin $
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

#ifndef __LFL_LFAPP_AED_H__
#define __LFL_LFAPP_AED_H__
namespace LFL {

DECLARE_FLAG(speech_client, string);
bool speech_client_flood() { return FLAGS_speech_client == "flood"; }
bool speech_client_auto() { return FLAGS_speech_client == "auto"; }
bool speech_client_manual() { return !speech_client_auto() && !speech_client_flood(); }

struct AcousticEventDetector {
    FixedAlloc<65536*2> alloc;
    int feature_rate;
    long long samples_processed, samples_available;

    FeatureSink *sink;
    vector<StatefulFilter> featureFilters;
    RingBuf *featureBuf;

    /* An Adaptive and Fast Speech Detection Algorithm by Dragos Burileanu */
    RingBuf pe, zcr, szcr;
    float emax, emin, sl, sl2, nl, zcravg, zcrdev, szcravg, szcrdev;
    int initcount, slcount, sl2count, nlcount, zcrcount, szcrcount;
    RollingAvg<unsigned> remax, remin, rsl, rsl2, rnl;

    int t1countA, t1countB, t2countA, t2countB;
    long long total, t1A, t1B, t2A, t2B;

    struct Word {
        long long PS, PE;
        int ES, EE, ZE;
        void incrementEnd(int n=1) { PE+=n; EE+=n; }
        Word() : PS(0), PE(0), ES(0), EE(0), ZE(0) {}
        Word(long long ps, long long pe, bool es=0) : PS(ps), PE(pe), ES(es), EE(0), ZE(0) {}
        void reopen(long long pe, bool es=0) { PE=pe; ES=es; EE=0; ZE=0; }
    };
    deque<Word> words;
    Word &top() { return words.back(); }
    Word &bot() { return words.front(); }
    bool onair() { return words.size() && top().PE == -1; }

    static const int szcr_shift=3, d2jump_left=3, d2jump_right=7;
    double percent(long long n) const { return 1 - (double) (total - n) / pe.ring.size; }
    double seconds(long long n) const { return (double) (total - n) / pe.samples_per_sec; }

    double SNR(int t=8) {
        if      (t == 1) return sl/nl;
        else if (t == 2) return (sl-nl)/nl;
        else if (t == 3) return sl2/nl;
        else if (t == 4) return (sl2-nl)/nl;
        else if (t == 5) return rsl.Avg() / rnl.Avg();
        else if (t == 6) return (rsl.Avg() - rnl.Avg()) / rnl.Avg();
        else if (t == 7) return rsl2.Avg() / rnl.Avg();
        else if (t == 8) return (rsl2.Avg() - rnl.Avg()) / rnl.Avg();
        else FATAL("unknown SNR_id ", t);
    }

    void alwaysComputeFeatures() {
        featureBuf = new RingBuf(feature_rate, feature_rate*FLAGS_sample_secs, sizeof(double)*Features::dimension());
    }

    ~AcousticEventDetector() { delete featureBuf; }
    AcousticEventDetector(int FeatureRate, FeatureSink *Sink) :
        feature_rate(FeatureRate),
        samples_processed(0), samples_available(0),
        sink(Sink), featureBuf(0),
        pe(feature_rate, feature_rate*FLAGS_sample_secs),
        zcr(feature_rate, feature_rate*FLAGS_sample_secs),
        szcr(feature_rate, feature_rate*FLAGS_sample_secs),
        emax(-INFINITY), emin(INFINITY), initcount(0),
        sl(0), sl2(0), nl(0), zcravg(0), zcrdev(0), szcravg(0), szcrdev(0),
        slcount(0), sl2count(0), nlcount(0), zcrcount(0), szcrcount(0),
        remax(7*feature_rate), remin(7*feature_rate), rsl(feature_rate), rsl2(feature_rate), rnl(feature_rate),
        t1countA(1), t1countB(1), t2countA(1), t2countB(1),
        total(0) {}

    void signal(long long begin, long long end) {
        if (!sink || speech_client_manual()) return;
        alloc.Reset();

        int frames = end - begin + 1;
        int offset = total - begin;

        if (!featureBuf) {
            RingBuf::Handle L(app->audio->IL, app->audio->RL.next-behind()-offset*FLAGS_feat_hop, FLAGS_feat_window+(frames-1)*FLAGS_feat_hop);
            Matrix features(frames, Features::dimension(), 0.0, 0, &alloc);
            Matrix *out = Features::fromBuf(&L, &features, &featureFilters, &alloc);
            sink->Write(out, begin);
        }
        else {
            RingBuf::MatrixHandle out(featureBuf, featureBuf->ring.back-1-offset, frames);
            sink->Write(&out, begin);
        }
    }

    void flush() {
        if (!sink || speech_client_manual()) return;
        sink->Flush();
    }

    int behind() { return samples_available - samples_processed; }

    void update(unsigned samples) {
        samples_available += samples;
        while (samples_available >= samples_processed + FLAGS_feat_window) {
            RingBuf::Handle L(app->audio->IL, app->audio->RL.next-behind(), FLAGS_feat_window);
            update(&L);
            samples_processed += FLAGS_feat_hop;
        }
    }

    /* for each feat_hop */
    void update(RingBuf::Handle *in) {
        alloc.Reset();

        /* burn 10 frames on init */
        if (++initcount < 10) return;

        /* compute features */
        if (featureBuf) {
            RingBuf::RowMatHandle featBH(featureBuf);
            Matrix *feat = featBH.Write();
            Features::fromBuf(in, feat, &featureFilters, &alloc);
        }

        /* compute ZCR & PE */
        float ZCR = *(float*)zcr.Write() = ZeroCrossings(/*&filtered_input*/ in, FLAGS_feat_window, 0);
        float PE = *(float*)pe.Write() = PseudoEnergy(/*&filtered_input*/ in, FLAGS_feat_window, 0);
        total++;

        /* smoothed zero crossing rate */
        float SZCR = 0; int zcr_extend = pe.ring.size/2;
        for (int i=0; i<szcr_shift; i++) SZCR += *(float*)zcr.Read(-i-1);
        SZCR = *(float*)szcr.Write() = SZCR / szcr_shift;

        /* track average zcr and standard deviation */
        float zcrdist = pow(zcravg - ZCR, 2);
        zcravg = CumulativeAvg(zcravg, ZCR, ++zcrcount);
        zcrdev = CumulativeAvg(zcrdev, zcrdist, zcrcount);

        /* track average smoothed zcr and standard deviation */
        float szcrdist = pow(szcravg - SZCR, 2);
        szcravg = CumulativeAvg(szcravg, SZCR, ++szcrcount);
        szcrdev = CumulativeAvg(szcrdev, szcrdist, szcrcount);

        /* track min & max energy */
        if (PE < emin) emin = PE;
        if (PE > emax) emax = PE;

        /* connect remin to PEdevB and remax to PEdevA with springs */
        float PEdevB = rnl.Avg();
        float PEdevA = rnl.Avg() * 2;
        remin.Add(PE < PEdevB ? PE : PEdevB);
        remax.Add(PE > PEdevA ? PE : PEdevA);
        float min = remin.count == remin.window ? remin.Avg() : emin;
        float max = remax.count == remax.window ? remax.Avg() : emax;

        /* pass thru */
        if (speech_client_flood()) return signal(total, total);

        /* burn 10 more frames on init */
        if (total < 10) {
            float PEfakeA=min*2, PEfakeB=min;
            sl = CumulativeAvg(sl, PEfakeA, ++slcount);
            nl = CumulativeAvg(nl, PEfakeB, ++nlcount);
            rsl.Add(PEfakeA);
            rnl.Add(PEfakeB);
            return;
        }

        /* energy threshholds */
        float t1 = min * (1 + 2*log10(max/min));
        float t2 = t1 + 0.20 * t1;
//      float t2 = t1 + 0.25 * (rsl.avg() - t1);

        /* track t1 crossings */
        if (PE > t1) {
            sl = CumulativeAvg(sl, PE, ++slcount);
            rsl.Add(PE);

            t1countA++;
            if (t1countB > 0) { t1countB=0; t1A=total; }
        }
        else {
            nl = CumulativeAvg(nl, PE, ++nlcount);
            rnl.Add(PE);

            t1countB++;
            if (t1countA > 0) { t1countA=0; t1B=total; }
        }

        /* track t2 crossings */
        if (PE > t2 && slcount > 0) {
            sl2 = CumulativeAvg(sl2, PE, ++sl2count);
            rsl2.Add(PE);

            t2countA++;
            if (t2countB > 0) { t2countB=0; t2A=total; }
        }
        else {
            t2countB++;
            if (t2countA > 0) { t2countA=0; t2B=total; }
        }

        /* new word */
        if (t2A == total && !onair()) {
            if (words.size() && (top().PE >= total-d2jump_right && top().EE < d2jump_right)) {
                signal(top().PE+1, total);
                top().reopen(-1, true);
            }
            else {
                Word nw(t1A, -1, true);
                words.push_back(nw);

                long long lpe = words.size()>1 ? words[words.size()-1].PE : 0;
                int lastcount = words.size();

                /* extend word left by szcr */
                int shift = total - t1A;
                RingBuf::Handle szcrh(&szcr, szcr.ring.back - shift);
                int shifted = extend_top_word_left(&szcrh, t1A, lpe, max, szcravg, sqrt(szcrdev), true);

                /* extend word left by zcr */
                if (lastcount == words.size()) {
                    RingBuf::Handle zcrh(&zcr, zcr.ring.back - shift - shifted);
                    extend_top_word_left(&zcrh, t1A, lpe, max - shifted, zcravg, sqrt(zcrdev));
                }

                /* write word's beginning frames */
                signal(top().PS, total);
            }
        }
        else if (onair()) {
            /* write word's current frame */
            signal(total, total);

            /* finish word */
            if (t1B == total) top().PE = t1B;
        }

        /* extend word right by zcr & szcr */
        if (words.size() && ((top().PE == total-1 && top().EE < zcr_extend) ||
                             (top().PE >= total-d2jump_right && top().EE < d2jump_right && top().PE != total)))
        {
            int orig = top().PE;
            int shift = total - top().PE;
            bool jumponly = top().PE != total-1;

            if (top().ZE == 0 || jumponly) {
                float mean=szcravg, var=sqrt(szcrdev), v=RingBuf::Handle(&szcr, szcr.ring.back - shift).Read(0);
                if (v < mean - var || v > mean + var) { top().incrementEnd(total - top().PE); top().ZE = 0; }
                else if (!jumponly) top().ZE = 1;
            }
            else if (top().ZE == 1) {
                float mean=zcravg, var=sqrt(zcrdev), v=RingBuf::Handle(&zcr, zcr.ring.back - shift).Read(0);
                if (v < mean - var || v > mean + var) top().incrementEnd();
                else top().ZE = 2;
            }

            /* write word's extended frame */
            if (top().PE == total)
                signal(orig+1, total);
        }

        /* flush word */
        if (words.size() && top().PE == total-1) flush();

        /* clear olds words */
        while (words.size() && bot().PE != -1 && total - bot().PE > pe.ring.size) words.pop_front();
    }

    int extend_top_word_left(RingBuf::Handle *in, long long tps, long long lpe, int max, float mean, float var, bool jump=false) {
        int extend=0, extendmax=(lpe && tps-lpe < max ? tps-lpe : max);

        if (jump) for (extend=d2jump_left; extend>0; extend++) {
            float v = in->Read(-1-extend);
            if (!(v < mean - var || v > mean + var)) break;
        }

        for (/**/; extend<extendmax; extend++) {
            float v = in->Read(-1-extend);
            if (!(v < mean - var || v > mean + var)) break;
        }

        if (lpe && extend == tps-lpe) {
            words[words.size()-1].PE = top().PE;
            words.pop_back();
        }
        else if (extend) top().PS -= extend;

        return extend;
    }
};

#ifdef __LFL_LFAPP_GUI_H__
struct AcousticEventGUI {
    static void Draw(AcousticEventDetector *AED, Box win, bool flip=false) {
        AED->alloc.Reset();

        int maxverts = AED->words.size()*6+speech_client_flood()*2;
        Geometry *geom = new Geometry(GraphicsDevice::Lines, maxverts, (v2*)0, 0, 0, Color(1.0,1.0,1.0));
        v2 *verts = (v2*)&geom->vert[0];
        geom->count = 0;

        if (speech_client_flood()) {
            verts[geom->count++] = !flip ? v2(win.x + 0 * win.w, win.y) : v2(win.x, win.y + 0 * win.h);
            verts[geom->count++] = !flip ? v2(win.x + 1 * win.w, win.y) : v2(win.x, win.y + 1 * win.h);
            AED->words.clear();
        }

        for (deque<AcousticEventDetector::Word>::iterator i = AED->words.begin(); i != AED->words.end(); i++) {
            bool fs=((*i).PS >= 0), fe=((*i).PE >= 0);
            double percent1=AED->percent((*i).PS), percent2=AED->percent((*i).PE);

            if (fs && percent1 < 0) fs = 0;
            if (fe && percent2 < 0) continue;

            if (fs) {
                verts[geom->count++] = !flip ? v2(win.x + percent1 * win.w, win.y)       : v2(win.x,       win.y + percent1 * win.h);
                verts[geom->count++] = !flip ? v2(win.x + percent1 * win.w, win.y+win.h) : v2(win.x+win.w, win.y + percent1 * win.h);
            }
            if (fe) {
                verts[geom->count++] = !flip ? v2(win.x + percent2 * win.w, win.y)       : v2(win.x,       win.y + percent2 * win.h);
                verts[geom->count++] = !flip ? v2(win.x + percent2 * win.w, win.y+win.h) : v2(win.x+win.w, win.y + percent2 * win.h);
            }
            if (fs && fe) {
                verts[geom->count++] = !flip ? v2(win.x + percent1 * win.w, win.y)       : v2(win.x+win.w, win.y + percent1 * win.h);
                verts[geom->count++] = !flip ? v2(win.x + percent2 * win.w, win.y)       : v2(win.x+win.w, win.y + percent2 * win.h);
            }
            else if (fs && !fe) {
                verts[geom->count++] = !flip ? v2(win.x + percent1 * win.w, win.y)       : v2(win.x+win.w, win.y + percent1 * win.h);
                verts[geom->count++] = !flip ? v2(win.x +        1 * win.w, win.y)       : v2(win.x+win.w, win.y + 1        * win.h);
            }
            else if (!fs && fe) {
                verts[geom->count++] = !flip ? v2(win.x +        0 * win.w, win.y)       : v2(win.x+win.w, win.y + 0        * win.h);
                verts[geom->count++] = !flip ? v2(win.x + percent2 * win.w, win.y)       : v2(win.x+win.w, win.y + percent2 * win.h);
            }
            else if (!fs && !fe && AED->onair()) {
                verts[geom->count++] = !flip ? v2(win.x +        0 * win.w, win.y)       : v2(win.x+win.w, win.y + 0 * win.h);
                verts[geom->count++] = !flip ? v2(win.x +        1 * win.w, win.y)       : v2(win.x+win.w, win.y + 1 * win.h);
            }
        }

        if (AED->sink) {
            for (int i=0; i<AED->sink->decode.size(); i++) {
                double perc = AED->percent(AED->sink->decode[i].beg);
                if (!i && perc < 0) { FeatureSink::DecodedWord w = PopFront(AED->sink->decode); i--; continue; }

                int tx, ty;
                if (!flip) { tx = win.x + perc * win.w; ty = win.centerY();        }
                else       { tx = win.centerX();        ty = win.y + perc * win.h; }

                static Font *text = Fonts::Get(FLAGS_default_font, "", 12, Color::white, Color::clear, FontDesc::Outline);
                text->Draw(AED->sink->decode[i].text, point(tx, ty), 0, Font::DrawFlag::Orientation(!flip ? 1 : 3));
            }
        }

        screen->gd->DisableTexture();
        Scene::Select(geom);
        Scene::Draw(geom, 0);
    }
};
#endif /* __LFL_LFAPP_GUI_H__ */

}; // namespace LFL
#endif // __LFL_LFAPP_AED_H__
