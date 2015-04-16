/*
 * $Id: voice.cpp 1330 2014-11-06 03:04:15Z justin $
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
#include "voice.h"

namespace LFL {
int VoiceModel::read(const char *dir) {
    string pre = "seg", post = ".mat";
    DirectoryIter d(dir, 0, pre.c_str(), post.c_str());
    for (const char *fn = d.Next(); fn; fn = d.Next()) {
        const char *nb=fn+pre.length(), *ne=strstr(nb,post.c_str());
        int phoneme = Phoneme::id(nb, ne?ne-nb:0);
        if (phoneme == -1) continue;

        string pn = dir; pn += fn;

        int samples = MatrixArchiveIn::Count(pn);
        unit[phoneme].samples = samples;
        unit[phoneme].sample = (Unit::Sample*)calloc(sizeof(Unit::Sample), samples);

        MatrixArchiveIn index(pn.c_str());
        Matrix *m=0; string hdr; int err, count=0;
        for (err = index.Read(&m, &hdr); err != -1; err = index.Read(&m, &hdr)) {
            int beg = m->row(0)[0], end = m->row(0)[1];
            unit[phoneme].sample[count].offset = beg;
            unit[phoneme].sample[count].len = end - beg;

            Replace<Matrix>(&m, 0);
            count++;
        }
        
        pn = pn.substr(0, pn.size() - post.size());
        pn += ".wav";
        unit[phoneme].wav.Open(new LocalFile(pn, "r"));
    }
    return 0;
};

RingBuf *VoiceModel::synth(const char *text, int start) {
    PronunciationDict *dict = PronunciationDict::instance();
    const char *w[1024], *wa[1024]; int words, phones;
    words = dict->pronounce(text, w, wa, &phones, 1024);
    if (words == -1) return 0;

    struct Atom { int phone, sample; } atom[1024];
    int atoms=0, samples=0, WordPause=FLAGS_sample_rate/100;

    for (int i=0; i<words; i++) {
        int len = strlen(w[i]);
        for (int j=0; j<len; j++) {
            int phone = w[i][j];
            int sample = /*(!i && !j) ? start : */ nextPhone(phone, atom[atoms-1].phone, atom[atoms-1].sample);
            if (sample == -1) return 0;
            
            atom[atoms].phone = phone;
            atom[atoms].sample = sample;
            atoms++;

            samples += unit[phone].sample[sample].len;
        }
        
        atom[atoms].phone = 0;
        atom[atoms].sample = 0;
        atoms++;

        samples += WordPause;
    }

    int wrote=0;
    RingBuf *ret = new RingBuf(FLAGS_sample_rate, samples);
    RingBuf::Handle h(ret);
    for (int i=0; i<atoms; i++) {
        if (atom[i].phone) {
            int phone = atom[i].phone;
            int sample = atom[i].sample;
            Unit::Sample *s = &unit[phone].sample[sample];
            if (unit[phone].wav.Read(&h, s->offset, s->len) < 0) {
                ERROR("error: synth ", Phoneme::name(phone), " ", s->offset, ", ", s->len);
                delete ret;
                return 0;
            }
            wrote += s->len;
        }
        else {
            for (int i=0; i<WordPause; i++) {
                h.Write(0.0);
                wrote++;
            }
        }
    }
    if (wrote != ret->ring.size) ERROR("VoiceModelSynth wrote ", wrote, " != ", ret->ring.size);

    return ret;
}

int VoiceModel::nextPhone(int phone, int lastphone, int lastphoneindex) {
    return rand() % unit[phone].samples;
}

}; // namespace LFL
