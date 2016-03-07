/*
 * $Id: voice.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef LFL_SPEECH_VOICE_H__
#define LFL_SPEECH_VOICE_H__
namespace LFL {

/* unit database */
struct VoiceModel {
  struct Unit {
    struct Sample { int offset, len; } *sample;
    WavReader wav;
    int samples;
    Unit() : samples(0), sample(0) {}
    ~Unit() { delete wav.f; free(sample); }
  } unit[LFL_PHONES];

  int Read(const char *dir);
  RingSampler *Synth(const char *text, int start=0);

  int NextPhone(int phone, int lastphone, int lastphoneindex);
};

}; // namespace LFL
#endif // LFL_SPEECH_VOICE_H__
