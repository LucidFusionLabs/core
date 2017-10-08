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

#ifndef LFL_SPEECH_VOICE_H__
#define LFL_SPEECH_VOICE_H__
namespace LFL {

/* unit database */
struct VoiceModel {
  struct Unit {
    struct Sample { int offset=0, len=0; };
    vector<Sample> sample;
    WavReader wav;
  } unit[LFL_PHONES];

  int Read(const char *dir);
  unique_ptr<RingSampler> Synth(AssetLoading*, const char *text, int start=0);

  int NextPhone(int phone, int lastphone, int lastphoneindex);
};

}; // namespace LFL
#endif // LFL_SPEECH_VOICE_H__
