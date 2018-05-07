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

XX(SIL) /* silence */

#ifdef LFL_LANGUAGE_ENGLISH
/* Phoneme  Example  Translation */  /* Phoneme  Example Translation */ 
/* -------  -------  ----------- */  /* -------  ------- ----------- */
XX(AA)   /* odd      AA D        */  XX(AE)   /* at      AE T        */
XX(AH)   /* hut      HH AH T     */  XX(AO)   /* ought   AO T        */
XX(AW)   /* cow      K AW        */  XX(AY)   /* hide    HH AY D     */
XX(B )   /* be       B IY        */  XX(CH)   /* cheese  CH IY Z     */
XX(D )   /* dee      D IY        */  XX(DH)   /* thee    DH IY       */
XX(EH)   /* Ed       EH D        */  XX(ER)   /* hurt    HH ER T     */
XX(EY)   /* ate      EY T        */  XX(F )   /* fee     F IY        */
XX(G )   /* green    G R IY N    */  XX(HH)   /* he      HH IY       */
XX(IH)   /* it       IH T        */  XX(IY)   /* eat     IY T        */ 
XX(JH)   /* gee      JH IY       */  XX(K )   /* key     K IY        */
XX(L )   /* lee      L IY        */  XX(M )   /* me      M IY        */
XX(N )   /* knee     N IY        */  XX(NG)   /* ping    P IH NG     */ 
XX(OW)   /* oat      OW T        */  XX(OY)   /* toy     T OY        */
XX(P )   /* pee      P IY        */  XX(R )   /* read    R IY D      */
XX(S )   /* sea      S IY        */  XX(SH)   /* she     SH IY       */
XX(T )   /* tea      T IY        */  XX(TH)   /* theta   TH EY T AH  */ 
XX(UH)   /* hood     HH UH D     */  XX(UW)   /* two     T UW        */
XX(V )   /* vee      V IY        */  XX(W )   /* we      W IY        */
XX(Y )   /* yield    Y IY L D    */  XX(Z )   /* zee     Z IY        */ 
XX(ZH)   /* seizure  S IY ZH ER  */

#undef  LFL_LAST_PHONE
#define LFL_LAST_PHONE ZH

#endif /* LFL_LANGUAGE_ENGLISH */

#undef LFL_PHONES
#define LFL_PHONES (Phoneme::LFL_LAST_PHONE+1)

