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

#ifdef LFL_GDDEBUG
#define GDDebug(...) { \
  CheckForError(__FILE__, __LINE__); \
  if (FLAGS_gd_debug) DebugPrintf("%s", StrCat(__VA_ARGS__).c_str()); }
#else 
#define GDDebug(...)
#endif

#if defined(LFL_GDDEBUG) || defined(LFL_GDLOGREF)
#define GDLogRef(...) { \
  if (app->focused) app->focused->gd->CheckForError(__FILE__, __LINE__); \
  if (FLAGS_gd_debug) DebugPrintf("%s", StrCat(__VA_ARGS__).c_str()); }
#else 
#define GDLogRef(...)
#endif
