# $Id: LFLOS.cmake 1335 2014-12-02 04:13:46Z justin $
# Copyright (C) 2009 Lucid Fusion Labs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

if(NOT LFL_OS)
  if(APPLE)
    set(LFL_OS osx)
  elseif(UNIX)
    set(LFL_OS linux)
  elseif(WIN32 OR WIN64)
    set(LFL_OS win32)
  else()
    MESSAGE(FATAL_ERROR "unknown OS")
  endif()
endif()
