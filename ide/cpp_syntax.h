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

#ifdef LFL_CPP_SYNTAX_STATEMENT
XX(new)
XX(delete)
XX(this)
XX(friend)
XX(using)
#endif

#ifdef LFL_CPP_SYNTAX_ACCESS
XX(public)
XX(protected)
XX(private)
#endif

#ifdef LFL_CPP_SYNTAX_TYPE
XX(inline)
XX(virtual)
XX(explicit)
XX(export)
XX(bool)
XX(wchar_t)
#endif

#ifdef LFL_CPP_SYNTAX_EXCEPTIONS
XX(throw)
XX(try)
XX(catch)
#endif

#ifdef LFL_CPP_SYNTAX_OPERATOR
XX(operator)
XX(typeid)
XX(and)
XX(bitor)
XX(or)
XX(xor)
XX(compl)
XX(bitand)
XX(and_eq)
XX(or_eq)
XX(xor_eq)
XX(not)
XX(not_eq)
#endif

#ifdef LFL_CPP_SYNTAX_CAST
XX(const_cast)
XX(static_cast)
XX(dynamic_cast)
XX(reinterpret_cast)
#endif

#ifdef LFL_CPP_SYNTAX_STORAGECLASS
XX(mutable)
#endif

#ifdef LFL_CPP_SYNTAX_STRUCTURE
XX(class)
XX(typename)
XX(template)
XX(namespace)
#endif

#ifdef LFL_CPP_SYNTAX_NUMBER
XX(NPOS)
#endif

#ifdef LFL_CPP_SYNTAX_BOOL
XX(true)
XX(false)
#endif
