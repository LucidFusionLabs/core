/*
 * $Id: lfobjc.mm 1299 2011-12-11 21:39:12Z justin $
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

#include <stdlib.h>
#include "../lfexport.h"

#include <Foundation/NSThread.h>

#ifndef LFL_IPHONE
#include <AppKit/NSColor.h>
#include <AppKit/NSColorSpace.h>
#endif

@interface MyThread : NSObject {}
  @property int (*CB)(void*);
  @property void *arg;
@end

@implementation MyThread
  - (void) myMain { _CB(_arg); }
@end

extern "C" void NativeThreadStart(int (*CB)(void *), void *arg) {
  MyThread *thread = [MyThread alloc];
  [thread setCB:  CB];
  [thread setArg: arg];
  [NSThread detachNewThreadSelector:@selector(myMain) toTarget:thread withObject:nil];
}

extern "C" void ConvertColorFromGenericToDeviceRGB(const float *i, float *o) {
#ifdef LFL_IPHONE
#else
  double ib[4], ob[4], *ii = ib, *oi = ob;
  for (auto e = i + 4; i != e; ) *ii++ = *i++;
  NSColor *gen_color = [NSColor colorWithColorSpace:[NSColorSpace genericRGBColorSpace] components:ib count:4];
  NSColor *dev_color = [gen_color colorUsingColorSpaceName:NSDeviceRGBColorSpace];
  [dev_color getComponents: ob];
  for (auto e = o + 4; o != e; ) *o++ = *oi++;
#endif
}
