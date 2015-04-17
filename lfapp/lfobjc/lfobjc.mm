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
