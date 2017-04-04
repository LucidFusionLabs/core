#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

@implementation ObjcCallback { LFL::Callback cb; }
  - (id)initWithCB:(LFL::Callback)v { self = [super init]; cb = move(v); return self; }
  - (void)run { if (cb) cb(); }
@end

@implementation ObjcStringCallback { LFL::StringCB cb; }
  - (id)initWithStringCB:(LFL::StringCB)v { self = [super init]; cb = move(v); return self; }
  - (void)run:(const LFL::string&)a { if (cb) cb(a); }
@end

@implementation ObjcFileHandleCallback
  {
    std::function<bool()> cb;
    id<ObjcWindow> win;
  }

  - (id)initWithCB:(std::function<bool()>)v forWindow:(id<ObjcWindow>)w fileDescriptor:(int)fd {
    self = [super init];
    cb = move(v);
    win = w;
    _fh = [[NSFileHandle alloc] initWithFileDescriptor:fd];
    [[NSNotificationCenter defaultCenter] addObserver:self
      selector:@selector(fileDataAvailable:) name:NSFileHandleDataAvailableNotification object:_fh];
    [_fh waitForDataInBackgroundAndNotify];
    return self;
  }

  - (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self
      name:NSFileHandleDataAvailableNotification object:_fh];
    [super dealloc];
  }

  - (void)fileDataAvailable: (NSNotification *)notification {
    [win objcWindowSelect];
    if (cb()) [win objcWindowFrame];
    [_fh waitForDataInBackgroundAndNotify];
  }
@end

@interface AppleURLSessions : NSObject<NSURLSessionDelegate, NSURLSessionTaskDelegate, NSURLSessionStreamDelegate> {}
@end

@implementation AppleURLSessions
  {
    std::unordered_map<void*, LFL::NSURLSessionStreamConnection*> task_map;
  }

  - (void)addTask:(void*)task withConnection:(LFL::NSURLSessionStreamConnection*)conn {
    task_map[task] = conn;
  }

  - (void)URLSession:(NSURLSession *)session didBecomeInvalidWithError:(NSError *)error {
    INFOf("AppleURLSessions::URLSession: %p didBecomeInvalidWithError: %s", session,
          LFL::GetNSString([error localizedDescription]).c_str());
  }

  - (void)URLSessionDidFinishEventsForBackgroundURLSession:(NSURLSession *)session {
    INFOf("AppleURLSessions::URLSessionDidFinishEventsForBackgroundURLSession %p", session);
  }

  - (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(NSError *)error {
    // INFOf("AppleURLSessions::URLSession: task: %p didCompleteWithError: %s", task, LFL::GetNSString([error localizedDescription]).c_str());
    LFL::app->RunInMainThread([=](){
      auto it = task_map.find(task);
      CHECK_NE(task_map.end(), it);
      delete it->second;
      task_map.erase(it);
    });
  }

#if 0
  - (void)URLSession:(NSURLSession *)session readClosedForStreamTask:(NSURLSessionStreamTask *)stream {
    INFOf("AppleURLSessions::URLSession readClosedForStreamTask: %p", stream);
  }

  - (void)URLSession:(NSURLSession *)session writeClosedForStreamTask:(NSURLSessionStreamTask *)stream {
    INFOf("AppleURLSessions::URLSession writeClosedForStreamTask: %p", stream);
  }
#endif
@end

@interface ObjcTimer : NSObject
@end

@implementation ObjcTimer
  {
    NSTimer *timer;
    NSTimeInterval timer_start;
    LFL::Callback cb;
  }

  - (id)initWithCB:(LFL::Callback)v { self = [super init]; cb = move(v); return self; }
  - (void)dealloc { [self clearTrigger]; [super dealloc]; }
  - (void)clearTrigger { if (timer) [timer invalidate]; timer = nil; }
  - (void)triggerFired:(id)sender { [self clearTrigger]; if (cb) cb(); }

  - (void)triggerIn:(int)ms force:(bool)force {
    NSTimeInterval now = [NSDate timeIntervalSinceReferenceDate];
    if (timer) {
      int remaining = [[timer userInfo] intValue] - (now - timer_start)*1000.0;
      if (remaining <= ms && !force) return;
      [self clearTrigger]; 
    }
    timer = [NSTimer timerWithTimeInterval: ms/1000.0
      target:self selector:@selector(triggerFired:) userInfo:[NSNumber numberWithInt:ms] repeats:NO];
    [[NSRunLoop currentRunLoop] addTimer:timer forMode:NSDefaultRunLoopMode];
    [timer retain];
    timer_start = now;
  }
@end

namespace LFL {
struct AppleTimer : public TimerInterface {
  ObjcTimer *timer;
  ~AppleTimer() { [timer release]; }
  AppleTimer(Callback c) : timer([[ObjcTimer alloc] initWithCB: move(c)]) {}
  void Run(Time interval, bool force=false) { [timer triggerIn:interval.count() force:force]; }
  void Clear() { [timer clearTrigger]; }
};

static AppleURLSessions *apple_url_sessions = 0;
NSURLSession* NSURLSessionStreamConnection::session = 0;

NSURLSessionStreamConnection::~NSURLSessionStreamConnection() {
  INFO(endpoint_name, ": deleting NSURLSessionStreamTask");
}

void NSURLSessionStreamConnection::Close() {
  INFO(endpoint_name, ": Close()");
  if (stream) {
    [stream closeWrite];
    [stream closeRead];
    [stream cancel];
    [stream release];
    stream = 0;
  }
}

NSURLSessionStreamConnection::NSURLSessionStreamConnection(const string &hostport, int default_port,
                                                           Connection::CB s_cb) : connected_cb(move(s_cb)) {
  if (!session) {
    apple_url_sessions = [[AppleURLSessions alloc] init];
    NSURLSessionConfiguration* config = [NSURLSessionConfiguration defaultSessionConfiguration];
    NSString *appid = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleIdentifier"];
    CHECK(config);
    config.shouldUseExtendedBackgroundIdleMode = TRUE;
    config.sharedContainerIdentifier = appid;
    session = [NSURLSession sessionWithConfiguration:config delegate:apple_url_sessions delegateQueue:nil];
    CHECK(session);
    [session retain];
  }
  int port;
  string host;
  LFL::HTTP::SplitHostAndPort(hostport, default_port, &host, &port);
  endpoint_name = StrCat(host, ":", port);
  stream = [session streamTaskWithHostName:LFL::MakeNSString(host) port:port];
  INFOf("%s: connecting NSURLSessionStreamTask: %p", endpoint_name.c_str(), stream);
  CHECK(stream);
  [stream retain];
  [stream resume];
  app->RunInMainThread([=](){ [apple_url_sessions addTask: stream withConnection: this]; });
  ReadableCB(true, false, false);
}

int NSURLSessionStreamConnection::WriteFlush(const char *buf, int len) {
  [stream writeData:LFL::MakeNSData(buf, len) timeout:0 completionHandler:^(NSError *error){
    if (!stream) return;
    if (!connected) app->RunInMainThread([=]() { if (!connected && (connected=1)) ConnectedCB(); });
    if (error) app->RunInMainThread([=](){ if (!done && (done=1)) { if (readable_cb()) app->scheduler.Wakeup(app->focused); } });
  }];
  return len;
}

void NSURLSessionStreamConnection::ConnectedCB() {
  INFO(endpoint_name, ": connected NSURLSessionStreamTask");
  state = Connection::Connected;
  if (handler) handler->Connected(this);
  if (connected_cb) connected_cb(this);
}

void NSURLSessionStreamConnection::ReadableCB(bool init, bool error, bool eof) {
  if (!stream) return;
  CHECK(!done);
  if (error) done = true;
  if (!init) {
    if (!connected && (connected=1)) ConnectedCB();
    if (!error) rb.Add(buf.data(), buf.size());
    if (readable_cb()) app->scheduler.Wakeup(app->focused);
  }
  buf.clear();
  if (done) return;
  if (eof) {
    done = true;
    if (readable_cb()) app->scheduler.Wakeup(app->focused);
    return;
  }
  [stream readDataOfMinLength:1 maxLength:65536 timeout:0 completionHandler:^(NSData *data, BOOL atEOF, NSError *error){
    if (!stream) return;
    if (!error) buf.append(static_cast<const char *>(data.bytes), data.length);
    else INFO(endpoint_name, ": ", GetNSString([error localizedDescription]));
    app->RunInMainThread(bind(&NSURLSessionStreamConnection::ReadableCB, this, false, error != nil, atEOF));
  }];
}

void NSLogString(const string &text) { NSLog(@"%@", MakeNSString(text)); }

string GetNSDocumentDirectory() {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  return GetNSString([paths objectAtIndex:0]);
}

NSArray *MakeNSStringArray(const vector<string> &v) {
  vector<NSString*> vs;
  for (auto &vi : v) vs.push_back(MakeNSString(vi));
  return [NSArray arrayWithObjects:&vs[0] count:vs.size()];
}

NSSet *MakeNSStringSet(const vector<string> &v) {
  vector<NSString*> vs;
  for (auto &vi : v) vs.push_back(MakeNSString(vi));
  return [NSSet setWithObjects:&vs[0] count:vs.size()];
}

void CleanUpTextureCGImageData(void *info, const void *data, size_t size) {
  delete [] static_cast<unsigned char *>(info);
}

CGImageRef MakeCGImage(Texture &t) {
  if (!t.buf) return nullptr;
  unsigned char *buf = t.ReleaseBuffer();
  CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
  CGDataProviderRef provider = 
    CGDataProviderCreateWithData(buf, buf, t.BufferSize(), CleanUpTextureCGImageData);
  auto ret = CGImageCreate(t.width, t.height, 8, 8*t.PixelSize(), t.LineSize(), rgb,
                           kCGBitmapByteOrderDefault /* | kCGImageAlphaLast */,
                           provider, nullptr, false, kCGRenderingIntentDefault);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(rgb); 
  return ret;
}

string Application::GetVersion() {
  NSDictionary *info = [[NSBundle mainBundle] infoDictionary];
  NSString *version = [info objectForKey:@"CFBundleVersion"];
  return version ? GetNSString(version) : "";
}

String16 Application::GetLocalizedString16(const char *key) { return String16(); }
string Application::GetLocalizedString(const char *key) {
  NSString *localized = 
    [[NSBundle mainBundle] localizedStringForKey: [NSString stringWithUTF8String: key] value:nil table:nil];
  if (!localized) return StrCat("<missing localized: ", key, ">");
  else            return [localized UTF8String];
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  static NSNumberFormatter *formatter=0;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    formatter = [[NSNumberFormatter alloc] init];
    [formatter setNumberStyle:NSNumberFormatterDecimalStyle];
    [formatter setMaximumFractionDigits:0];
    [formatter setMinimumIntegerDigits:1];
    [formatter setLocale:[NSLocale autoupdatingCurrentLocale]];
  });
  return [[formatter stringFromNumber: [NSNumber numberWithLong:number]] UTF8String];
}

void Application::LoadDefaultSettings(const StringPairVec &v) {
  NSMutableDictionary *defaults = [[NSMutableDictionary alloc] init];
  for (auto &i : v) [defaults setValue:MakeNSString(i.second) forKey:MakeNSString(i.first)];
  [[NSUserDefaults standardUserDefaults] registerDefaults:defaults];
  [defaults release];
}

string Application::GetSetting(const string &key) {
  return GetNSString([[NSUserDefaults standardUserDefaults] stringForKey:MakeNSString(key)]);
}

void Application::SaveSettings(const StringPairVec &v) {
  NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
  for (auto &i : v) [defaults setObject:MakeNSString(i.second) forKey:MakeNSString(i.first)];
  [defaults synchronize];
}

string Application::PrintCallStack() {
  string ret;
  NSArray *callstack = [NSThread callStackSymbols];
  for (NSString *symbol in callstack) StrAppend(&ret, GetNSString(symbol), "\n");
  return ret;
}

unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AppleTimer>(move(cb)); }

}; // namespace LFL
