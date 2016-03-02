/*
 * $Id: bindings.cpp 1335 2014-12-02 04:13:46Z justin $
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

#include "core/app/app.h"
#include "core/web/dom.h"
#include "core/web/css.h"
#include "core/app/flow.h"
#include "core/app/gui.h"
#include "core/app/browser.h"

#ifdef LFL_CUDA
#include <cuda_runtime.h>
#include "lfcuda/lfcuda.h"
#include "speech/hmm.h"
#include "speech/speech.h"
#endif

extern "C" {
#ifdef LFL_LUA
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#endif
};

#ifdef LFL_V8JS
#include <v8.h>
#endif

namespace LFL {
#ifdef LFL_CUDA
void PrintCUDAProperties(cudaDeviceProp *prop) {
  DEBUGf("Major revision number:         %d", prop->major);
  DEBUGf("Minor revision number:         %d", prop->minor);
  DEBUGf("Name:                          %s", prop->name);
  DEBUGf("Total global memory:           %u", prop->totalGlobalMem);
  DEBUGf("Total shared memory per block: %u", prop->sharedMemPerBlock);
  DEBUGf("Total registers per block:     %d", prop->regsPerBlock);
  DEBUGf("Warp size:                     %d", prop->warpSize);
  DEBUGf("Maximum memory pitch:          %u", prop->memPitch);
  DEBUGf("Maximum threads per block:     %d", prop->maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of block: %d", i, prop->maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of grid:  %d", i, prop->maxGridSize[i]);
  DEBUGf("Clock rate:                    %d", prop->clockRate);
  DEBUGf("Total constant memory:         %u", prop->totalConstMem);
  DEBUGf("Texture alignment:             %u", prop->textureAlignment);
  DEBUGf("Concurrent copy and execution: %s", (prop->deviceOverlap ? "Yes" : "No"));
  DEBUGf("Number of multiprocessors:     %d", prop->multiProcessorCount);
  DEBUGf("Kernel execution timeout:      %s", (prop->kernelExecTimeoutEnabled ? "Yes" : "No"));
}

int CUDA::Init() {
  INFO("CUDA::Init()");
  FLAGS_lfapp_cuda = 0;

  int cuda_devices = 0;
  cudaError_t err;
  if ((err = cudaGetDeviceCount(&cuda_devices)) != cudaSuccess)
  { ERROR("cudaGetDeviceCount error ", cudaGetErrorString(err)); return 0; }

  cudaDeviceProp prop;
  for (int i=0; i<cuda_devices; i++) {
    if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) { ERROR("cudaGetDeviceProperties error ", err); return 0; }
    if (FLAGS_lfapp_debug) PrintCUDAProperties(&prop);
    if (strstr(prop.name, "Emulation")) continue;
    FLAGS_lfapp_cuda=1;
  }

  if (FLAGS_lfapp_cuda) {
    INFO("CUDA device detected, enabling acceleration: lfapp_cuda(", FLAGS_lfapp_cuda, ") devices ", cuda_devices);
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    cuda_init_hook();
  }
  else INFO("no CUDA devices detected ", cuda_devices);
  return 0;
}
#else
int CUDA::Init() { FLAGS_lfapp_cuda=0; INFO("CUDA not supported lfapp_cuda(", FLAGS_lfapp_cuda, ")"); return 0; }
#endif /* LFL_CUDA */

#ifdef LFL_LIBCLANG
}; // namespace LFL
#include "clang-c/Index.h"
namespace LFL {
typedef function<CXChildVisitResult(CXCursor)> ClangCursorVisitor;  
unsigned ClangVisitChildren(CXCursor p, ClangCursorVisitor f) {
  auto visitor_closure = [](CXCursor c, CXCursor p, CXClientData v) -> CXChildVisitResult {
    return (*reinterpret_cast<const ClangCursorVisitor*>(v))(c);
  };
  return clang_visitChildren(p, visitor_closure, &f);
}

string GetClangString(const CXString &s) {
  string v = BlankNull(clang_getCString(s));
  clang_disposeString(s);
  return v;
}

ClangTranslationUnit::ClangTranslationUnit(const string &f, const string &cc, const string &wd) :
  index(clang_createIndex(0, 0)), filename(f), compile_command(cc), working_directory(wd) {
  vector<string> argv;
  vector<const char*> av = { "-xc++", "-std=c++11" };
  Split(compile_command, isspace, &argv);
  for (int i=1; i<(int)argv.size()-4; i++) if (!PrefixMatch(argv[i], "-O") && !PrefixMatch(argv[i], "-m")) av.push_back(argv[i].data());
  chdir(working_directory.c_str());
  tu = clang_parseTranslationUnit(index, filename.c_str(), av.data(), av.size(), 0, 0, CXTranslationUnit_None);
}

ClangTranslationUnit::~ClangTranslationUnit() {
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
}

FileNameAndOffset ClangTranslationUnit::FindDefinition(const string &fn, int offset) {
  CXFile cf = clang_getFile(tu, fn.c_str());
  if (!cf) return FileNameAndOffset();
  CXCursor cursor = clang_getCursor(tu, clang_getLocationForOffset(tu, cf, offset));
  CXCursor cursor_canonical = clang_getCanonicalCursor(cursor), null = clang_getNullCursor();
  CXCursor parent = clang_getCursorSemanticParent(cursor), child = null;
  if (clang_equalCursors(parent, null)) return FileNameAndOffset();
  ClangVisitChildren(parent, [&](CXCursor c){
    if (!clang_equalCursors(clang_getCanonicalCursor(c), cursor_canonical)) return CXChildVisit_Continue;
    else { child = c; return CXChildVisit_Break; }
  });
  if (clang_equalCursors(child, null)) return FileNameAndOffset();
  CXFile rf;
  unsigned ry, rx, ro;
  CXSourceRange sr = clang_getCursorExtent(child);
  clang_getSpellingLocation(clang_getRangeStart(sr), &rf, &ry, &rx, &ro);
  string rfn = GetClangString(clang_getFileName(rf));
  return FileNameAndOffset(GetClangString(clang_getFileName(rf)), ro, ry, rx);
}

void ClangTokenVisitor::Visit() {
  CXToken* tokens=0;
  unsigned num_tokens=0, by=0, bx=0, ey=0, ex=0;
  CXFile cf = clang_getFile(tu->tu, tu->filename.c_str());
  CXSourceRange sr = clang_getRange(clang_getLocationForOffset(tu->tu, cf, 0),
                                    clang_getLocationForOffset(tu->tu, cf, LocalFile(tu->filename, "r").Size()));
  clang_tokenize(tu->tu, sr, &tokens, &num_tokens);
  for (int i = 0; i < num_tokens; i++) {
    sr = clang_getTokenExtent(tu->tu, tokens[i]);
    clang_getSpellingLocation(clang_getRangeStart(sr), NULL, &by, &bx, NULL);
    clang_getSpellingLocation(clang_getRangeEnd  (sr), NULL, &ey, &ex, NULL);

    if (1)                  CHECK_LE(last_token.y, (int)by);
    if (by == last_token.y) CHECK_LT(last_token.x, (int)bx);
    cb(this, clang_getTokenKind(tokens[i]), by, bx);
    last_token = point(bx, by);
  }
}
#else // LFL_LIBCLANG
ClangTranslationUnit::ClangTranslationUnit(const string &f, const string &cc, const string &wd) {}
ClangTranslationUnit::~ClangTranslationUnit() {}
FileNameAndOffset ClangTranslationUnit::FindDefinition(const string&, int) { return FileNameAndOffset(); }
#endif // LFL_LIBCLANG

#ifdef LFL_LUA
struct MyLuaContext : public LuaContext {
  lua_State *L;
  ~MyLuaContext() { lua_close(L); }
  MyLuaContext() : L(luaL_newstate()) {
    luaopen_base(L);
    luaopen_table(L);
    luaopen_io(L);
    luaopen_string(L);
    luaopen_math(L);
  }
  string Execute(const string &s) {
    if (luaL_loadbuffer(L, s.data(), s.size(), "MyLuaExec")) { ERROR("luaL_loadstring ", lua_tostring(L, -1)); return ""; }
    if (lua_pcall(L, 0, LUA_MULTRET, 0))                     { ERROR("lua_pcall ",       lua_tostring(L, -1)); return ""; }
    return "";
  }
};
unique_ptr<LuaContext> LuaContext::Create() { return make_unique<MyLuaContext>(); }
#else /* LFL_LUA */
unique_ptr<LuaContext> LuaContext::Create() { return 0; }
#endif /* LFL_LUA */

#ifdef LFL_V8JS
v8::Local<v8::String> NewV8String(v8::Isolate *I, const char  *s) { return v8::String::NewFromUtf8(I, s); }
v8::Local<v8::String> NewV8String(v8::Isolate *I, const short *s) { return v8::String::NewFromTwoByte(I, MakeUnsigned(s)); }
template <class X> inline X CastV8InternalFieldTo(v8::Local<v8::Object> &self, int field_index) {
  return static_cast<X>(v8::Local<v8::External>::Cast(self->GetInternalField(field_index))->Value());
}

#define GET_V8_SELF() \
  v8::Local<v8::Object> self = args.Holder(); \
  X *inst = CastV8InternalFieldTo<X*>(self, 1);

#define RETURN_V8_TYPE(X, type, ret) \
  GET_V8_SELF(); \
  args.GetReturnValue().Set(type(args.GetIsolate(), (ret)));

#define RETURN_V8_OBJECT(X, OT, ret) \
  v8::Local<v8::Object> self = args.Holder(); \
  auto val = (ret); \
  if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; } \
  MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0); \
  v8::Local<v8::Object> ret_obj = (js_context->*OT)->NewInstance(); \
  ret_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context)); \
  ret_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val)); \
  ret_obj->SetInternalField(2, v8::External::New(args.GetIsolate(), TypeId<decltype(val)>())); \
  args.GetReturnValue().Set(ret_obj);

#define GET_V8_PROPERTY_STRING() \
  string n = BlankNull(*v8::String::Utf8Value(name)); \
  if (n == "toString" || n == "valueOf" || n == "length" || n == "item") return;

template <typename X, int       (*f)(const X*)> void IntGetter   (v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) { RETURN_V8_TYPE(X, v8::Integer::New, f(inst)); }
template <typename X, DOMString (*f)(const X*)> void StringGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) { RETURN_V8_TYPE(X, NewV8String,      f(inst).c_str()); }

template <typename X, DOMString (*f)(const X*, int)> void IndexedStringFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
  if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
  RETURN_V8_TYPE(X, NewV8String, f(inst, args[0]->Int32Value()).c_str());
}
template <typename X, DOMString (*f)(const X*, int)> void IndexedStringProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
  RETURN_V8_TYPE(X, NewV8String, f(inst, index).c_str());
}
template <typename X, DOMString (*f)(const X*, const DOMString &)>
void NamedStringProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
  GET_V8_PROPERTY_STRING();
  RETURN_V8_TYPE(X, NewV8String, f(inst, DOMString(n)).c_str());
}
template <typename X, void (*f)(X*, const DOMString &, const DOMString&)>
void SetNamedStringProperty(v8::Local<v8::String> name, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<v8::Value> &args) {
  GET_V8_PROPERTY_STRING();
  GET_V8_SELF();
  string v = BlankNull(*v8::String::Utf8Value(value->ToString()));
  f(inst, n, v);
}

template <class X, int        (X::*Z)()                 const> int       CallIntMember          (const X *x)                     { return (x->*Z)(); }
template <class X, DOMString  (X::*Z)()                 const> DOMString CallStringMember       (const X *x)                     { return (x->*Z)(); }
template <class X, DOMString  (X::*Z)(int)              const> DOMString CallIndexedStringMember(const X *x, int i)              { return (x->*Z)(i); }
template <class X, DOMString  (X::*Z)(const DOMString&) const> DOMString CallNamedStringMember  (const X *x, const DOMString &n) { return (x->*Z)(n); }
template <class X, void (X::*Z)(const DOMString&, const DOMString&)> void CallNamedSetterMember(X *x, const DOMString &n, const DOMString &v) { return (x->*Z)(n, v); }

template <class X, class Y, Y  (X::*Z)                        > Y* GetObjectMember        (X *x)                     { return &(x->*Z); }
template <class X, class Y, Y  (DOM::Element::*Z)             > Y* GetObjectFromElement   (X *x)                     { return x->AsElement() ? &(x->AsElement()->*Z) : 0; }
template <class X, class Y, Y* (X::*Z)(int)              const> Y* CallIndexedObjectMember(X *x, int i)              { return (x->*Z)(i); }
template <class X, class Y, Y* (X::*Z)(const DOMString&) const> Y* CallNamedObjectMember  (X *x, const DOMString &n) { return (x->*Z)(n); }
template <class X> DOM::CSSStyleDeclaration* GetRenderStyle(X *x) { return x->render ? &x->render->style : 0; }
template <class X> DOM::CSSStyleDeclaration* GetInlineStyle(X *x) {
  if (!x->render) return 0;
  if (!x->render->inline_style.Computed()) x->render->ComputeStyle(x->ownerDocument->inline_style_context, &x->render->inline_style);
  return &x->render->inline_style;
}

struct MyV8JSInit { MyV8JSInit() { v8::V8::Initialize(); } };

struct MyV8JSContext : public JSContext {
  template <typename X, typename Y, Y* (*f)(X*), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void ObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1)));
  }
  template <typename X, typename Y, Y* (*f)(X*, int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void IndexedObjectFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), args[0]->Int32Value()));
  }
  template <typename X, typename Y, Y* (*f)(X*), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void ArgumentObjectFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    if (args.Length() < 1 || !args[0]->IsObject()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    v8::Local<v8::Object> arg_obj = args[0]->ToObject();
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(arg_obj, 1)));
  }
  template <typename X, typename Y, Y *(*f)(X*, int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void IndexedObjectProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), index));
  }
  template <typename X, typename Y, Y *(*f)(X*, const DOMString&), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void NamedObjectProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
    string v = BlankNull(*v8::String::Utf8Value(name));
    if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), DOMString(v)));
  }

  v8::Isolate*                   isolate;
  v8::Isolate::Scope             isolate_scope;
  v8::HandleScope                handle_scope;
  v8::Handle<v8::Context>        context;
  v8::Context::Scope             context_scope;
  v8::Handle<v8::ObjectTemplate> global, console, window, node, node_list, named_node_map, css_style_declaration;
  Console*                       js_console;

  virtual ~MyV8JSContext() {}
  MyV8JSContext(Console *C, DOM::Node *D) : isolate(v8::Isolate::New()), isolate_scope(isolate),
  handle_scope(isolate), context(v8::Context::New(isolate)), context_scope(context), global(v8::ObjectTemplate::New()),
  console(v8::ObjectTemplate::New()), window(v8::ObjectTemplate::New()), node(v8::ObjectTemplate::New()),
  node_list(v8::ObjectTemplate::New()), named_node_map(v8::ObjectTemplate::New()), css_style_declaration(v8::ObjectTemplate::New()),
  js_console(C) {
    console->SetInternalFieldCount(1);
    console->Set(v8::String::NewFromUtf8(isolate, "log"), v8::FunctionTemplate::New(isolate, consoleLog));
    v8::Local<v8::Object> console_obj = console->NewInstance();
    console_obj->SetInternalField(0, v8::External::New(isolate, this));
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);

    window->SetInternalFieldCount(1);
    window->Set(v8::String::NewFromUtf8(isolate, "getComputedStyle"), v8::FunctionTemplate::New
                (isolate, ArgumentObjectFunc<DOM::Node, DOM::CSSStyleDeclaration, &GetRenderStyle<DOM::Node>, &MyV8JSContext::css_style_declaration>));
    v8::Local<v8::Object> window_obj = window->NewInstance();
    window_obj->SetInternalField(0, v8::External::New(isolate, this));
    window_obj->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "window"), window_obj);

    node->SetInternalFieldCount(3);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeName"),
                      StringGetter<DOM::Node, CallStringMember<DOM::Node, &DOM::Node::nodeName>>, donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeValue"),
                      StringGetter<DOM::Node, CallStringMember<DOM::Node, &DOM::Node::nodeValue>>, donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "childNodes"),
                      ObjectGetter<DOM::Node, DOM::NodeList, &GetObjectMember<DOM::Node, DOM::NodeList, &DOM::Node::childNodes>, &MyV8JSContext::node_list>,
                      donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "attributes"), 
                      ObjectGetter<DOM::Node, DOM::NamedNodeMap, &GetObjectFromElement<DOM::Node, DOM::NamedNodeMap, &DOM::Element::attributes>, &MyV8JSContext::named_node_map>,
                      donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "style"),
                      ObjectGetter<DOM::Node, DOM::CSSStyleDeclaration, &GetInlineStyle<DOM::Node>, &MyV8JSContext::css_style_declaration>,
                      donothingSetter);
    node_list->SetInternalFieldCount(3);
    node_list->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                           IntGetter<DOM::NodeList, &CallIntMember<DOM::NodeList, &DOM::NodeList::length>>, donothingSetter);
    node_list->Set(v8::String::NewFromUtf8(isolate, "item"),
                   v8::FunctionTemplate::New(isolate, IndexedObjectFunc<DOM::NodeList, DOM::Node, 
                                             &CallIndexedObjectMember<DOM::NodeList, DOM::Node, &DOM::NodeList::item>, &MyV8JSContext::node>));
    node_list->SetIndexedPropertyHandler(IndexedObjectProperty<DOM::NodeList, DOM::Node,
                                         &CallIndexedObjectMember<DOM::NodeList, DOM::Node, &DOM::NodeList::item>, &MyV8JSContext::node>);

    named_node_map->SetInternalFieldCount(3);
    named_node_map->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                IntGetter<DOM::NamedNodeMap, &CallIntMember<DOM::NamedNodeMap, &DOM::NamedNodeMap::length>>, donothingSetter);
    named_node_map->Set(v8::String::NewFromUtf8(isolate, "item"),
                        v8::FunctionTemplate::New(isolate, IndexedObjectFunc<DOM::NamedNodeMap, DOM::Node, 
                                                  CallIndexedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::item>, &MyV8JSContext::node>));
    named_node_map->SetIndexedPropertyHandler(IndexedObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                              &CallIndexedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::item>, &MyV8JSContext::node>);
    named_node_map->SetNamedPropertyHandler(NamedObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                            &CallNamedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::getNamedItem>, &MyV8JSContext::node>);

    css_style_declaration->SetInternalFieldCount(3);
    css_style_declaration->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                       IntGetter<DOM::CSSStyleDeclaration, &CallIntMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::length>>, donothingSetter);
    css_style_declaration->Set(v8::String::NewFromUtf8(isolate, "item"),
                               v8::FunctionTemplate::New(isolate, IndexedStringFunc<DOM::CSSStyleDeclaration, CallIndexedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>>));
    css_style_declaration->SetIndexedPropertyHandler(IndexedStringProperty<DOM::CSSStyleDeclaration, &CallIndexedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>>);
    css_style_declaration->SetNamedPropertyHandler(   NamedStringProperty<DOM::CSSStyleDeclaration, &CallNamedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::getPropertyValue>>,
                                                   SetNamedStringProperty<DOM::CSSStyleDeclaration, &CallNamedSetterMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::setPropertyValue>>);

    if (D) {
      v8::Local<v8::Object> node_obj = node->NewInstance();
      node_obj->SetInternalField(0, v8::External::New(isolate, this));
      node_obj->SetInternalField(1, v8::External::New(isolate, D));
      node_obj->SetInternalField(2, v8::External::New(isolate, TypeId<DOM::Node*>()));
      context->Global()->Set(v8::String::NewFromUtf8(isolate, "document"), node_obj);
    }
  }
  string Execute(const string &s) {
    v8::Handle<v8::String> source = v8::String::NewFromUtf8(isolate, s.c_str());
    v8::Handle<v8::Script> script = v8::Script::Compile(source);
    { v8::TryCatch trycatch;
      v8::Handle<v8::Value> result = script->Run();
      if (!result.IsEmpty()) {
        if (result->IsObject() && js_console) {
          v8::Local<v8::Object> obj = result->ToObject();
          if (obj->InternalFieldCount() >= 3) {
            if (CastV8InternalFieldTo<Void>(obj, 2) == TypeId<DOM::Node*>()) {
              js_console->Write(CastV8InternalFieldTo<DOM::Node*>(obj, 1)->DebugString());
            }
          }
        }
      } else result = trycatch.Exception();
      return BlankNull(*v8::String::Utf8Value(result));
    }
  }
  static void donothingSetter(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<void>& args) {}
  static void windowGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    args.GetReturnValue().Set(args.Holder());
  }
  static void consoleLog(const v8::FunctionCallbackInfo<v8::Value> &args) {
    v8::Local<v8::Object> self = args.Holder(); string msg;
    MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
    for (int i=0; i < args.Length(); i++) StrAppend(&msg, BlankNull(*v8::String::Utf8Value(args[i]->ToString())));
    if (js_context->js_console) js_context->js_console->Write(msg);
    else INFO("VSJ8(", Void(js_context), ") console.log: ", msg);
    args.GetReturnValue().Set(v8::Null(args.GetIsolate()));
  };
};
unique_ptr<JSContext> JSContext::Create(Console *js_console, DOM::Node *doc) { Singleton<MyV8JSInit>::Get(); return make_unique<MyV8JSContext>(js_console, doc); }
#else /* LFL_V8JS */
unique_ptr<JSContext> JSContext::Create(Console *js_console, DOM::Node *doc) { return nullptr; }
#endif /* LFL_V8JS */
}; // namespace LFL
