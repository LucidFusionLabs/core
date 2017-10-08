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

#include "core/app/app.h"
#include "core/app/ipc.h"

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitReader.h>
#include <llvm-c/BitWriter.h>

namespace LFL {
struct JIT {
  vector<LLVMModuleRef> module;
  vector<LLVMMemoryBufferRef> buffer;
  vector<string> buffer_data;
  LLVMExecutionEngineRef engine=0;
  bool lazy=false, own_module=true;
  const char* const* argv;
  int argc;

  JIT(int ac, const char* const *av) : argc(ac), argv(av) {}
  virtual ~JIT() {
    if (engine) LLVMDisposeExecutionEngine(engine);
    if (own_module) FreeModule();
    FreeBuffer();
  }

  void FreeModule() { for (auto &m : module) LLVMDisposeModule(m); module.clear(); }
  void FreeBuffer() { for (auto &b : buffer) LLVMDisposeMemoryBuffer(b); buffer.clear(); }

  bool LoadBitcode(const string &filename) {
    char *error = nullptr;
    if (SuffixMatch(filename, ".a")) {
      ArchiveIter archive(filename.c_str());
      for (const char *fn = archive.Next(); fn; fn = archive.Next()) {
        if (!SuffixMatch(fn, ".o") || !archive.LoadData()) continue;
        swap(PushBack(buffer_data, string()), archive.buf);
        const string &buf = buffer_data.back();
        if (auto b = LLVMCreateMemoryBufferWithMemoryRange
            (buf.data(), buf.size(), StrCat(filename, ":", fn).c_str(), false)) {
          buffer.push_back(b);
        } else buffer_data.pop_back();
      }
      if (!buffer.size()) { fprintf(stderr, "No .o in %s\n", filename.c_str()); return 0; }
    } else {
      if (LLVMCreateMemoryBufferWithContentsOfFile
          (filename.c_str(), &PushBack(buffer, LLVMMemoryBufferRef()), &error) || error)
      { fprintf(stderr, "LLVMCreateMemoryBufferWithContentsOfFile: %s\n", BlankNull(error)); return 0; }
    };

    module.resize(buffer.size());
    for (int i=0, l=module.size(); i != l; ++i) {
      if (lazy) {
        if (LLVMGetBitcodeModule(buffer[i], &module[i], &error) || error)
        { fprintf(stderr, "LLVMGetBitcodeModule: %s\n", BlankNull(error)); return 0; }
      } else {
        if (LLVMParseBitcode(buffer[i], &module[i], &error) || error)
        { fprintf(stderr, "LLVMParseBitcode: %s\n", BlankNull(error)); return 0; }
      }
    }

    // LLVMLinkInInterpreter();
    LLVMLinkInMCJIT();
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    own_module = false;
    if (LLVMCreateExecutionEngineForModule(&engine, module[0], &error) || error)
    { return fprintf(stderr, "LLVMCreateExecutionEngineForModule: %s", BlankNull(error)); return 0; }
    if (!lazy) FreeBuffer();
    return true;
  }

  int Run() {
    LLVMValueRef app_create = LLVMGetNamedFunction(module[0], "MyAppCreate");
    LLVMValueRef app_main   = LLVMGetNamedFunction(module[0], "MyAppMain");
    if (!app_create) { fprintf(stderr, "LLVMGetNamedFunction MyAppCreate"); return -1; }
    if (!app_main)   { fprintf(stderr, "LLVMGetNamedFunction MyAppMain");   return -1; }

    // LLVMRunStaticConstructors(engine);
    LLVMGenericValueRef rv = LLVMRunFunction(engine, app_create, 0, nullptr);
    fprintf(stderr, "MyAppCreate returns %p\n", rv);

    vector<const char*> av{ argv[0] };
    for (int i=2; i<argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);

    int ret = LLVMRunFunctionAsMain(engine, app_main, av.size()-1, av.data(), nullptr);
    fprintf(stderr, "MyAppMain returns %d\n", ret);
    return ret;
  }
};

unique_ptr<JIT> my_jit;

}; // namespace LFL
using namespace LFL;

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
  my_jit = make_unique<JIT>(argc, argv);
  return nullptr;
}

extern "C" int MyAppMain() {
  if (my_jit->argc < 2) { fprintf(stderr, "usage: %s <bitcode file> [args]\n", my_jit->argv[0]); return -1; }
  if (!my_jit->LoadBitcode(my_jit->argv[1])) return -1;
  return my_jit->Run();
}
