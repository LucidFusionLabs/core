# $Id: CMakeLists.txt 1273 2011-11-25 03:16:33Z justin $

include_directories(. src)
add_library(sregex STATIC src/sregex/sre_palloc.c src/sregex/sre_regex.c src/sregex/sre_yyparser.c
            src/sregex/sre_regex_compiler.c src/sregex/sre_vm_bytecode.c src/sregex/sre_vm_thompson.c
            src/sregex/sre_vm_pike.c src/sregex/sre_capture.c src/sregex/sre_vm_thompson_jit.c)
