set(OSXCERT OSX Developer \(FFFFFFFFFF\))

# set(CMAKE_OSX_ARCHITECTURES i686 CACHE string "arch")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=10.9")
set(ENV{CFLAGS}   "-mmacosx-version-min=10.9")
set(ENV{CXXFLAGS} "-mmacosx-version-min=10.9")
set(ENV{LDFLAGS}  "-mmacosx-version-min=10.9")
add_definitions("-mmacosx-version-min=10.9")